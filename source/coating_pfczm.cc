#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>

using namespace dealii;

namespace
{
  constexpr unsigned int dim = 2;

  enum Component : unsigned int
  {
    ux = 0,
    uy = 1,
    damage = 2,
    eta = 3
  };

  struct MaterialPointState
  {
    SymmetricTensor<2, 3> eps_p;
    SymmetricTensor<2, 3> eps_vp;
    double                eqp      = 0.0;
    double                anand_s  = 15.0;
    double                H_damage = 0.0;
    double                H_eta    = 0.0;
  };

  struct Parameters
  {
    double mesh_size         = 0.1;
    double final_time        = 1.0;
    double dt_initial        = 0.1;
    double dt_min            = 1e-4;
    unsigned int max_alt_it  = 15;
    double alt_tol           = 1e-6;
    double E_sub             = 210000.0;
    double nu_sub            = 0.3;
    double sigma_y0          = 250.0;
    double H_iso             = 1500.0;
    double E_coat            = 120000.0;
    double nu_coat           = 0.22;
    double anand_A           = 0.005;
    double anand_n           = 8.0;
    double anand_h0          = 40.0;
    double Gc_d              = 12.0;
    double l_d               = 0.08;
    double Gc_eta            = 8.0;
    double l_eta             = 0.06;
    double residual_stiffness = 1e-6;
    unsigned int output_every = 1;
  };

  class CoatingPFCZM
  {
  public:
    CoatingPFCZM(const Parameters &parameters);
    void run();

  private:
    void make_grid();
    void setup_system();
    void initialize_history();
    void build_constraints(double time);

    bool solve_increment(const double dt, const double time_target);
    bool solve_displacement(const double dt);
    bool solve_phase_fields(const double dt);

    void assemble_displacement_system();
    void assemble_damage_system();
    void assemble_eta_system();

    void output_results(const unsigned int step) const;
    double compute_reaction_force() const;
    double displacement_history(const double time) const;

    SymmetricTensor<2, 3> to_3d_strain(const Tensor<2, dim> &grad_u) const;
    SymmetricTensor<2, 3>
    substrate_stress_update(const SymmetricTensor<2, 3> &eps,
                            MaterialPointState           &state,
                            const double                  dt,
                            const double                  d,
                            const bool                    update_state) const;

    SymmetricTensor<2, 3>
    coating_stress_update(const SymmetricTensor<2, 3> &eps,
                          MaterialPointState           &state,
                          const double                  dt,
                          const double                  d,
                          const double                  eta_val,
                          const bool                    update_state) const;

    double positive_energy_density(const SymmetricTensor<2, 3> &sigma,
                                   const SymmetricTensor<2, 3> &eps) const;

    const Parameters      prm;
    Triangulation<dim>    triangulation;
    FESystem<dim>         fe;
    DoFHandler<dim>       dof_handler;

    AffineConstraints<double> constraints;
    SparsityPattern          sparsity_pattern;
    SparseMatrix<double>     system_matrix;
    Vector<double>           system_rhs;

    Vector<double> solution;
    Vector<double> old_solution;

    std::vector<std::vector<MaterialPointState>> qp_states;

    unsigned int n_q_points = 0;

    mutable ConditionalOStream pcout;
  };

  CoatingPFCZM::CoatingPFCZM(const Parameters &parameters)
    : prm(parameters)
    , fe(FE_Q<dim>(1), 2, FE_Q<dim>(1), 1, FE_Q<dim>(1), 1)
    , dof_handler(triangulation)
    , pcout(std::cout, true)
  {}

  void CoatingPFCZM::make_grid()
  {
    const Point<dim> p1(-10.0, -0.5);
    const Point<dim> p2(0.0, 1.2);

    const unsigned int nx = std::max(2u, static_cast<unsigned int>(std::ceil((p2[0] - p1[0]) / prm.mesh_size)));
    const unsigned int ny = std::max(2u, static_cast<unsigned int>(std::ceil((p2[1] - p1[1]) / prm.mesh_size)));

    GridGenerator::subdivided_hyper_rectangle(triangulation, {nx, ny}, p1, p2, true);

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const auto center = cell->center();
        if (center[1] <= 0.0)
          cell->set_material_id(0); // coating
        else
          cell->set_material_id(1); // substrate

        for (const auto f : cell->face_indices())
          {
            if (!cell->face(f)->at_boundary())
              continue;
            const auto fc = cell->face(f)->center();
            if (std::abs(fc[0]) < 1e-12)
              cell->face(f)->set_boundary_id(1); // symmetry x=0
            else if (std::abs(fc[1] - 1.2) < 1e-12)
              cell->face(f)->set_boundary_id(2); // loading edge
            else
              cell->face(f)->set_boundary_id(0);
          }
      }
  }

  void CoatingPFCZM::setup_system()
  {
    dof_handler.distribute_dofs(fe);

    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    const QGauss<dim> quadrature(fe.degree + 1);
    n_q_points = quadrature.size();

    initialize_history();
  }

  void CoatingPFCZM::initialize_history()
  {
    qp_states.clear();
    qp_states.resize(triangulation.n_active_cells(),
                     std::vector<MaterialPointState>(n_q_points));
  }

  void CoatingPFCZM::build_constraints(const double time)
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    const FEValuesExtractors::Scalar d_fe(2);
    const FEValuesExtractors::Scalar eta_fe(3);

    const FEValuesExtractors::Scalar ux_fe(0);
    VectorTools::interpolate_boundary_values(
      dof_handler,
      1,
      Functions::ZeroFunction<dim>(4),
      fe.component_mask(ux_fe));

    VectorTools::interpolate_boundary_values(
      dof_handler,
      1,
      Functions::ZeroFunction<dim>(4),
      fe.component_mask(d_fe));
    VectorTools::interpolate_boundary_values(
      dof_handler,
      1,
      Functions::ZeroFunction<dim>(4),
      fe.component_mask(eta_fe));

    class LoadingFunction : public Function<dim>
    {
    public:
      LoadingFunction(const double val)
        : Function<dim>(4)
        , v(val)
      {}
      double value(const Point<dim> &, const unsigned int component) const override
      {
        return component == 1 ? v : 0.0;
      }

    private:
      const double v;
    } loading(displacement_history(time));

    const FEValuesExtractors::Scalar uy_fe(1);
    VectorTools::interpolate_boundary_values(
      dof_handler,
      2,
      loading,
      fe.component_mask(uy_fe));

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto v : cell->vertex_indices())
        {
          const Point<dim> p = cell->vertex(v);
          if (!vertex_touched[cell->vertex_index(v)] &&
              std::abs(p[0] + 10.0) < 1e-10 && std::abs(p[1] + 0.5) < 1e-10)
            {
              for (unsigned int c = 0; c < 2; ++c)
                constraints.add_line(cell->vertex_dof_index(v, c));
              constraints.set_inhomogeneity(cell->vertex_dof_index(v, ux), 0.0);
              constraints.set_inhomogeneity(cell->vertex_dof_index(v, uy), 0.0);
              vertex_touched[cell->vertex_index(v)] = true;
            }
        }

    constraints.close();
  }

  SymmetricTensor<2, 3>
  CoatingPFCZM::to_3d_strain(const Tensor<2, dim> &grad_u) const
  {
    SymmetricTensor<2, 3> eps;
    eps[0][0] = grad_u[0][0];
    eps[1][1] = grad_u[1][1];
    eps[0][1] = 0.5 * (grad_u[0][1] + grad_u[1][0]);
    eps[2][2] = 0.0; // plane strain
    return eps;
  }

  SymmetricTensor<2, 3>
  CoatingPFCZM::substrate_stress_update(const SymmetricTensor<2, 3> &eps,
                                        MaterialPointState           &state,
                                        const double /*dt*/,
                                        const double d,
                                        const bool update_state) const
  {
    const double mu     = prm.E_sub / (2.0 * (1.0 + prm.nu_sub));
    const double lambda = prm.E_sub * prm.nu_sub /
                          ((1.0 + prm.nu_sub) * (1.0 - 2.0 * prm.nu_sub));

    const SymmetricTensor<2, 3> elastic_strain_trial = eps - state.eps_p;
    const double                tr                    = trace(elastic_strain_trial);

    SymmetricTensor<2, 3> sigma_trial = 2.0 * mu * elastic_strain_trial;
    for (unsigned int i = 0; i < 3; ++i)
      sigma_trial[i][i] += lambda * tr;

    const SymmetricTensor<2, 3> s_trial = deviator(sigma_trial);
    const double                seq_trial = std::sqrt(1.5) * s_trial.norm();
    const double                sigma_y   = prm.sigma_y0 + prm.H_iso * state.eqp;

    SymmetricTensor<2, 3> sigma = sigma_trial;

    if (seq_trial > sigma_y + 1e-12)
      {
        const double dgamma = (seq_trial - sigma_y) / (3.0 * mu + prm.H_iso);
        const auto   n_dev  = s_trial / s_trial.norm();
        sigma               = sigma_trial - 2.0 * mu * dgamma * std::sqrt(1.5) * n_dev;

        if (update_state)
          {
            state.eps_p += 1.5 * dgamma * n_dev;
            state.eqp += dgamma;
          }
      }

    const double g = (1.0 - d) * (1.0 - d) + prm.residual_stiffness;
    return g * sigma;
  }

  SymmetricTensor<2, 3>
  CoatingPFCZM::coating_stress_update(const SymmetricTensor<2, 3> &eps,
                                      MaterialPointState           &state,
                                      const double                  dt,
                                      const double                  d,
                                      const double                  eta_val,
                                      const bool                    update_state) const
  {
    const double mu     = prm.E_coat / (2.0 * (1.0 + prm.nu_coat));
    const double lambda = prm.E_coat * prm.nu_coat /
                          ((1.0 + prm.nu_coat) * (1.0 - 2.0 * prm.nu_coat));

    const SymmetricTensor<2, 3> elastic_strain = eps - state.eps_vp;
    const double                tr             = trace(elastic_strain);

    SymmetricTensor<2, 3> sigma = 2.0 * mu * elastic_strain;
    for (unsigned int i = 0; i < 3; ++i)
      sigma[i][i] += lambda * tr;

    const SymmetricTensor<2, 3> s = deviator(sigma);
    const double                seq = std::sqrt(1.5) * s.norm();

    const double s_eff = std::max(state.anand_s, 1e-8);
    const double rate  = prm.anand_A * std::pow(std::max(seq / s_eff, 0.0), prm.anand_n);

    if (update_state && seq > 0.0)
      {
        const auto flow_dir = s / s.norm();
        const double dgamma = rate * dt;
        state.eps_vp += 1.5 * dgamma * flow_dir;
        state.anand_s += prm.anand_h0 * (1.0 - state.anand_s / 200.0) * dgamma;
      }

    const double g = ((1.0 - d) * (1.0 - d) + prm.residual_stiffness) * eta_val;
    return g * sigma;
  }

  double CoatingPFCZM::positive_energy_density(const SymmetricTensor<2, 3> &sigma,
                                               const SymmetricTensor<2, 3> &eps) const
  {
    return 0.5 * scalar_product(sigma, eps);
  }

  void CoatingPFCZM::assemble_displacement_system()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature,
                            update_values | update_gradients | update_JxW_values);

    const FEValuesExtractors::Scalar d_fe(2);
    const FEValuesExtractors::Scalar eta_fe(3);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q            = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim>> grad_u_values(n_q);
    std::vector<double>         d_values(n_q);
    std::vector<double>         eta_values(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        fe_values[u_fe].get_function_gradients(solution, grad_u_values);
        fe_values[d_fe].get_function_values(solution, d_values);
        fe_values[eta_fe].get_function_values(solution, eta_values);

        auto &states = qp_states[cell->active_cell_index()];

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const auto eps3d = to_3d_strain(grad_u_values[q]);

            MaterialPointState tmp_state = states[q];

            SymmetricTensor<2, 3> sigma;
            if (cell->material_id() == 1)
              sigma = substrate_stress_update(eps3d, tmp_state, 0.0, d_values[q], false);
            else
              sigma = coating_stress_update(eps3d,
                                            tmp_state,
                                            0.0,
                                            d_values[q],
                                            std::max(eta_values[q], 0.0),
                                            false);

            const double mu = (cell->material_id() == 1) ?
                                prm.E_sub / (2.0 * (1.0 + prm.nu_sub)) :
                                prm.E_coat / (2.0 * (1.0 + prm.nu_coat));
            const double nu = (cell->material_id() == 1) ? prm.nu_sub : prm.nu_coat;
            const double lambda = 2.0 * mu * nu / (1.0 - 2.0 * nu);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe.system_to_component_index(i).first >= 2)
                  continue;

                const Tensor<2, dim> grad_ni = fe_values[u_fe].gradient(i, q);
                const auto eps_i = to_3d_strain(grad_ni);

                cell_rhs(i) -= scalar_product(sigma, eps_i) * fe_values.JxW(q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    if (fe.system_to_component_index(j).first >= 2)
                      continue;
                    const Tensor<2, dim> grad_nj = fe_values[u_fe].gradient(j, q);
                    const auto eps_j = to_3d_strain(grad_nj);
                    const auto Cepsj = 2.0 * mu * eps_j + lambda * trace(eps_j) * unit_symmetric_tensor<3>();
                    cell_matrix(i, j) += scalar_product(eps_i, Cepsj) * fe_values.JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }
  }

  void CoatingPFCZM::assemble_damage_system()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature,
                            update_values | update_gradients | update_JxW_values);

    const FEValuesExtractors::Scalar d_fe(2);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q            = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim>> grad_u_values(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        fe_values[FEValuesExtractors::Vector(0)].get_function_gradients(solution,
                                                                        grad_u_values);

        auto &states = qp_states[cell->active_cell_index()];

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const auto eps3d = to_3d_strain(grad_u_values[q]);

            MaterialPointState tmp_state = states[q];
            SymmetricTensor<2, 3> sigma;
            if (cell->material_id() == 1)
              sigma = substrate_stress_update(eps3d, tmp_state, 0.0, 0.0, false);
            else
              sigma = coating_stress_update(eps3d, tmp_state, 0.0, 0.0, 1.0, false);

            const double Wp = std::max(positive_energy_density(sigma, eps3d), 0.0);
            states[q].H_damage = std::max(states[q].H_damage, Wp);

            const double coeff_m = prm.Gc_d / prm.l_d + 2.0 * states[q].H_damage;
            const double rhs_v   = 2.0 * states[q].H_damage;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe.system_to_component_index(i).first != damage)
                  continue;
                const double        Ni    = fe_values[d_fe].value(i, q);
                const Tensor<1, dim> gradNi = fe_values[d_fe].gradient(i, q);

                cell_rhs(i) += rhs_v * Ni * fe_values.JxW(q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    if (fe.system_to_component_index(j).first != damage)
                      continue;
                    const double Nj = fe_values[d_fe].value(j, q);
                    const Tensor<1, dim> gradNj = fe_values[d_fe].gradient(j, q);

                    cell_matrix(i, j) +=
                      (prm.Gc_d * prm.l_d * (gradNi * gradNj) + coeff_m * Ni * Nj) *
                      fe_values.JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }
  }

  void CoatingPFCZM::assemble_eta_system()
  {
    system_matrix = 0;
    system_rhs    = 0;

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(fe,
                            quadrature,
                            update_values | update_gradients | update_JxW_values);

    const FEValuesExtractors::Scalar eta_fe(3);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q            = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<Tensor<2, dim>> grad_u_values(n_q);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0;
        cell_rhs    = 0;
        fe_values.reinit(cell);

        fe_values[FEValuesExtractors::Vector(0)].get_function_gradients(solution,
                                                                        grad_u_values);

        auto &states = qp_states[cell->active_cell_index()];

        for (unsigned int q = 0; q < n_q; ++q)
          {
            const auto eps3d = to_3d_strain(grad_u_values[q]);
            MaterialPointState tmp_state = states[q];

            const auto sigma = coating_stress_update(eps3d, tmp_state, 0.0, 0.0, 1.0, false);
            const double drive = (cell->material_id() == 0) ?
                                   std::max(std::sqrt(1.5) * deviator(sigma).norm() - 30.0,
                                            0.0) :
                                   0.0;
            states[q].H_eta = std::max(states[q].H_eta, drive);

            const double coeff_m = prm.Gc_eta / prm.l_eta + states[q].H_eta;
            const double rhs_v   = states[q].H_eta;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                if (fe.system_to_component_index(i).first != eta)
                  continue;

                const double Ni = fe_values[eta_fe].value(i, q);
                const Tensor<1, dim> gradNi = fe_values[eta_fe].gradient(i, q);

                cell_rhs(i) += rhs_v * Ni * fe_values.JxW(q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  {
                    if (fe.system_to_component_index(j).first != eta)
                      continue;
                    const double Nj = fe_values[eta_fe].value(j, q);
                    const Tensor<1, dim> gradNj = fe_values[eta_fe].gradient(j, q);
                    cell_matrix(i, j) +=
                      (prm.Gc_eta * prm.l_eta * (gradNi * gradNj) + coeff_m * Ni * Nj) *
                      fe_values.JxW(q);
                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               system_matrix,
                                               system_rhs);
      }
  }

  bool CoatingPFCZM::solve_displacement(const double dt)
  {
    (void)dt;
    assemble_displacement_system();

    SolverControl solver_control(5000, 1e-10 * system_rhs.l2_norm() + 1e-14);
    SolverCG<Vector<double>> solver(solver_control);

    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    Vector<double> delta(solution.size());
    solver.solve(system_matrix, delta, system_rhs, preconditioner);
    constraints.distribute(delta);
    solution += delta;

    return solver_control.last_check() == SolverControl::success;
  }

  bool CoatingPFCZM::solve_phase_fields(const double dt)
  {
    (void)dt;

    Vector<double> old_d(solution);

    assemble_damage_system();
    SolverControl solver_control_d(5000, 1e-10 * system_rhs.l2_norm() + 1e-14);
    SolverCG<Vector<double>> solver_d(solver_control_d);
    PreconditionSSOR<SparseMatrix<double>> pre_d;
    pre_d.initialize(system_matrix, 1.2);

    Vector<double> new_d(solution.size());
    solver_d.solve(system_matrix, new_d, system_rhs, pre_d);
    constraints.distribute(new_d);

    for (unsigned int i = 0; i < solution.size(); ++i)
      {
        if (fe.system_to_component_index(i).first == damage)
          {
            const double lower = old_solution[i];
            solution[i] = std::min(1.0, std::max(lower, new_d[i]));
          }
      }

    assemble_eta_system();
    SolverControl solver_control_eta(5000, 1e-10 * system_rhs.l2_norm() + 1e-14);
    SolverCG<Vector<double>> solver_eta(solver_control_eta);
    PreconditionSSOR<SparseMatrix<double>> pre_eta;
    pre_eta.initialize(system_matrix, 1.2);

    Vector<double> new_eta(solution.size());
    solver_eta.solve(system_matrix, new_eta, system_rhs, pre_eta);
    constraints.distribute(new_eta);

    for (unsigned int i = 0; i < solution.size(); ++i)
      {
        if (fe.system_to_component_index(i).first == eta)
          solution[i] = std::min(1.0, std::max(0.0, new_eta[i]));
      }

    return solver_control_d.last_check() == SolverControl::success &&
           solver_control_eta.last_check() == SolverControl::success;
  }

  bool CoatingPFCZM::solve_increment(const double dt, const double time_target)
  {
    build_constraints(time_target);

    Vector<double> trial_old = solution;
    auto           qp_backup = qp_states;

    bool converged = false;

    for (unsigned int alt = 0; alt < prm.max_alt_it; ++alt)
      {
        const Vector<double> iterate_start = solution;

        if (!solve_displacement(dt))
          {
            solution  = trial_old;
            qp_states = qp_backup;
            return false;
          }

        if (!solve_phase_fields(dt))
          {
            solution  = trial_old;
            qp_states = qp_backup;
            return false;
          }

        Vector<double> diff(solution);
        diff -= iterate_start;
        if (diff.l2_norm() < prm.alt_tol)
          {
            converged = true;
            break;
          }
      }

    if (!converged)
      {
        solution  = trial_old;
        qp_states = qp_backup;
        return false;
      }

    const QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim>     fe_values(fe,
                            quadrature,
                            update_gradients | update_values | update_JxW_values);
    const FEValuesExtractors::Scalar d_fe(2);
    const FEValuesExtractors::Scalar eta_fe(3);
    std::vector<Tensor<2, dim>> grad_u_values(n_q_points);
    std::vector<double>         d_values(n_q_points);
    std::vector<double>         eta_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        fe_values[FEValuesExtractors::Vector(0)].get_function_gradients(solution,
                                                                        grad_u_values);
        fe_values[d_fe].get_function_values(solution, d_values);
        fe_values[eta_fe].get_function_values(solution, eta_values);

        auto &states = qp_states[cell->active_cell_index()];
        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const auto eps3d = to_3d_strain(grad_u_values[q]);
            if (cell->material_id() == 1)
              {
                (void)substrate_stress_update(eps3d, states[q], dt, d_values[q], true);
              }
            else
              {
                (void)coating_stress_update(eps3d,
                                            states[q],
                                            dt,
                                            d_values[q],
                                            std::max(eta_values[q], 0.0),
                                            true);
              }
          }
      }

    return true;
  }

  double CoatingPFCZM::compute_reaction_force() const
  {
    QGauss<dim - 1> face_quadrature(fe.degree + 1);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature,
                                     update_values | update_gradients |
                                       update_normal_vectors | update_JxW_values);

    const FEValuesExtractors::Scalar d_fe(2);
    const FEValuesExtractors::Scalar eta_fe(3);
    std::vector<Tensor<2, dim>> grad_u(face_quadrature.size());
    std::vector<double> d_values(face_quadrature.size());
    std::vector<double> eta_values(face_quadrature.size());

    double reaction = 0.0;

    for (const auto &cell : dof_handler.active_cell_iterators())
      for (const auto f : cell->face_indices())
        {
          if (!cell->face(f)->at_boundary() || cell->face(f)->boundary_id() != 2)
            continue;

          fe_face_values.reinit(cell, f);
          fe_face_values[FEValuesExtractors::Vector(0)].get_function_gradients(solution,
                                                                                grad_u);
          fe_face_values[d_fe].get_function_values(solution, d_values);
          fe_face_values[eta_fe].get_function_values(solution, eta_values);

          const auto &states = qp_states[cell->active_cell_index()];

          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
              MaterialPointState tmp_state = states[q % n_q_points];
              const auto eps3d = to_3d_strain(grad_u[q]);
              SymmetricTensor<2, 3> sigma;
              if (cell->material_id() == 1)
                sigma = substrate_stress_update(eps3d, tmp_state, 0.0, d_values[q], false);
              else
                sigma = coating_stress_update(eps3d,
                                              tmp_state,
                                              0.0,
                                              d_values[q],
                                              std::max(eta_values[q], 0.0),
                                              false);

              Tensor<1, dim> traction;
              const Tensor<1, dim> n = fe_face_values.normal_vector(q);
              traction[0]            = sigma[0][0] * n[0] + sigma[0][1] * n[1];
              traction[1]            = sigma[1][0] * n[0] + sigma[1][1] * n[1];

              reaction += traction[1] * fe_face_values.JxW(q);
            }
        }

    return reaction;
  }

  void CoatingPFCZM::output_results(const unsigned int step) const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> names = {"ux", "uy", "damage", "eta"};
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation = {
        DataComponentInterpretation::component_is_part_of_vector,
        DataComponentInterpretation::component_is_part_of_vector,
        DataComponentInterpretation::component_is_scalar,
        DataComponentInterpretation::component_is_scalar};

    data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, interpretation);

    Vector<float> eqp(triangulation.n_active_cells());
    Vector<float> anand_s(triangulation.n_active_cells());
    Vector<float> H(triangulation.n_active_cells());

    for (const auto &cell : triangulation.active_cell_iterators())
      {
        const auto &states = qp_states[cell->active_cell_index()];
        double eqp_sum = 0.0, anand_sum = 0.0, H_sum = 0.0;
        for (const auto &s : states)
          {
            eqp_sum += s.eqp;
            anand_sum += s.anand_s;
            H_sum += s.H_damage;
          }
        eqp[cell->active_cell_index()]     = eqp_sum / states.size();
        anand_s[cell->active_cell_index()] = anand_sum / states.size();
        H[cell->active_cell_index()]       = H_sum / states.size();
      }

    data_out.add_data_vector(eqp, "eqp");
    data_out.add_data_vector(anand_s, "anand_s");
    data_out.add_data_vector(H, "H_damage");

    data_out.build_patches();

    std::ofstream out("solution-" + Utilities::int_to_string(step, 4) + ".vtu");
    data_out.write_vtu(out);
  }

  double CoatingPFCZM::displacement_history(const double time) const
  {
    if (time <= 0.5)
      return -0.02 * (time / 0.5); // Step-5
    return -0.02 - 0.03 * ((time - 0.5) / 0.5); // Step-6
  }

  void CoatingPFCZM::run()
  {
    make_grid();
    setup_system();

    std::ofstream force_disp("force_displacement.csv");
    force_disp << "step,time,imposed_displacement,reaction_force\n";

    output_results(0);

    double       time = 0.0;
    unsigned int step = 0;
    double       dt   = prm.dt_initial;

    while (time < prm.final_time - 1e-14)
      {
        const double remaining = prm.final_time - time;
        dt = std::min(dt, remaining);

        bool accepted = false;
        while (!accepted)
          {
            old_solution = solution;
            const double t_new = time + dt;

            pcout << "Attempt step " << (step + 1) << " with dt=" << dt << std::endl;
            accepted = solve_increment(dt, t_new);

            if (!accepted)
              {
                dt *= 0.5;
                solution = old_solution;
                if (dt < prm.dt_min)
                  throw std::runtime_error("Increment cut-back reached dt_min; abort.");
              }
          }

        time += dt;
        ++step;

        const double reaction = compute_reaction_force();
        force_disp << step << ',' << time << ',' << displacement_history(time) << ','
                   << reaction << '\n';

        if (step % prm.output_every == 0)
          output_results(step);
      }

    pcout << "Finished. Steps: " << step << std::endl;
  }

  Parameters parse_parameters(int argc, char **argv)
  {
    Parameters prm;
    for (int i = 1; i < argc; ++i)
      {
        const std::string arg = argv[i];
        const auto split      = arg.find('=');
        if (split == std::string::npos)
          continue;
        const std::string key   = arg.substr(0, split);
        const std::string value = arg.substr(split + 1);

        if (key == "--mesh")
          prm.mesh_size = std::stod(value);
        else if (key == "--final-time")
          prm.final_time = std::stod(value);
        else if (key == "--dt")
          prm.dt_initial = std::stod(value);
        else if (key == "--output-every")
          prm.output_every = std::stoul(value);
      }
    return prm;
  }
} // namespace

int main(int argc, char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      const auto prm = parse_parameters(argc, argv);
      CoatingPFCZM solver(prm);
      solver.run();
    }
  catch (const std::exception &exc)
    {
      std::cerr << "Error: " << exc.what() << std::endl;
      return 1;
    }
  return 0;
}
