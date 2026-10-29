#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace PF
{
  using namespace dealii;

  struct Parameters
  {
    double L = 1.0;
    double fiber_center_x = 0.5;
    double fiber_center_y = 0.5;
    double fiber_radius = 0.25;
    double interface_half_thickness = 0.02;

    double E_matrix = 3.0e3;
    double nu_matrix = 0.35;
    double E_fiber  = 8.0e3;
    double nu_fiber = 0.20;

    double Gc = 3.0;
    double ft = 30.0;
    double l0 = 0.02;
    double kappa = 1e-8;

    unsigned int global_refinement = 5;
    unsigned int n_steps_5 = 100;
    unsigned int n_steps_6 = 500;
    double step5_u = 0.0025;
    double step6_u = 0.0060;

    unsigned int max_alt_iters = 40;
    double alt_tol = 1e-8;
    unsigned int output_stride = 10;

    bool smoke = false;
  };

  class RightDisplacement : public Function<2>
  {
  public:
    explicit RightDisplacement(const double ux)
      : Function<2>(2)
      , ux_value(ux)
    {}

    double value(const Point<2> &, const unsigned int component = 0) const override
    {
      return component == 0 ? ux_value : 0.0;
    }

  private:
    const double ux_value;
  };

  class FiberPFCZM
  {
  public:
    FiberPFCZM(const Parameters &par, const std::string &output_dir);
    void run();

  private:
    void setup_mesh();
    void setup_systems();

    void assemble_and_solve_u(const double right_disp);
    void update_history_field();
    void assemble_and_solve_d();

    void output_results(const unsigned int step) const;
    double compute_reaction_force_x() const;
    double prescribed_displacement(const unsigned int step) const;

    SymmetricTensor<4, 2> elasticity_tensor(const bool in_fiber,
                                            const double d_val) const;

    Parameters prm;
    std::filesystem::path out_dir;

    Triangulation<2> triangulation;

    FESystem<2> fe_u;
    DoFHandler<2> dof_u;

    FE_Q<2> fe_d;
    DoFHandler<2> dof_d;

    AffineConstraints<double> constraints_u;

    SparsityPattern sparsity_u;
    SparseMatrix<double> matrix_u;
    Vector<double> solution_u;
    Vector<double> rhs_u;

    SparsityPattern sparsity_d;
    SparseMatrix<double> matrix_d;
    Vector<double> solution_d;
    Vector<double> old_solution_d;
    Vector<double> rhs_d;

    std::vector<double> eta_nodal;
    std::vector<double> history_nodal;

    std::ofstream force_disp_csv;
  };

  FiberPFCZM::FiberPFCZM(const Parameters &par, const std::string &output_dir)
    : prm(par)
    , out_dir(output_dir)
    , fe_u(FE_Q<2>(1), 2)
    , dof_u(triangulation)
    , fe_d(1)
    , dof_d(triangulation)
  {}

  SymmetricTensor<4, 2>
  FiberPFCZM::elasticity_tensor(const bool in_fiber, const double d_val) const
  {
    const double E = in_fiber ? prm.E_fiber : prm.E_matrix;
    const double nu = in_fiber ? prm.nu_fiber : prm.nu_matrix;

    const double lambda = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu));
    const double mu = E / (2.0 * (1.0 + nu));

    const double g = (1.0 - d_val) * (1.0 - d_val) + prm.kappa;
    return g * (lambda * outer_product(unit_symmetric_tensor<2>(),
                                       unit_symmetric_tensor<2>()) +
                2.0 * mu * identity_tensor<2>());
  }

  void FiberPFCZM::setup_mesh()
  {
    GridGenerator::subdivided_hyper_cube(triangulation, 8, 0.0, prm.L);
    triangulation.refine_global(prm.global_refinement);

    for (auto &cell : triangulation.active_cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary())
        {
          const auto c = cell->face(f)->center();
          if (std::abs(c[0]) < 1e-12)
            cell->face(f)->set_boundary_id(1);
          else if (std::abs(c[0] - prm.L) < 1e-12)
            cell->face(f)->set_boundary_id(2);
          else
            cell->face(f)->set_boundary_id(0);
        }
  }

  void FiberPFCZM::setup_systems()
  {
    dof_u.distribute_dofs(fe_u);
    dof_d.distribute_dofs(fe_d);

    DoFRenumbering::Cuthill_McKee(dof_u);
    DoFRenumbering::Cuthill_McKee(dof_d);

    constraints_u.clear();
    constraints_u.close();

    DynamicSparsityPattern dsp_u(dof_u.n_dofs());
    DoFTools::make_sparsity_pattern(dof_u, dsp_u, constraints_u, false);
    sparsity_u.copy_from(dsp_u);
    matrix_u.reinit(sparsity_u);
    solution_u.reinit(dof_u.n_dofs());
    rhs_u.reinit(dof_u.n_dofs());

    DynamicSparsityPattern dsp_d(dof_d.n_dofs());
    DoFTools::make_sparsity_pattern(dof_d, dsp_d);
    sparsity_d.copy_from(dsp_d);
    matrix_d.reinit(sparsity_d);
    solution_d.reinit(dof_d.n_dofs());
    old_solution_d.reinit(dof_d.n_dofs());
    rhs_d.reinit(dof_d.n_dofs());

    eta_nodal.assign(dof_d.n_dofs(), 0.0);
    history_nodal.assign(dof_d.n_dofs(), 0.0);

    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_d, support_points);
    for (const auto &[i, p] : support_points)
    {
      const double dx = p[0] - prm.fiber_center_x;
      const double dy = p[1] - prm.fiber_center_y;
      const double r = std::sqrt(dx * dx + dy * dy);
      const double interface_dist = std::abs(r - prm.fiber_radius);
      eta_nodal[i] = interface_dist <= prm.interface_half_thickness ?
                       1.0 - interface_dist / prm.interface_half_thickness : 0.0;
    }

    force_disp_csv.open(out_dir / "force_displacement.csv", std::ios::trunc);
    force_disp_csv << "step,ux_right,reaction_fx\n";
  }

  void FiberPFCZM::assemble_and_solve_u(const double right_disp)
  {
    constraints_u.clear();

    const FEValuesExtractors::Scalar ux(0);

    VectorTools::interpolate_boundary_values(dof_u,
                                             1,
                                             Functions::ZeroFunction<2>(2),
                                             constraints_u,
                                             fe_u.component_mask(ux));

    RightDisplacement right_bc(right_disp);
    VectorTools::interpolate_boundary_values(dof_u,
                                             2,
                                             right_bc,
                                             constraints_u,
                                             fe_u.component_mask(ux));

    bool pin_applied = false;
    for (const auto &cell : dof_u.active_cell_iterators())
      for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        if (std::abs(cell->vertex(v)[0]) < 1e-12 &&
            std::abs(cell->vertex(v)[1]) < 1e-12)
        {
          constraints_u.add_line(cell->vertex_dof_index(v, 1));
          constraints_u.set_inhomogeneity(cell->vertex_dof_index(v, 1), 0.0);
          pin_applied = true;
          break;
        }

    AssertThrow(pin_applied, ExcMessage("Failed to pin uy at corner point (0,0)."));

    constraints_u.close();

    matrix_u = 0;
    rhs_u = 0;

    QGauss<2> quadrature(fe_u.degree + 1);
    FEValues<2> fe_values(fe_u,
                          quadrature,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe_u.n_dofs_per_cell();
    const unsigned int n_q = quadrature.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> d_q(n_q);
    std::vector<double> eta_q(n_q);

    for (const auto &cell_u : dof_u.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit(cell_u);

      const auto cell_d = DoFHandler<2>::active_cell_iterator(&triangulation,
                                                               cell_u->level(),
                                                               cell_u->index(),
                                                               &dof_d);
      std::vector<types::global_dof_index> dof_indices_d(fe_d.n_dofs_per_cell());
      cell_d->get_dof_indices(dof_indices_d);
      double d_val = 0.0;
      double eta_val = 0.0;
      for (unsigned int i = 0; i < fe_d.n_dofs_per_cell(); ++i)
      {
        d_val += old_solution_d[dof_indices_d[i]];
        eta_val += eta_nodal[dof_indices_d[i]];
      }
      d_val /= static_cast<double>(fe_d.n_dofs_per_cell());
      eta_val /= static_cast<double>(fe_d.n_dofs_per_cell());
      std::fill(d_q.begin(), d_q.end(), std::clamp(d_val, 0.0, 1.0));
      std::fill(eta_q.begin(), eta_q.end(), std::clamp(eta_val, 0.0, 1.0));

      bool in_fiber = false;
      {
        const auto c = cell_u->center();
        const double dx = c[0] - prm.fiber_center_x;
        const double dy = c[1] - prm.fiber_center_y;
        in_fiber = (std::sqrt(dx * dx + dy * dy) <= prm.fiber_radius);
      }

      for (unsigned int q = 0; q < n_q; ++q)
      {
        const SymmetricTensor<4, 2> Cq = elasticity_tensor(in_fiber, d_q[q] * (1.0 - 0.5 * eta_q[q]));

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const SymmetricTensor<2, 2> eps_i =
            symmetrize(fe_values[FEValuesExtractors::Vector(0)].gradient(i, q));

          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            const SymmetricTensor<2, 2> eps_j =
              symmetrize(fe_values[FEValuesExtractors::Vector(0)].gradient(j, q));
            cell_matrix(i, j) += (eps_i * Cq * eps_j) * fe_values.JxW(q);
          }
        }
      }

      cell_u->get_dof_indices(local_dof_indices);
      constraints_u.distribute_local_to_global(cell_matrix,
                                               cell_rhs,
                                               local_dof_indices,
                                               matrix_u,
                                               rhs_u);
    }

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(matrix_u);
    A_direct.vmult(solution_u, rhs_u);
    constraints_u.distribute(solution_u);
  }

  void FiberPFCZM::update_history_field()
  {
    QGauss<2> quadrature(fe_u.degree + 1);
    FEValues<2> fe_values(fe_u,
                          quadrature,
                          update_gradients | update_JxW_values);

    const unsigned int n_q = quadrature.size();
    std::vector<types::global_dof_index> u_ids(fe_u.n_dofs_per_cell());

    for (const auto &cell_u : dof_u.active_cell_iterators())
    {
      fe_values.reinit(cell_u);
      cell_u->get_dof_indices(u_ids);

      bool in_fiber = false;
      {
        const auto c = cell_u->center();
        const double dx = c[0] - prm.fiber_center_x;
        const double dy = c[1] - prm.fiber_center_y;
        in_fiber = (std::sqrt(dx * dx + dy * dy) <= prm.fiber_radius);
      }

      const SymmetricTensor<4, 2> C0 = elasticity_tensor(in_fiber, 0.0);
      double psi_cell = 0.0;

      for (unsigned int q = 0; q < n_q; ++q)
      {
        Tensor<2, 2> grad_u;
        for (unsigned int i = 0; i < fe_u.n_dofs_per_cell(); ++i)
        {
          const unsigned int comp = fe_u.system_to_component_index(i).first;
          grad_u[comp] += solution_u[u_ids[i]] * fe_values.shape_grad(i, q);
        }
        const SymmetricTensor<2, 2> eps = symmetrize(grad_u);
        psi_cell += 0.5 * (eps * C0 * eps);
      }
      psi_cell /= static_cast<double>(n_q);

      const auto cell_d = DoFHandler<2>::active_cell_iterator(&triangulation,
                                                               cell_u->level(),
                                                               cell_u->index(),
                                                               &dof_d);
      std::vector<types::global_dof_index> d_ids(fe_d.n_dofs_per_cell());
      cell_d->get_dof_indices(d_ids);
      for (const auto id : d_ids)
        history_nodal[id] = std::max(history_nodal[id], psi_cell);
    }
  }

  void FiberPFCZM::assemble_and_solve_d()
  {
    matrix_d = 0;
    rhs_d = 0;

    QGauss<2> quadrature(fe_d.degree + 1);
    FEValues<2> fe_values(fe_d,
                          quadrature,
                          update_values | update_gradients |
                            update_JxW_values | update_quadrature_points);

    const unsigned int dofs_per_cell = fe_d.n_dofs_per_cell();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_d.active_cell_iterators())
    {
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.reinit(cell);

      for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        double Hq = 0.0;
        double eta_q = 0.0;
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          const auto id = cell->vertex_dof_index(i, 0);
          Hq += history_nodal[id] * fe_values.shape_value(i, q);
          eta_q += eta_nodal[id] * fe_values.shape_value(i, q);
        }
        Hq = std::max(Hq, 0.0);
        const double coupling = 1.0 + 0.5 * eta_q;

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
          {
            cell_matrix(i, j) +=
              (prm.Gc * prm.l0 * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) +
               (prm.Gc / prm.l0 + 2.0 * coupling * Hq) * fe_values.shape_value(i, q) *
                 fe_values.shape_value(j, q)) *
              fe_values.JxW(q);
          }
          cell_rhs(i) += 2.0 * coupling * Hq * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }
      }

      cell->get_dof_indices(local_dof_indices);
      matrix_d.add(local_dof_indices, cell_matrix);
      rhs_d.add(local_dof_indices, cell_rhs);
    }

    SparseDirectUMFPACK A_direct;
    A_direct.initialize(matrix_d);
    A_direct.vmult(solution_d, rhs_d);

    std::map<types::global_dof_index, Point<2>> support_points;
    DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof_d, support_points);
    for (const auto &[idx, p] : support_points)
    {
      const bool in_no_damage_strip = std::abs(p[1] - 0.5 * prm.L) < 0.03 * prm.L;
      if (in_no_damage_strip)
        solution_d[idx] = 0.0;
      solution_d[idx] = std::max(old_solution_d[idx], solution_d[idx]);
      solution_d[idx] = std::clamp(solution_d[idx], 0.0, 1.0);
    }
  }

  double FiberPFCZM::compute_reaction_force_x() const
  {
    QGauss<1> face_quadrature(fe_u.degree + 1);
    FEFaceValues<2> fe_face_values(fe_u,
                                   face_quadrature,
                                   update_values | update_gradients |
                                     update_normal_vectors | update_JxW_values |
                                     update_quadrature_points);

    double reaction_x = 0.0;
    std::vector<types::global_dof_index> u_ids(fe_u.n_dofs_per_cell());

    for (const auto &cell : dof_u.active_cell_iterators())
      for (unsigned int f = 0; f < GeometryInfo<2>::faces_per_cell; ++f)
        if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2)
        {
          fe_face_values.reinit(cell, f);
          cell->get_dof_indices(u_ids);

          bool in_fiber = false;
          {
            const auto c = cell->center();
            const double dx = c[0] - prm.fiber_center_x;
            const double dy = c[1] - prm.fiber_center_y;
            in_fiber = (std::sqrt(dx * dx + dy * dy) <= prm.fiber_radius);
          }

          std::vector<types::global_dof_index> d_ids(fe_d.n_dofs_per_cell());
          const auto cell_d = DoFHandler<2>::active_cell_iterator(&triangulation,
                                                                   cell->level(),
                                                                   cell->index(),
                                                                   &dof_d);
          cell_d->get_dof_indices(d_ids);

          for (unsigned int q = 0; q < face_quadrature.size(); ++q)
          {
            Tensor<2, 2> grad_u;
            for (unsigned int i = 0; i < fe_u.n_dofs_per_cell(); ++i)
            {
              const unsigned int comp = fe_u.system_to_component_index(i).first;
              grad_u[comp] += solution_u[u_ids[i]] * fe_face_values.shape_grad(i, q);
            }

            double d_val = 0.0;
            for (unsigned int i = 0; i < fe_d.n_dofs_per_cell(); ++i)
              d_val += old_solution_d[d_ids[i]];
            d_val /= static_cast<double>(fe_d.n_dofs_per_cell());

            const auto C = elasticity_tensor(in_fiber, std::clamp(d_val, 0.0, 1.0));
            const auto eps = symmetrize(grad_u);
            const auto sigma = C * eps;
            const Tensor<1, 2> traction = sigma * fe_face_values.normal_vector(q);
            reaction_x += traction[0] * fe_face_values.JxW(q);
          }
        }

    return reaction_x;
  }

  void FiberPFCZM::output_results(const unsigned int step) const
  {
    DataOut<2> data_out_u;
    data_out_u.attach_dof_handler(dof_u);
    data_out_u.add_data_vector(solution_u, {"ux", "uy"});
    data_out_u.build_patches();
    std::ofstream out_u(out_dir / ("u_step_" + Utilities::int_to_string(step, 4) + ".vtu"));
    data_out_u.write_vtu(out_u);

    DataOut<2> data_out_d;
    data_out_d.attach_dof_handler(dof_d);
    data_out_d.add_data_vector(solution_d, "damage");
    Vector<double> eta_output(solution_d.size());
    for (unsigned int i = 0; i < eta_output.size(); ++i)
      eta_output[i] = eta_nodal[i];
    data_out_d.add_data_vector(eta_output, "eta");
    data_out_d.build_patches();
    std::ofstream out_d(out_dir / ("damage_eta_step_" + Utilities::int_to_string(step, 4) + ".vtu"));
    data_out_d.write_vtu(out_d);
  }

  double FiberPFCZM::prescribed_displacement(const unsigned int step) const
  {
    if (step <= prm.n_steps_5)
      return prm.step5_u * static_cast<double>(step) / prm.n_steps_5;

    const unsigned int s6 = step - prm.n_steps_5;
    return prm.step5_u +
           (prm.step6_u - prm.step5_u) * static_cast<double>(s6) / prm.n_steps_6;
  }

  void FiberPFCZM::run()
  {
    setup_mesh();
    setup_systems();

    const unsigned int total_steps = prm.n_steps_5 + prm.n_steps_6;

    for (unsigned int step = 1; step <= total_steps; ++step)
    {
      const double ux = prescribed_displacement(step);

      for (unsigned int iter = 0; iter < prm.max_alt_iters; ++iter)
      {
        const Vector<double> d_prev(old_solution_d);

        assemble_and_solve_u(ux);
        update_history_field();
        assemble_and_solve_d();

        Vector<double> diff(solution_d);
        diff -= d_prev;
        old_solution_d = solution_d;

        if (diff.l2_norm() < prm.alt_tol)
          break;
      }

      const double force_x = compute_reaction_force_x();
      force_disp_csv << step << "," << std::setprecision(12) << ux << "," << force_x << "\n";

      if (step % prm.output_stride == 0 || step == 1 || step == total_steps)
        output_results(step);

      std::cout << "step=" << step << "/" << total_steps
                << " ux=" << ux << " Fx=" << force_x << std::endl;
    }
  }

} // namespace PF

int main(int argc, char **argv)
{
  using namespace PF;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  Parameters prm;
  std::string out_dir = "results";

  for (int i = 1; i < argc; ++i)
  {
    const std::string arg(argv[i]);
    if (arg == "--smoke")
      prm.smoke = true;
    else if (arg.rfind("--output=", 0) == 0)
      out_dir = arg.substr(9);
  }

  if (prm.smoke)
  {
    prm.global_refinement = 3;
    prm.n_steps_5 = 4;
    prm.n_steps_6 = 6;
    prm.output_stride = 1;
    prm.max_alt_iters = 15;
    prm.alt_tol = 1e-7;
  }

  std::filesystem::create_directories(out_dir);

  try
  {
    FiberPFCZM solver(prm, out_dir);
    solver.run();

    std::ofstream meta(std::filesystem::path(out_dir) / "run_info.txt", std::ios::app);
    meta << "status=success\n";
    meta << "smoke=" << (prm.smoke ? "true" : "false") << "\n";
    meta << "steps=" << (prm.n_steps_5 + prm.n_steps_6) << "\n";
  }
  catch (const std::exception &e)
  {
    std::ofstream meta(std::filesystem::path(out_dir) / "run_info.txt", std::ios::app);
    meta << "status=failure\n";
    meta << "message=" << e.what() << "\n";
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
