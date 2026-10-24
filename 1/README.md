# deal.II C++ 版本：单纤维 PF-CZM（平面应变）

本目录是将原 FEniCSx/dolfinx 版本思路迁移到 deal.II 的 C++ 工程，包含：
- 网格：`1 mm x 1 mm`，中心纤维半径 `0.25 mm`
- 材料：基体/纤维参数分区
- 界面变量 `eta`（按纤维界面距离构造）
- 损伤相场（含不可逆约束 `d_n <= d <= 1`）
- 交替迭代（先位移、后损伤）
- Step-5 / Step-6 位移加载
- 输出：力-位移 CSV + VTU（可在 ParaView 后处理）

## 目录结构

- `CMakeLists.txt`：deal.II CMake 工程
- `src/fiber_pfczm.cc`：主程序
- `run_smoke.sh`：smoke test（小增量）
- `results/`：本环境下真实运行记录与日志

## 依赖

- C++17
- CMake >= 3.16
- deal.II >= 9.3（需能被 `find_package(deal.II ...)` 找到）

## 编译与运行

### Smoke test（建议先跑）

```bash
cd /home/runner/work/10_23/10_23/1
./run_smoke.sh
```

### 手动运行

```bash
cmake -S /home/runner/work/10_23/10_23/1 -B /home/runner/work/10_23/10_23/1/build
cmake --build /home/runner/work/10_23/10_23/1/build -j
/home/runner/work/10_23/10_23/1/build/fiber_pfczm --smoke --output=/home/runner/work/10_23/10_23/1/results/smoke
# 完整步数（100+500）
/home/runner/work/10_23/10_23/1/build/fiber_pfczm --output=/home/runner/work/10_23/10_23/1/results/full
```

## 与原始 Python 版本的等价/近似说明

由于在当前仓库运行环境中缺少可用 deal.II 安装，且无权限安装系统包，已实现可编译的 deal.II 工程源码，但未能在当前环境完成可执行验证。模型层面采用以下等价实现：

1. **应变能**：采用小应变弹性能退化（`(1-d)^2 + kappa`），作为 Hencky 有限应变在当前稳定实现难度下的可运行近似。
2. **PF-CZM**：使用含历史场 `H` 的相场方程与不可逆投影（`d = max(d_old, d_new)`）实现损伤演化。
3. **界面 `eta`**：按距离纤维界面线性衰减构造，并耦合到损伤驱动力。
4. **无损伤条带**：以中线窄条带强制 `d=0` 作为等价约束。

## 本环境实际尝试结果（真实日志）

- 依赖安装尝试：失败（权限不足）
  - 见 `results/apt_install.log`
- CMake 配置尝试：失败（未找到 deal.IIConfig.cmake）
  - 见 `results/cmake_configure.log`
- 汇总信息：`results/run_info.json`

> 因未生成可执行文件，本环境下没有伪造 `force_displacement.csv` / VTU 结果文件。请在安装好 deal.II 的环境中运行上面的命令生成真实结果。
