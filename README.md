# deal.II 版 coating PF-CZM（三点弯曲半模型）

本仓库新增了一个可编译的 deal.II C++ 实现，用于迁移原 FEniCSx/DOLFINx 的 `coating_pfczm.py` 主流程（位移-相场交替、增量 cut-back、历史变量、力-位移输出）。

## 新增文件

- `/home/runner/work/10_23/10_23/source/coating_pfczm.cc`
  - 2D 平面应变（显式处理 33 分量）
  - 结构化四边形网格（区域：`x∈[-10,0], y∈[-0.5,1.2]`）
  - 涂层/基底分区（`y<=0` 涂层，`y>0` 基底）
  - 基底：J2 塑性（线性各向同性硬化、径向回归）
  - 涂层：Hencky 弹性 + Anand 型显式黏塑历史
  - PF-CZM 风格损伤相场 `d` 与界面相场 `eta`
  - 不可逆约束 `d_old <= d <= 1`（投影实现）
  - 交替求解 + 失败 cut-back
  - 输出 `.vtu` 和 `force_displacement.csv`
- `/home/runner/work/10_23/10_23/CMakeLists.txt`
- `/home/runner/work/10_23/10_23/postprocess.py`
  - 读取 `force_displacement.csv` 画力-位移曲线
  - 可选读取 VTU 打印字段摘要

## 依赖

- C++17
- deal.II >= 9.4
- CMake >= 3.16
- Python（后处理可选）: `pandas`, `matplotlib`（可选 `meshio`）

## 构建

```bash
cd /home/runner/work/10_23/10_23
cmake -S . -B build -DDEAL_II_DIR=/path/to/dealii
cmake --build build -j
```

## 运行

### 快速 smoke test（小网格、少步）

```bash
cd /home/runner/work/10_23/10_23/build
./coating_pfczm --mesh=0.5 --final-time=0.2 --dt=0.1 --output-every=1
```

### 更细网格（接近原始设置）

```bash
./coating_pfczm --mesh=0.0125 --final-time=1.0 --dt=0.02 --output-every=5
```

## 输出

- `solution-XXXX.vtu`：位移 `ux,uy`、`damage`、`eta`、`eqp`、`anand_s`、`H_damage`
- `force_displacement.csv`

## 后处理

```bash
cd /home/runner/work/10_23/10_23
python postprocess.py --csv force_displacement.csv --out force_displacement.png
```

## 已知限制

- 当前实现保持了原问题中的关键数值流程和变量组织，但为保证在纯 deal.II 单进程环境可落地，
  相场受限求解使用投影 active-set 等价策略（非 PETSc SNES）。
- 若运行环境未安装 deal.II，则可完成静态代码检查，但无法在本地执行编译/算例。
