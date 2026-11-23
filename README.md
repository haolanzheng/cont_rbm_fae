# Continuous greedy reduced basis methods

主文稿：`paper/continuous_greedy.pdf`。公共代码采用六个模块，
各 PDE 与实验协议放在 `examples/`。重构范围与参数来源见
[架构说明](docs/ARCHITECTURE.md)和[参数对照](docs/PARAMETER_AUDIT.md)。

## 目录与阅读顺序

| 路径 | 职责 |
|---|---|
| `src/contgreedy/data/` | 数据读写、统计、通用 GRF 装配 |
| `src/contgreedy/domain/` | box、ball、sphere 约束操作 |
| `src/contgreedy/hf/` | 高保真求解公共约定 |
| `src/contgreedy/rom/` | 加权基构造、greedy 外循环 |
| `src/contgreedy/sampling/` | 候选分布、pool、JAX Adam、host 搜索和 R-LBFGS |
| `src/contgreedy/fae/` | 公共 FAE 接口、模型组件、训练器、检查点、latent 梯度回传 |
| `examples/<case>/` | 物理输入、HF、特定 ROM/指标、按需实现的 FAE、实验驱动和配置 |
| `tests/` | 数值对照、论文参数、CLI、FAE 检查点与续训回归 |
| `paper/` | 主文稿和补充图 |
| `runs/`、`data/` | 检查点和输入数据 |

先读 `examples/baseline`，再读 `examples/thermal_fin`，各目录 README 给出阅读顺序。
旧 `experiments` Python 命名空间保留为兼容转发，实际文件已迁到 `examples`。
原 JSON、图、检查点和训练历史保留，不能将它们当作本次重跑结果。

## 安装与快速验证

需要 Python >=3.10。在自己的环境安装：

```bash
python -m pip install -e '.[test,fem]'
python run.py verify
python -m pytest -q
python run.py spectral --config examples/baseline/configs/smoke.json --output runs/spectral_smoke
python run.py thermal-fin --config examples/thermal_fin/configs/smoke.json --output runs/fin_smoke
```

新入口要求全新的输出目录；已存在时报错，不会静默命中历史结果。
测试是 CPU 小规模检查，正式 GPU/全尺寸实验另行执行。
本次本地环境是 `.venv`；可使用 `.venv/bin/python` 执行上述命令。

JAX float64 用于公共数值核及 spectral/thermal-fin 在线计算。
SciPy 保留稀疏装配、LU 和历史局部求解器。新计时包含编译，
不能直接与旧 NumPy CPU 计时表比较。其他历史算例保留原数值后端；
详细迁移范围见架构说明。

## 连续源项与有限元 HF baseline

`examples/baseline/continuous_fem.py` 给出不依赖网格的制造源项：先在连续
能量内积下正交化正弦函数，再定义参数化右端。HF 使用协调三角形有限元，
并通过同一弱形式装配载荷；已有 greedy/RB 代码无需改变。P1、P2、P3
均可选，默认 P3。旧差分与网格标定载荷仍保留，仅供复现历史数据；两套
实验不能混作同一个 PDE 的数值结果。

```bash
python run.py spectral --config examples/baseline/configs/continuous_fe_smoke.json --output runs/continuous_fe_smoke
OPENBLAS_NUM_THREADS=1 python examples/baseline/check_continuous_fe.py
```

第二条命令在 `d=64` 对 P3 网格作收敛检查，结果写入
`runs/report_baseline/fe_p3_convergence.json`。40×40 内部顶点配置有
14,884 个自由度，整个制造解空间的最大相对能量投影误差为 0.1954%；
半秩处的有限元离散宽度与连续宽度相差约 `1.24e-7`（相对值）。
完整的精确指标、GD--100、固定 pool 对照及概率界已按原种子协议重跑，
结果在 `runs/report_baseline/fe_p3_grid40_results.json`，报告和图也已更新。
旧差分数据保留于 `grid48_results.json`，不得与有限元数据混用。

## FAE

当前选择的输入 FAE 使用 JAX：UNO 通道为 32/48/96/192，latent dimension
为 64，decoder 是 width-256、四层的条件 residual MLP。raw latent 在输入和
每个 residual block 注入；decoder 没有 LayerNorm、POD 基底或 Pirate block。
完整模型有 1,137,743 个实参数。

```bash
sbatch --parsable jobs/fae_nonlinear_decoder_comparison.sbatch
```

训练使用 4096/256/256 的固定数据划分、batch 32、200 epochs（25,600
updates）、Adam 与 norm-1 gradient clipping；学习率从 1e-3 余弦降至
1e-5，无 warmup。GH200 配对实验 64632254 中，所选模型的 validation/test
mean relative L2 为 0.151978/0.153882，优于固定 POD-like decoder 的
0.169347/0.171808。相同协议下的 modulated SIREN 发生优化塌缩，不作为
默认模型。

完整参数、尾部误差、latent 有效秩及限制见
[非线性 decoder 对照](docs/FAE_NONLINEAR_DECODERS.md)。前序的数据量/容量
实验与固定/FiLM 基底对照分别见
[数据量与模型容量实验](docs/FAE_SCALING.md) 和
[POD-like decoder 对照](docs/FAE_DECODER_ABLATION.md)。JAX 环境、检查点与
续训方式见 [JAX/GH200 训练说明](docs/FAE_JAX_GPU.md)。

原 PyTorch epoch-4 记录保留于 runs/fae128；FAE 的检查与训练统一使用
`fae-check` 和 `fae-train`。运行说明见 [docs/FAE_GPU.md](docs/FAE_GPU.md)。
`examples/hyperelasticity` 当前实现输入生成和 FAE，PDE/ROM 尚未实现。
该算例的 FAE 组装、配对数据加载与度量位于
`examples/hyperelasticity/fae.py`；其他算例按需实现自己的 FAE。

## 论文实验与历史命令

```bash
python run.py list --all
python run.py paper
```

原 53 个命令继续可用，新增 `spectral`、`thermal-fin` 配置入口。
历史数据重建与缺失材料见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
不要批量执行所有历史命令；部分复现需重建大型数组。
历史输出保留在算例目录，新实验建议写入独立 `runs/<name>`。

`python run.py package` 生成可移植包与当前 SHA-256 清单。
`docs/original_manifest.json` 保留重构前清单，记录原始来源。
