# Benchmark 训练与 RoboScope 评测接入规范

面向新增 benchmark 的实现者：以 RoboTwin 为参考，完成 LeRobot v3.0 数据加载、训练和 RoboScope 推理评测。
下文是通用交付要求；GroundingPI/π0.5、EEF20 和三相机拼图是当前示例，不要求其他 benchmark 照搬。

## 1. 目录与入口

```text
<Benchmark>/
├── README.md                    # 环境、数据、训练和评测命令
├── train_files/
│   ├── <model>.yaml              # 训练配置
│   ├── run_train.sh              # 训练启动入口
│   └── <benchmark>_dataset.py    # 数据适配（需要时添加）
└── eval_files/
    ├── run_eval.sh               # RoboScope 评测启动入口
    └── evaluate.py               # 组织 benchmark manifest、policy 配置和评测
```

文件名可沿用已有约定；优先复用公共训练器、加载器和 RoboScope adapter。
benchmark 特有的观测／动作转换放在适配层，不复制一套训练框架。

## 2. 训练配置与启动脚本

- YAML 明确模型与预训练权重、数据源及划分、相机顺序和图像尺寸／拼图方式。
- 明确状态／动作维度、字段顺序、绝对或增量动作、单位、坐标系、旋转与夹爪约定、动作预测长度及归一化方式。
- 明确 seed、每卡 batch、GPU 数、梯度累积、学习率、训练步数、日志和 checkpoint 保存间隔。
  有效 batch = 每卡 batch × 训练进程数 × 梯度累积步数。
- Shell 入口使用 `set -euo pipefail`，支持指定 Python、GPU、配置与输出目录；明确参数覆盖规则，保存最终生效配置和命令。
- 按“数据验证 → 少量 smoke 训练 → 正式训练 → checkpoint 加载验证”执行；任何阶段失败均返回非零退出码。
  正式实验使用新目录；恢复训练须显式指定 checkpoint。若调整 batch，记录实际值与有效 batch。

参考：[训练 YAML](Robotwin/train_files/groundingpipi05_eef_clean.yaml)、
[训练流水线](Robotwin/train_files/run_groundingpi_pi05_pipeline.py)。
RoboTwin 的 LeRobot 入口会覆盖 YAML 中的数据加载器等字段，应以运行产物 `launch.yaml` 为准。

## 3. LeRobot v3.0 数据加载

- 依据 `meta/info.json` 和 episode 元数据定位 Parquet 与图像／视频，不能假设一个文件只有一个 episode。
- 明确任务文本、相机、`observation.state`、`action` 的来源和映射，保持输出与所选模型的现有 batch 接口一致。
- 按 episode、frame／时间戳对齐图像、状态、动作；动作块不得跨 episode。
  明确动作是否已指向下一时刻、末尾截断／填充规则及 padding 的损失处理，避免重复时移。
- 归一化统计量只来自训练划分，随 checkpoint 保存；推理复用同一统计量，禁止重复归一化或反归一化。
- 验证字段、维度、有限值、图像解码和帧对齐，人工抽查至少一个样本；缺失或坏数据须报错或显式记录过滤数量。
- 源数据只读，生成的索引／统计／预处理缓存另存；缓存记录源数据版本与预处理配置。
  可选 CoT 按源 episode 和 frame 对齐，明确无标注样本的损失处理，推理不得依赖真实答案。

参考：[LeRobot 数据适配](Robotwin/train_files/robotwin_eef_lerobot_dataset.py)、
[数据与训练说明](Robotwin/train_files/README_groundingpi_pi05_eef_lerobot.md)。

## 4. RoboScope 推理与评测

- 评测入口至少接收 checkpoint、独立输出目录、GPU 和 suite；记录任务集合、seed、回合数与并发数。
- 通过 RoboScope 的 benchmark manifest 和 policy 配置启动 `roboscope_kit.runtime.benchmark`。
  manifest 指定仿真运行方式；policy 配置指定策略进程及环境。新增 benchmark 时实现或复用对应 adapter。
- Policy 将 RoboScope 观测映射为模型输入，处理回合重置及历史状态清空，并按 SDK 契约返回动作块。
  相机顺序、图像处理、提示词、状态编码必须与 checkpoint 一致，训练期随机增强不用于评测。
- 明确状态归一化、动作反归一化及环境动作转换分别由哪一层负责，每项只执行一次。
  核对动作块长度／执行频率、单位、坐标系、四元数顺序及夹爪范围；不得直接套用其他机器人的转换。
- 按“无策略渲染检查 → 单回合 smoke → 全量评测”验证；环境异常与正常任务失败分开记录。
  使用场景缓存时校验来源与版本；所需图形库和环境变量在启动入口显式配置。

参考：[RoboScope 评测组织](Robotwin/eval_files/evaluate_groundingpi_cached.py)、
[Shell 入口](Robotwin/eval_files/evaluate_groundingpi_cached.sh)。后者透传 Python 入口的参数。
RoboTwin 示例使用缓存场景，每种模式 50 个任务、每任务 20 回合；这些数值及 NVIDIA/Vulkan 路径不作为其他 benchmark 的默认值。
`start_eval.sh` 走 RoboTwin 原生策略服务评测，与上述 RoboScope 入口不同。

## 5. 日志与交付验收

每次实验使用独立编号／目录，至少保留：

| 内容 | 最小记录 |
| --- | --- |
| 可复现信息 | 最终配置、启动命令、代码版本及本地差异（含相关新增源码）、环境和数据版本 |
| 训练 | stdout／stderr、step/loss/学习率等指标、checkpoint 及其配置／归一化统计量 |
| 评测 | checkpoint 标识或哈希、manifest／policy 配置、任务／seed／回合数、逐任务成功率和总结果 |
| 失败与状态 | 当前阶段、起止时间、退出码、异常任务／回合及错误日志；不得将未完成评测报为成功 |

本地日志必须完整，W&B 等在线记录可选；RoboTwin 流水线中的 `status.json`、`metrics.jsonl` 和评测 `result.json` 可作参考。
清理临时 smoke 权重，保留验证结论与必要日志，正式 checkpoint 单独保存。

交付时提供训练 YAML、训练／评测脚本、必要适配代码和 README 中可复现的命令。
验收须证明：真实数据样本正确、smoke loss 有限、checkpoint 可重载、三阶段评测流程可运行、日志能追溯结果。

## 6. 当前 RoboTwin 命令示例

从仓库根目录执行；先配置相应环境、预训练权重和数据路径。以下路径占位符需替换，输出目录须为新目录。
训练只运行到 checkpoint 验证，随后单独启动 RoboScope 评测：

```bash
STARVLA_PYTHON=/path/to/starvla/bin/python \
bash examples/simBenchmarks/Robotwin/train_files/run_groundingpi_pi05_eef_lerobot.sh \
  --config-yaml examples/simBenchmarks/Robotwin/train_files/groundingpipi05_eef_clean.yaml \
  --dataset-root /path/to/lerobot_v3_dataset --data-root /path/to/writable_cache \
  --output /path/to/new_train_run --gpus 0 --batch-size 1 --train-only

SDK_PYTHON=/path/to/roboscope-kit/bin/python \
bash examples/simBenchmarks/Robotwin/eval_files/evaluate_groundingpi_cached.sh \
  --checkpoint /path/to/pytorch_model.pt --output /path/to/new_eval_run \
  --cache-source /path/to/verified_robotwin_cache --gpus 0 \
  --suites demo_clean demo_randomized
```

该评测示例还使用本机约定的策略／仿真 Python 路径，迁移机器时须检查 Python 入口生成的 policy 配置及 manifest。
