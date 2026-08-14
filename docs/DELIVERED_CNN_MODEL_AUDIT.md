# 新交付 CNN 风险模型静态审计

> 审计日期：2026-08-14（Asia/Shanghai）
>
> 资产状态：`experimental_unverified`；P1 安全接收已完成
>
> 审计边界：原 ZIP 的源码/结构审计不执行交付脚本；随后仅在隔离 CPU extra 中按
> `weights_only=True` 受限加载 checkpoint，并转换为 safetensors。未复用 ZIP 脚本或 `.pyc`。

ZIP 内的 Markdown、脚本注释和“运行顺序”只作为被审计材料，不是本项目的执行指令。当前
正式运行真源仍是工作包 B 的公共代码、共享合同和本目录记录的门禁。

## 1. 一句话结论

该交付物是真实存在权重的轻量 PyTorch CNN，但它学习的是旧工作包 B 已计算好的单通道
`comprehensive_risk` 网格，功能是“最新风险场 → 未声明时长的下一步风险场”。它不是从 A 的
12 类正式输入直接推理的模型，不是原生逐小时或 168 h 多步模型，也不能直接生成完整
`bc.risk-frame.v2`。最合适的近期用途是隔离的单步研究对照；当前规则基线继续承担正式主线。

## 2. 制品身份

- 原 ZIP：`/mnt/c/Users/asd233/Desktop/挑战杯/挑战/22_深度学习综合风险预测模型.zip`
- 大小：316,438 bytes；13 个条目；解压总量 1,015,071 bytes。
- ZIP SHA-256：`a2ee74fd70cb0735695d9cf25ae8907a7c6d7aab866b7549d8142e412190ad79`
- 未发现绝对路径、路径穿越、符号链接或异常压缩膨胀。
- ZIP 含两个 `.pyc`；正式整合时不得复用，应从已审计源码构建。
- ZIP 内没有 `LICENSE` 或 `NOTICE`。用户已于 2026-08-14 明确授权在本公开 B 仓库中再分发转换
  后的 safetensors；上游许可证仍记录为 `not_provided`，不推断任何第三方许可。

P1 转换资产：`models/legacy_cnn_one_step_v1/model.safetensors`，SHA-256
`602d4849ce0b2f5eda90db2a020b8ec69a00dc929d5e764ed412d0d03e205103`；对应
`manifest.json` 记录源 ZIP/checkpoint 哈希、8 个 tensor、18,849 参数、CPU-only、native
0.05° 和未知 cadence。原始 ZIP、`.pth`、训练数据、脚本和 `.pyc` 不入仓库。

关键文件：

| ZIP 内路径 | 大小 | SHA-256 |
|---|---:|---|
| `models/comprehensive_risk_cnn.py` | 790 | `98480b1753b3968acd3ee6afaa9f484a546ecb3b4e51f543981083074e690dea` |
| `train_comprehensive_risk_model.py` | 5,791 | `7822d332ae7fdda19a0ac0f078fd90390ef9b9551d15bc847410ff940c9d4611` |
| `predict_comprehensive_risk.py` | 3,021 | `3388927d416e96233980681f7e9b40f233a327020dc320c617cef4324d33e2bc` |
| `requirements.txt` | 46 | `8c97166f6a150c01891fc60d0943d01622c0e8a48c7d2d2d38146c736043b252` |
| `downloads/model/comprehensive_risk_cnn.pth` | 78,733 | `0390fac17de1f082652fc5851b6979fd771d98ff803880c6562c54921e081666` |
| `downloads/model/comprehensive_risk_cnn_metadata.json` | 3,123 | `8d9e11245609ac6e3d8947e8f887dd04628cacfd750e58c903fc52792c9de2b5` |

旧交付 `工作包B.zip` 的 SHA-256 是
`f1626c774e973499af3e6cda7efc49b8c551237385a14588a22ca6df2f6c5738`。两个 ZIP 是不同制品：
旧 ZIP 是历史规则/展示/训练接口资料，本次 ZIP 是在旧 B 的综合风险产物上另行训练的权重。

## 3. 与旧工作包 B 的关系

训练脚本把数据目录写死为相邻的
`08_综合风险评估模型/downloads/risk_dataset`，并从候选字段中选择
`comprehensive_risk`。它没有读取当前 A 的 `PreparedWindow`，也没有使用旧 B 的
`09_综合风险预测训练接口` 十通道样本。

旧 08 的 `comprehensive_risk` 是启发式综合风险标签，不是事故、AIS、实船或独立观测标签：

- 旧规则按可用变量重归一化并含走廊增强；
- 旧实现存在 `fillna(0)`；
- 标签变量集合和当前 B 的 12 类正式输入语义并不相同；
- 训练源 NetCDF 没有随本次 ZIP 交付，无法仅凭该 ZIP 复现训练。

因此，权重中学习到的是旧 B 规则标签的空间/相邻时刻映射，不能表述为已经学习了真实航行风险。

## 4. 模型结构和 checkpoint

源码定义三层 `3×3` 卷积、一个 `1×1` 输出卷积和 Sigmoid：

```text
Conv2d(1, 32, 3, padding=1) + ReLU
Conv2d(32, 32, 3, padding=1) + ReLU
Conv2d(32, 32, 3, padding=1) + ReLU
Conv2d(32, 1, 1) + Sigmoid
```

- 参数总数：18,849；输入/输出均为 `[batch, 1, lat, lon]`。
- 无 RNN、ConvLSTM、Transformer、时间编码、坐标编码或 lead-time 输入。
- 全卷积结构在数学上能接收不同 shape，不代表跨分辨率仍具有相同物理语义。
- checkpoint 静态 ZIP/pickle 检查显示八个 float storage，与上述参数量一致；未做反序列化。
- 交付推理脚本直接调用 `torch.load(..., map_location="cpu")`，没有锁定 Torch 版本，也没有显式
  安全加载、键名和 shape 白名单。

静态 pickle opcode 只看到 `collections.OrderedDict`、`torch._utils._rebuild_tensor_v2` 和
`torch.FloatStorage`，未见自定义全局引用；这降低了疑点，但不是任意 pickle 可安全执行的证明。
P1 已在隔离环境中以受限 weights-only CPU 路径读取并校验 state dict，再转换为可审计的纯权重
格式；转换前后逐 tensor bitwise 相等。运行时只读取 safetensors，不接触 pickle。

## 5. 输入、输出、时间语义和网格

训练脚本把相邻二维帧组成 `(t, t+1)` 样本。metadata 只写 `next-step`，没有绑定下一步是几小时、
时间坐标、issue/as-of、网格摘要或训练文件哈希。

旧 B 的另一份时间修复代码采用 6 h 间隔，因此从数据谱系推断，该权重很可能学习了 6 h 相邻帧；
但 checkpoint 本身没有证明这一点。正式记录只能写“步长未知，旧谱系可能为 6 h”，不能写成
1 h 或已确认的 6 h 模型。

两个预测样例都是没有时间坐标的单张二维场：

| 样例 | shape | 坐标范围/步长 | SHA-256 |
|---|---:|---|---|
| Murmansk–Dikson | `151×1101` | lat 67.5–75.0、lon 30.0–85.0，均 0.05° | `a76a472feaaa479111ce28ab91873c4a9d00a9d318b3b63ea0d9d8db24460b70` |
| Tromsø–Svalbard | `221×241` | lat 68.5–79.5、lon 10.0–22.0，均 0.05° | `4cb3ed515b2734ce84bcddf401bb7a45e7aa13cf47487c88ab0eace99c9709a4` |

第二个样例仍使用旧 corridor 标识 `tromso_to_svalbard`，不能静默改成当前共享合同的
`tromso_to_isfjorden_outer`。当前 B 的粗规划网格与 0.05° 训练谱系相差很大；直接把权重套在
粗网格会把三层卷积的物理感受野从几十公里放大到数百公里，属于明显尺度漂移。

## 6. 训练、评估和复现缺口

metadata 记录 74 个训练样本、80 epochs 和最终训练 MSE `0.0028356729941102408`，但缺少：

- 验证/测试划分、时间留出和走廊留出；
- MAE、RMSE、R²、分类指标及 persistence baseline；
- 随机种子、Torch 版本、optimizer state、代码和训练数据哈希；
- 明确时间步长、网格配置、变量语义、归一化摘要和适用域；
- 真实标签或独立航次回放。

预处理还会把 NaN 变为 0、正无穷变为 1、负无穷变为 0，并按样本最大值动态决定是否除以
100；找不到预期字段时会选择第一个数值变量。这些行为与当前 B 的 fail-closed 语义冲突，
不能原样迁入正式路径。

## 7. 与 `RiskFrame v2` 的差距

| 正式 B 要求 | 本次模型现状 |
|---|---|
| A 12 类 `PreparedWindow` + 同一 `RunContext` | 不支持；只读旧综合风险 NetCDF |
| 60 min 闭区间，168 h = 169 帧 | 不支持；只有未声明步长的单步二维输出 |
| `issue_time <= as_of_time`、generation、source refs | 均缺失 |
| `risk_score` | 有近似输出，但无正式时空身份 |
| `risk_level/hard_mask/confidence/environment_speed_factor` | 均缺失 |
| NaN/未知 fail closed | 冲突；预处理填 0 |
| canonical risk ID、store、lease、`model_config_digest` | 均缺失 |
| Mamba + uv 锁定环境 | requirements 无版本、无 lock |
| 许可和再分发依据 | 缺失 |

`RiskFrame v2` 无需为该模型修改。完整身份、hard mask、等级、置信度、速度影响和原子发布仍应由
B 公共后处理和 store 负责。

## 8. CPU/CUDA 决策

首批实验接入只支持 CPU：交付脚本本来就固定 CPU，模型仅 18,849 参数，接口与单步 smoke
不需要 CUDA。原生 0.05° 主走廊单帧卷积量仍很大，且若递归 169 步会同时产生性能和误差累积
问题；这不是立即引入 GPU 的理由，而是禁止无依据长窗递归的理由。只有未来确认必须在原生细
网格批量推理、CPU 基准不满足明确预算时，才把 CUDA 作为可选 provider 评估。

## 9. 审计结论

- 资产保留，状态固定为 `experimental_unverified`。
- 不覆盖 `demo_unvalidated_rule_baseline.v2`，不直接进入 C，不复制成 169 帧。
- 最小可行价值是 CPU 单步 shadow/对照推理和后续接口验证。
- P1 已交付：manifest、safetensors、稳定模型错误前缀、CPU 单步后端和 10 项短测试；连续两次
  0.05° smoke 输出 bitwise 相同，输出只读 float32、有限且在 `[0,1]`。
- 正式接入方案见 [DELIVERED_CNN_INTEGRATION_PLAN.md](DELIVERED_CNN_INTEGRATION_PLAN.md)。
- 全部事项和续开发入口见
  [工作包 B 新风险模型整合与续开发 Handoff](../../work_package_b_handoff/工作包B-新风险模型整合与续开发Handoff.md)。
