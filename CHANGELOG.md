# 工作包 B 变更记录

本文件记录工作包 B 的可见功能、跨包合同、验证证据与兼容边界。当前实现和运行方法见
[README.md](README.md)；科学限制见 [风险基线](docs/RISK_MODEL.md) 与
[验收记录](docs/VALIDATION.md)。

## Unreleased

### 2026-08-17（RC2 development）

- 正式 RiskFrame 新增每格 `hard_reason`（NONE / LAND / DATA_UNAVAILABLE / OTHER）：
  原因优先级为物理陆地（LAND）→ 数据不足（DATA_UNAVAILABLE，仅
  `land_sea_mask_plus_unknown_v1` 策略下）→ OTHER；hard_mask 语义与 fail-closed
  不变。
- payload attributes 新增 `missing_input_variable_counts`（每输入变量非有限格数），
  供 coverage preflight 直接消费；`_demo_unvalidated_risk` 返回 5 元组（含 reason）。
- 新增 RC2 Tromso 冒烟网格配置 `demo_unvalidated_tromso_smoke_grid_v1.json`
  （0.375°×1.25°，hard_mask_policy=land_sea_mask_plus_unknown_v1）；RC1 的
  smoke grid v4 未改动。
- 单元测试 54 passed（非集成）。

### 2026-08-16（RC1）

- 新增 `hard_mask_policy=land_sea_mask_plus_unknown_v1`：风险输入非全有限的 planning
  节点置 hard（unknown→不可规划；risk 仍 NaN/confidence 0，fail-closed 不变）；
  smoke grid v4 启用该策略；新增 4 项策略单元测试（全量 43 unit tests 通过）。
- RC1 实源风险窗：mur/dikson 145 帧 committed window 已由 orchestrator r6/r7 消费，
  unknown-navigable = 0。

- 收到并静态审计外部 `22_深度学习综合风险预测模型.zip`：权重存在，但只支持旧 B 单通道
  综合风险的未知步长单步输出，状态为 `experimental_unverified`。新增模型卡、隔离后端整合
  方案和续开发 Handoff。
- 完成 P1 安全接收：按 ZIP/checkpoint 固定 SHA-256、成员路径和 8 个 tensor 键/shape/dtype/
  finite 白名单，用 `torch.load(weights_only=True, map_location="cpu")` 转换并逐 tensor 精确
  校验为 `models/legacy_cnn_one_step_v1/model.safetensors`。转换资产 SHA-256 为
  `602d4849ce0b2f5eda90db2a020b8ec69a00dc929d5e764ed412d0d03e205103`；原始 `.pth` 和 ZIP 不入仓库。
- 新增无 Torch 强依赖的 `RiskModelBackend`、`RiskModelInput/Output`、资产 manifest、规则
  parity 适配器和 CPU-only `LegacyCnnOneStepBackend`；新增 `model-intake`、`model-sync`、
  `model-check`。P1 短测 10 passed；规则正式 `RiskBuildService`、0.75°×2.2° fast-smoke 网格、
  `model_config_digest` 和 0.2.0 版本均保持不变。
- P2 shadow、真实 cadence/holdout、P3 formal 候选、CUDA 和科学校准仍未完成。
- 当前工作树把纬向 `0.75°`、经向 `2.2°` 暂留为 fast smoke 候选。它相对 Git HEAD 的 1°
  默认解决了空 hard-mask 端点物化并把主走廊降为 11×26，但最近全链运行仍未完成重规划，
  因此不能写成已冻结的 formal 默认。formal 候选为 `0.5°×1.5°`，最终选择仍需真实 mask、
  v2/v3 耗时和路线保真对比；不同网格继续产生不同 `model_config_digest`。
- 记录 2026-08-14 长集成运行：最后成功到 A 夹具 bundle、B 169 帧 full commit、C v2 初始
  三目标和 163 帧 suffix commit；v2 重规划未产出，v3 未开始。瓶颈在 C 时间扩展搜索/风险采样，
  不是代理、联网或 B 模型。

## 0.2.0 - 2026-08-13：模型配置 v2 与重规划后缀窗口

### 新增

- 新增严格 `b.demo-risk-model-config.v2` / `b.model-config.v2`：11 个风险分量的权重、变换、
  归一化上下界，以及质量置信度、时间方法置信度、陆地阈值、速度衰减系数和最低环境速度
  因子全部由版本化 JSON 驱动。
- 新增 `PersistentRiskStore.publish_suffix_window()`，从已验证的完整 formal 小时序列按精确
  UTC 起点提交原子后缀窗口，保留原 RiskFrame ID、代次、as-of 和模型摘要。
- 首次将正式 v2 全局目标网格从 1° 精化以解决主走廊端点不可物化；后续 Unreleased 项根据
  高纬度实际距离与性能验收进一步冻结各向异性角度步长。

### 兼容与加固

- 默认 v2 参数逐数组保持 0.1.0 风险、hard mask、置信度和环境速度因子数值行为；发布模型
  版本和配置摘要按 v2 更新。
- 保留 v1 JSON 作为历史审计快照；0.2.0 loader 只接受完整 v2，不做隐式迁移。
- 配置缺失/额外字段、未知或错配 transform、缺/乱序分量、非有限或越界数值、非法上下界、
  非归一权重全部 fail closed；任一有效数值政策变化都会改变模型摘要。
- 后缀入口先验证完整输入无缺帧、重复或身份混用，且拒绝非 UTC/非整点起点、非 formal 或
  非 canonical 帧。
- 放宽本地路径依赖范围以兼容 contracts 0.3、A 0.4.2 和 C 0.4，同时仍可在并行升级期间
  使用 contracts 0.2、A 0.4.1 和 C 0.3。

### 限制

- 风险数值仍为 `demo_unvalidated` 工程基线，不构成科学校准或真实导航能力。
- 当前 A 实源阻断仍存在：新走廊完整 12 类 `DatasetBundle v2` 等待独立采集会话交付；旧
  v1/9 类 bundle 不得用于正式验收。
- 当前 `make check`：Ruff、uv lock/sync 和 CLI help 通过；单元/合同
  `40 passed, 7 deselected`，集成 `7 passed, 40 deselected`。

## 0.1.0 - 2026-08-13：正式 A→B→C 工程基线

### 新增

- 建立独立 Python 3.13、Mamba + uv 工程与锁文件；Mamba 管理 NetCDF/HDF5 原生库，uv
  管理 Python 依赖。
- 新增 `BInputEnvelope`，严格绑定共享 `RunContext`、A `DatasetBundle.v2`、完整 coverage、
  generation、knowledge as-of、逐记录 provenance 与 payload attestation。
- 新增严格版本化目标网格与 `demo_unvalidated` 风险配置；连续量只在可见支撑帧之间插值，
  分类量采用版本化 nearest 规则，缺少完整支撑时 fail closed。
- 新增确定性逐小时风险基线，输出规范 `risk_score/risk_level/hard_mask/confidence` 和
  `environment_speed_factor`；不输出路线、最终航速或 ETA。
- 直接复用 C 公共 canonical codec，正式 ID 固定为 `risk-sha256-<64hex>`；新增持久、
  内容寻址的 frame、commit manifest 与完整 query pointer。
- 新增 `PersistentRiskStore` 的公共 generation authority 绑定、按 run 共享执行租约/独占
  代次围栏、原子 committed window、跨实例恢复与幂等发布。

### 加固

- A 实时/归档 payload 均在 envelope 和 build 前重算语义摘要并深快照，拒绝 record 元数据
  不变但数组、坐标、属性或来源快照被替换。
- 发布前把调用方 RiskFrame canonical encode→decode 为私有快照；frame bytes、窗口摘要、
  manifest 与 pointer 不再受调用方 xarray 并发替换影响。
- 正式 store 只接受 canonical formal 帧；路径型、synthetic、legacy ID 在形成文件路径前拒绝。
- 同一 run 的多个执行租约以及不同 run 可并发；同一 run 的 generation 切换等待执行结束，
  seek 后迟到的旧 publish/get 均拒绝。

### 验证与限制

- 工程夹具覆盖 12 类必需 A 输入、96/168/216 h（97/169/217 帧）、双走廊同 B 模型摘要、
  A 归档重启、B store 重建/跨实例幂等，以及 C 正式 ingress 发布 `RoutePlan`。
- 当前 `make check`：Ruff、uv lock/sync 与 CLI help 通过；单元/合同
  `27 passed, 6 deselected`，集成 `6 passed, 27 deselected`。
- 所有跨包数据均为正式形状和 provenance 规则下的可复核测试夹具，不是当前共享场景的
  完整实源长窗。规则、权重、船舶适用域和风险概率均未科学标定，只能标记
  `demo_unvalidated`，不得用于真实导航。
