# 工作包 B 变更记录

本文件记录工作包 B 的可见功能、跨包合同、验证证据与兼容边界。当前实现和运行方法见
[README.md](README.md)；科学限制见 [风险基线](docs/RISK_MODEL.md) 与
[验收记录](docs/VALIDATION.md)。

## Unreleased

- 尚无未发布变更。

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
