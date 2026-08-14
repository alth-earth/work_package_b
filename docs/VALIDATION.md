# 验收与限制

本包的自动门槛：

```bash
make lint
make test
make integration
make check
make model-check  # 可选 CPU extra；不进入正式 RiskFrame
git diff --check
```

单元/合同测试覆盖身份串线、future issue、旧 generation、缺帧、逐小时连续性、缺测不填零、
C Schema/codec、risk ID、幂等与冲突发布、发布中途调用方 xarray 变异隔离、
active-generation fence、同/跨 run 并发 execution lease、同 run 代次切换等待、committed query
精确匹配，以及 attestation 后 payload/坐标篡改和外部数组别名隔离。
版本化 JSON 配置也必须严格解析并与实际默认运行对象/摘要一致。
0.2.0 进一步枚举测试 11 个分量的每项权重/上下界、全部质量/时间置信度、陆地阈值、速度
系数和最低因子：每个有效数值变化都必须改变 `model_config_digest`。默认 v2 配置与 0.1.0
风险、hard mask、confidence、speed factor 做逐数组比较；缺/额外字段、非法 transform、范围
和置信度均有负例。

跨包测试使用实际公共 API 验证：

- A `prepare_window_for_b()` → shared bundle/RunContext → B 逐小时窗口/store → C
  `RiskSourcePlanningIngress` → formal RoutePlan；
- 96/168/216 h 分别输出 97/169/217 帧；主/迁移走廊复用相同
  `model_config_digest`；
- A 通过 `AcquisitionPublisher` 落盘 12 类和不可变 source snapshot，进程重启后仅通过
  `resolve_dataset_bundle_for_b()` 恢复，随后进入 B；B 发布后重建 `PersistentRiskStore`，验证
  committed window 可恢复且相同窗口跨实例重复发布幂等。该恢复路径通过真实
  `SimulationClock`/`bind_generation_authority()` 驱动，seek 后旧代次 publish/get 均 fail closed；
  A 恢复 payload 被改写时同样 fail closed。
- B 对完整 formal 小时序列可提交精确 +6 h suffix window；测试固定其 risk IDs、generation、
  as-of 不变，并拒绝非帧边界起点和在截取前就存在缺帧的“完整”输入。
- 纬向 0.75°、经向 2.2° 的 fast smoke 候选通过 C 公共端点映射验证：主走廊与迁移走廊完整
  `data_bbox` 的 start/destination allowed region 均含可航节点，调整距离不超过显式 150 km；
  不以扩大业务允许区绕过失败。该测试使用空 hard mask，只证明端点/连通性几何；不证明真实
  land mask、路线保真或可接受全链耗时，也不把该值冻结为 formal 默认。

这些集成数据是正式形状和 provenance 规则下的**测试夹具**，不是公开源下载结果。当前真实 A
长窗仍是历史 v1/9 类/旧 corridor，故测试通过只证明工程合同闭环，不证明实源闭环或科学正确性。

2026-08-13 的 0.2.0 `make check` 结果：Ruff 通过；单元/合同
`40 passed, 7 deselected`；跨包集成 `7 passed, 40 deselected`；uv lock/sync 检查和 CLI help
通过。本版本验收还执行 `git diff --check`，不得引用 0.1.0 的历史计数代替当前证据。

2026-08-14 的长集成尝试在 0.75°×2.2°/11×26 下完成 A 夹具 bundle、B 169 帧 full commit、
C v2 初始三目标和 163 帧 suffix commit；初始三目标约 27 分钟，重规划未形成 output，v3 未开始。
这不是 `make check` 通过证据。复盘见
[orchestrator incident](../../arctic_route_orchestrator/docs/INCIDENT_2026-08-14_LONG_INTEGRATION_RUN.md)。

尚未宣称完成的科学项：真实风险标签、船型校准、风浪流相对航向、净水深规则、限制区法律规则、
Q50/Q90/概率预测和真实航次回放。新 CNN 已完成固定哈希、weights-only CPU 接收、safetensors
转换和单步短测（`make model-check`：10 passed），但仍缺独立验证、明确 cadence 和 formal 输入
输出；因此不能称为完成的逐小时预测能力。当前系统只能称科研演示闭环。
