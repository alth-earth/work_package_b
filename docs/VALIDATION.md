# 验收与限制

本包的自动门槛：

```bash
make lint
make test
make integration
make check
git diff --check
```

单元/合同测试覆盖身份串线、future issue、旧 generation、缺帧、逐小时连续性、缺测不填零、
C Schema/codec、risk ID、幂等与冲突发布、发布中途调用方 xarray 变异隔离、
active-generation fence、同/跨 run 并发 execution lease、同 run 代次切换等待、committed query
精确匹配，以及 attestation 后 payload/坐标篡改和外部数组别名隔离。
版本化 JSON 配置也必须严格解析并与实际默认运行对象/摘要一致。

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

这些集成数据是正式形状和 provenance 规则下的**测试夹具**，不是公开源下载结果。当前真实 A
长窗仍是历史 v1/9 类/旧 corridor，故测试通过只证明工程合同闭环，不证明实源闭环或科学正确性。

2026-08-13 当前快照的 `make check` 结果：Ruff 通过；单元/合同
`27 passed, 6 deselected`；集成 `6 passed, 27 deselected`；uv lock/sync 检查和 CLI help
通过。共享合同另为 `17 passed`，A 为 `169 passed`，C 为 `126 passed, 1 skipped`（仅缺少
可选旧版外部归档）。

尚未宣称完成的科学项：真实风险标签、船型校准、风浪流相对航向、净水深规则、限制区法律规则、
Q50/Q90/概率预测、Transformer/CNN、真实航次回放指标。当前系统只能称科研演示闭环。
