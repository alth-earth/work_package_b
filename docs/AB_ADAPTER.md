# A → B 公共适配

正式入口是 A 的 `PreparedWindow`，其中包含 `DatasetBundle`、实际 `StandardDataFrame`、
逐 data ID payload attestation、代次、知识截止时间和逐类型 coverage。
`BInputEnvelope` 会执行五层复核：

1. 使用共享 `verify_dataset_bundle()` 从完整文档重算 v2 身份、cadence、来源和 coverage；
2. 核对 RunContext 的 bundle ID/digest、走廊和完整模拟窗口；
3. 核对 PreparedWindow 的 route、generation、as-of、类型集合和 coverage；
4. 逐一核对 bundle record 与 live frame 的 data ID、时间、来源、质量、checksum 和
   `source_snapshot_id`；
5. 用 A 公共 `semantic_payload_digest()` 独立重算每个实际 payload，要求与 attestation
   一致后深拷贝；build 读取前再次复核并形成新的私有快照。

任何 future issue、缺 payload、额外帧、漏帧、旧 generation、内容篡改或不完整类型都明确失败。

同进程直接传 A `prepare_window_for_b()` 的返回值。持久/跨进程路径只调用 A 公共
`resolve_dataset_bundle_for_b(bundle, generation_id=..., knowledge_as_of=...)`；B 不读取 A
SQLite、ready/raw、缓存或 manifest 内部结构。

两个运行态参数必须来自编排器的当前权威快照，而不是从 PreparedWindow 缺省回填。A
prepare 完成后若发生 seek，B store 通过 `bind_generation_authority()` 跟随公共时钟切换并
拒绝旧代次发布。
