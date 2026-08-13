# B → C 发布合同

B 直接构造 C 公共 `RiskFrame`，并使用 C 公共 codec 生成：

- JSON `null` ↔ Python NaN 的唯一传输语义；
- `risk-sha256-<full digest>` 规范内容 ID；
- canonical JSON bytes 和内容摘要；
- `RiskWindowQuery` 与 `CommittedRiskWindow`。

`PersistentRiskStore` 同时实现传统 `get_window/latest_before` 和正式
`get_committed_window(query)`。commit manifest 绑定完整查询身份、起止/间隔/数量、每个 risk ID
及内容摘要。帧和 commit 都不可覆盖；同内容重复发布幂等，同 ID 或同查询不同内容拒绝。

正式编排应调用 `bind_generation_authority(run_id, simulation_clock)` 订阅公共 seek 权威；
`activate_generation()` 保留为显式低层入口和测试工具。每个 run 使用独立的共享/独占
generation fence：执行 lease 和发布持共享端，代次切换持同一 run 的独占端。因此同 run
多个执行及不同 run 可以并发，切换某个 run 的代次仍会等待该 run 的全部执行结束。store 全局
写锁只短时保护 active-generation map 和不可变文件发布，不包住 C 的规划过程。激活新代次后，
旧任务即使晚完成，也会在发布门禁被拒绝。commit 指针最后原子创建，因此崩溃时只可能留下
不可达内容，不会出现半个“已提交窗口”。

正式 C 执行应使用 `with store.lease_committed_window(query): ...` 将该 run 的共享 generation
lease 从精确 commit 读取保持到规划返回；此期间同 run 的 `activate_generation()` 会等待，避免
查询通过后、规划尚未结束时发生 seek，但新修订和其他 run 不会被 store 全局串行。持久 store
只接受 `formal` 且符合 `risk-sha256-<64 hex>` 的帧，任何 synthetic/legacy 或路径型 ID 都在
形成文件路径前拒绝。

`publish()` 和 `publish_window()` 在取得任何 store 锁之前，先把每个输入帧执行 canonical
encode→decode，形成与调用方 xarray 完全脱离的私有快照。frame bytes、window digest、manifest
和 query pointer 全程只从该快照产生；调用方在发布中途替换变量或属性不能制造相互矛盾且无法
恢复的已提交制品。

C 的正式 `execute()` 会在租约内再次取得同一 committed window，将帧经 C canonical codec
编码/解码为私有快照，再从该快照重建规划输入；因此 prepare 后对暴露 xarray 的替换不会进入
实际规划。B 的锁负责保证整个过程期间 active generation 不改变。
