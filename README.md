# 工作包 B：逐小时环境风险场

工作包 B 消费 A 公共接口提供的、与同一个共享 `RunContext` 精确绑定的
`PreparedWindow + a.dataset-bundle.v2`，发布完整逐小时 `bc.risk-frame.v2` 窗口供 C
规划。当前 `0.1.0` 是可审计的确定性规则基线，明确标记为 `demo_unvalidated`：它没有经过
事故、AIS 或真实船舶数据校准，不能用于真实导航。

## 一眼看懂流程

```text
A PreparedWindow + payload attestations + shared RunContext
              ↓  BInputEnvelope 严格身份/coverage/as-of/内容门禁
       版本化目标网格 + 可见支撑帧逐小时连续化
              ↓  demo_unvalidated 风险/置信度/环境速度影响
       C 公共 canonical codec + risk-sha256 内容身份
              ↓  私有发布快照 + per-run generation fence
                    + 原子 committed-window 指针
       PersistentRiskStore → C RiskSourcePlanningIngress
```

B 不下载数据、不读取 A 私有 SQLite/目录，不生成路线、最终船速或 ETA。首版 hard mask 仅由
A 的正式 `land_sea_mask` 中“陆地/海岸”产生；`source_valid_mask` 不具有这一语义。

## 环境

Mamba 固定 Python 3.13 和 NetCDF/HDF5 原生库，uv 解析、锁定并安装 Python 依赖：

```bash
make env-create
make sync
make check
make integration
```

已有 Python 3.13 和 uv 时，可以先执行 `uv sync --locked`。缓存和环境都在本工程目录内，
不会把本地绝对路径写入运行合同。

## 公共 API

```python
from arctic_route_risk import (
    BInputEnvelope,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
    load_risk_build_configuration,
)

runtime_snapshot = simulation_clock.snapshot()
envelope = BInputEnvelope.from_prepared_window(
    run_context=run_context,
    prepared_window=prepared_window,
    generation_id=runtime_snapshot.generation_id,
    knowledge_as_of=knowledge_as_of,
)
configuration = load_risk_build_configuration(
    "configs/models/demo_unvalidated_v1.json"
)
request = RiskBuildRequest(
    envelope=envelope,
    target_bbox=(west, south, east, north),
    grid_config=configuration.grid_config,
    model_config=configuration.model_config,
)
frames = RiskBuildService(utc_now=lambda: generated_at).build_window(request)

store = PersistentRiskStore("data/bc-risk")
unbind = store.bind_generation_authority(run_context.run_id, simulation_clock)
try:
    commit = store.publish_window(frames)
finally:
    unbind()
```

正式 build 不允许用参数静默裁剪 RunContext 全窗。持久回放通过 A 公共
`WorkPackageA.resolve_dataset_bundle_for_b(...)` 恢复精确 payload，再走同一个 envelope；B
不会自行解析 A 的数据库或归档路径。

`generation_id` 和 `knowledge_as_of` 必须由运行编排显式传入，不能从旧的
`PreparedWindow` 反推为当前权威状态。`bind_generation_authority()` 订阅公共 simulation clock
的 seek；旧任务即使在 A prepare 或 B build 后才完成，也会在提交门禁被当前代次拒绝。
同一 run 的 C 执行 lease 共享 generation fence，代次切换使用其独占端；同 run 新修订和不同
run 的规划不会被一个全局长锁串行，只有 generation map/不可变制品写入短时使用 store 写锁。

发布入口还会先用 C canonical codec 编码并解码所有输入帧，形成私有快照。后续 frame、manifest
和 pointer 只引用该快照，调用方在发布中途替换可检查 xarray 内容不会污染已提交窗口。

A 的 `payload_attestations` 逐 data ID 绑定完整 manifest record 与规范 payload。B 在创建
`BInputEnvelope` 时独立重算、做私有深快照，并在 `build_window()` 真正读取前再次重算和
快照。record 元数据不变但数组/坐标被替换、或外部别名试图恢复 NumPy 可写位时，都不能让
旧 attestation 继续生成正式帧。

默认策略文件通过严格公共 loader 解析；缺字段、额外字段、未知版本或非法值都会拒绝。
`model_config_digest` 从解析后的网格/模型政策计算，不存在“JSON 写一套、运行默认另一套”的
隐式路径。

`model_config_digest` 只绑定网格政策、时间政策和风险规则，不绑定具体走廊 bbox 或已实现坐标，
因此同一配置可在两条走廊保持相同摘要；每个 RiskFrame 的 `grid_id` 则绑定实际坐标。

## 当前风险基线

风险由冰密集度/厚度/类型/冰缘/漂移、浪、流、风、温度、能见度和水位的简单归一化分量
确定性融合。权重只是工程占位符，没有科学校准。`risk_level` 精确遵循 C v2 当前规则：

```text
min(5, floor(risk_score * 5) + 1)
```

未知点保留 `risk_score=NaN`、`confidence=0`，不会 `fillna(0)`；环境速度因子始终在
`(0,1]`，最终速度和 ETA 仍由 C 负责。

进一步边界、算法和验收说明见 [docs/](docs/)。

## 当前工程证据与限制

- 公共接口夹具已覆盖 12 类必需 A 输入、96/168/216 h 动态闭区间（97/169/217 帧）、
  两走廊同一 `model_config_digest`、B 原子持久窗口以及 C 正式 ingress/RoutePlan 发布；
- 归档测试已覆盖 A 正式发布、进程重启、公共 exact-bundle resolver、B build/commit；B store
  重建后可恢复 commit 且跨实例重复发布幂等，并由真实 clock authority 保证 seek 后旧 publish/get
  拒绝；A 恢复后 payload 篡改同样拒绝；
- 上述数据和 source snapshot 都是可复核测试夹具。当前工作区唯一真实长窗 A bundle 是
  历史 v1、旧走廊、9 类，不能创建正式 RunContext，因此尚未完成真实来源 A→B→C 验收；
- 当前规则仍是 `demo_unvalidated`，没有训练权重、真实风险标签或航次校准证据。
