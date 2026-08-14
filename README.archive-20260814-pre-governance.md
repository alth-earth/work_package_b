> [!NOTE]
> **治理归档声明**
> - 文件角色：正式工作包 B 的治理前入口文档，仅供历史审计。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 现行去向：[README.md](README.md)。
> - 改造原因：旧文档同时包含现状、计划与已过时的 CNN 状态，需由职责单一且与当前实现一致的入口替代。
> - 内容保护：下方标记之后为归档前原正文，逐字保留。

<!-- ORIGINAL CONTENT START -->
# 工作包 B：逐小时环境风险场

工作包 B 消费 A 公共接口提供的、与同一个共享 `RunContext` 精确绑定的
`PreparedWindow + a.dataset-bundle.v2`，发布完整逐小时 `bc.risk-frame.v2` 窗口供 C
规划。当前 `0.2.0` 是可审计的确定性规则基线，明确标记为 `demo_unvalidated`：它没有经过
事故、AIS 或真实船舶数据校准，不能用于真实导航。

2026-08-14 收到的新 CNN ZIP 已按固定哈希受限转换为 safetensors，并完成 CPU 单步短测试。它
基于旧 B 单通道综合风险、只输出未声明时长的下一步二维场；当前状态仍为
`experimental_unverified`，只作为 opt-in shadow 后端，不替换规则主线、不生成逐小时窗口、不
进入 store/C。审计、整合方案和完成状态见 [新模型静态审计](docs/DELIVERED_CNN_MODEL_AUDIT.md)、
[整合方案](docs/DELIVERED_CNN_INTEGRATION_PLAN.md) 与
[续开发 Handoff](../work_package_b_handoff/工作包B-新风险模型整合与续开发Handoff.md)。

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

模型 CPU extra 是可选的，不影响默认规则环境：

```bash
make model-sync
make model-check
```

转换资产位于 `models/legacy_cnn_one_step_v1/`，其中 safetensors SHA-256 为
`602d4849ce0b2f5eda90db2a020b8ec69a00dc929d5e764ed412d0d03e205103`。原始 ZIP、`.pth`、脚本、
训练数据和 `.pyc` 不进入仓库。资产的上游许可证未随交付提供，但用户已明确授权其于
2026-08-14 在本公开仓库中再分发；详见 [资产声明](models/legacy_cnn_one_step_v1/MODEL_ASSET_NOTICE.md)。

`LegacyCnnOneStepBackend` 只读 safetensors、固定 CPU/eval/inference mode，接受严格 0.05°
的 `comprehensive_risk` 二维网格，输出 `predicted_valid_time=None` 和
`time_step_status="unknown"`。`RiskBuildService` 仍直接使用规则函数，故本后端不会改变正式
RiskFrame、store 或 C 链路。

## 公共 API

```python
from datetime import timedelta

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
    "configs/models/demo_unvalidated_v2.json"
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
    suffix_commit = store.publish_suffix_window(
        frames,
        start=run_context.simulation_start + timedelta(hours=6),
    )
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

0.2.0 的严格配置 v2 明列 11 个风险分量的权重、变换和归一化上下界，以及来源质量、时间
处理方法、陆地阈值、风险—速度系数和最低环境速度因子。缺字段、额外字段、未知分量/变换、
非有限数值、非法范围或非归一权重都会拒绝；所有数值叶子都进入
`b.model-config.v2` 摘要。默认值保持 0.1.0 的输出数组语义不变，但模型版本和摘要按发布规则
升级。
旧 `demo_unvalidated_v1.json` 仅为 0.1.0 审计快照，当前严格 loader 不会把它静默升级为 v2。

当前未冻结的 0.2.0 工作树暂用纬向 `0.75°`、经向 `2.2°` 作为 **fast smoke 候选**。按约
70°N 的 `cos(latitude)` 粗略换算，两者约对应 83 km 与 84 km；两条走廊的空 hard-mask
端点允许区均含节点，主走廊降为 `11×26`。但最近一次全链运行中，C 初始三目标仍约 27 分钟，
重规划未完成；也没有真实 land mask、路线保真或科学分辨率证据。因此该值暂留用于合同和失败
定位，不再称为正式默认。formal 候选 `0.5°×1.5°`、模型 native `0.05°×0.05°` 与 fast
网格的冻结门槛见 [整合方案](docs/DELIVERED_CNN_INTEGRATION_PLAN.md)。所有网格政策仍进入
`model_config_digest`，实际坐标由 `grid_id` 绑定。

`publish_suffix_window()` 只对一组完整、formal、规范 ID、严格逐小时的帧做闭区间后缀提交。
起点必须精确命中已有帧；它不会重建 RiskFrame，因此适合 +6 h 重规划复用同一 generation、
as-of、模型摘要和 risk IDs。

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
  历史 v1、旧走廊、9 类，不能创建正式 RunContext；新的完整 12 类 `DatasetBundle v2` 由
  独立数据采集会话交付，在它通过接收门之前尚未完成真实来源 A→B→C 验收；
- 当前规则仍是 `demo_unvalidated`，没有训练权重、真实风险标签或航次校准证据。

外部 CNN 权重的存在不改变最后一条：它尚未集成，训练标签仍是旧 B 启发式风险，且没有独立
验证。2026-08-14 长运行的时间线、最后成功制品和下一次运行规则见
[orchestrator 事故复盘](../arctic_route_orchestrator/docs/INCIDENT_2026-08-14_LONG_INTEGRATION_RUN.md)。
