---
Overall Status: ACTIVE
Content Status:
  - COMPLETED
  - IN_PROGRESS
Document Role: CANONICAL
Scope: Work Package B risk-explanation.v1 research producer interface and semantics
Canonical For: B component trace capture and research sidecar export behavior
Branch: research-validation-system
Last Verified: 2026-08-23
Related Canonical Docs:
  - ../README.md
  - RISK_MODEL.md
Related Supporting Docs:
  - ../../arctic_route_governance/reports/research-validation/RISK_EXPLANATION_SIDECAR_DESIGN_REPORT.md
---

# B Risk Explanation Research Exporter

## 定位与职责（2026-08-23 21:40 +08:00）

`RiskExplanationResearchExporter` 是 B 拥有的研究输出器，用于生成可选
`risk-explanation.v1` JSON sidecar。它解释当前 `demo_unvalidated` 加权规则怎样得到某个格点
的 `risk_score`，不修改、替代或扩展 `bc.risk-frame.v2`。

生产者职责包括：在 B 公式实际求值点捕获 component 证据；将解释绑定到已经原子提交的
`CommittedRiskWindow`；对 frame/time/grid/risk 做交叉验证；按 `COMPLETE`、`PARTIAL`、
`UNAVAILABLE` 发布；任何不能证明的解释失败关闭。它不负责科学因果推断，也不解释 C 的
route objective、ETA、候选比较或 selection。

## Producer 接口（2026-08-23 21:40 +08:00）

生产流程分为两个明确阶段：

```python
result = RiskBuildService(...).build_window_with_explanation_trace(request)
committed = store.publish_window(result.frames)
document = RiskExplanationResearchExporter(...).export(
    committed_window=committed,
    build_result=result,
)
```

`build_window_with_explanation_trace()` 的输入仍是现有 `RiskBuildRequest`，只消费 A 公共
`PreparedWindow` / `DatasetBundle.v2` 与匹配的 `RunContext`。它的输出
`RiskBuildTraceResult` 包含两部分：

- 未改变的 `tuple[RiskFrame, ...]`；
- B 私有研究 trace：每个公式 component 的 `normalized_value` 二维数组、配置 `weight`、
  以及同次求值得到的 `contribution = normalized_value * weight`。

`export()` 输入是已提交 `CommittedRiskWindow` 与带完整性摘要的 `RiskBuildTraceResult`；输出是 JSON-ready
`dict[str, Any]`，`schema_version = risk-explanation.v1`。输出器不读取文件系统中的 D artifact，
也不接受只含最终 `risk_score` 的输入来重建贡献。

`RiskBuildTraceResult` 通过 B pipeline 内部 factory 封装；摘要绑定 frames、窗口身份、坐标、
validity mask、全部 normalized arrays、weights 和 contributions。对象被 `replace()` 或数组重分配
后，完整性校验失败，不能进入 exporter。这是防止“总和相同但 component attribution 被调换”
的工程完整性门禁，不是密码学来源认证或跨进程签名。

现有 DRAFT v1 Schema 的公开 `contributor` 只包含 group-level `contribution` 与
`component_ids`，不允许额外的 `normalized_value` / `weight` 字段。因此这两类审计值保留在 B
trace API，Sidecar 仅发布与 Schema 一致的加性贡献。若未来 consumer 必须直接消费它们，应
先由 contract owner 提出新版本或兼容扩展，不能由 B 单方面向 strict v1 JSON 塞入未知字段。

## Identity binding（2026-08-23 21:40 +08:00）

输出前必须精确绑定：

| Sidecar 字段 | 权威来源 | 校验规则 |
|---|---|---|
| `identity.risk_window_id` | `CommittedRiskWindow.commit_id` | 原样写入；不得使用未提交 frames |
| `frames[].risk_frame_id` | `RiskFrame.risk_id` | 与 trace frame ID 完全相等 |
| `frames[].frame_time` | `RiskFrame.valid_time` | UTC 且完全相等 |
| `frames[].grid.grid_id` | `RiskFrame.grid.grid_id` | 同时要求 latitude/longitude 数组逐值相等 |
| run/scenario/corridor/vessel/config/model/generation/as-of | committed window + trace | 任一不一致拒绝整个 Sidecar |
| `risk.score/level/confidence` | `RiskFrame.payload` | 只读镜像；贡献和校验用途，不覆盖 RiskFrame |

identity mismatch、坐标错位、公式版本错配或完整解释的贡献和不等于 RiskFrame score（绝对容差
`1e-6`）时，`export()` 抛出 B pipeline error，不返回部分可信 JSON。Sidecar consumer 可因此
关闭解释功能，同时继续使用原有 RiskFrame/Viewer。

## 数据来源与 contributor 生成（2026-08-23 21:40 +08:00）

真实数据流是：

```text
A public frames
    -> B temporal alignment / regrid
    -> _risk_component()
    -> normalized_value
    -> configured weight
    -> weighted contribution
    -> same-call risk_score
    -> B research trace
    -> risk-explanation.v1 exporter
```

当前 11 个 component 按展示语义汇总为 `ice`、`wave`、`current`、`wind`、`freezing`、
`visibility`、`water_level`。`ice` 保留其实际覆盖的 5 个 `component_ids`，因此分组不会隐藏
公式覆盖情况。`COMPLETE` 格点必须覆盖全部 11 个 component、land/sea validity 有效，且 group contribution 之和等于
RiskFrame score。权重不重归一化，缺项不补零。

## 状态与缺测语义（2026-08-23 21:40 +08:00）

| 状态 | 条件 | 输出行为 |
|---|---|---|
| `COMPLETE` | 11 个 component 全部有限，贡献和匹配有限 `risk_score` | 发布全部 contributor 与 B 生成的主要贡献原因 |
| `PARTIAL` | 至少一个 component 有真实证据、至少一个缺失 | 仅发布有限 component；缺项写入 `missing_data` 与 `explanation_gaps` |
| `UNAVAILABLE` | 所有 component 均无有限证据 | `contributors=[]`，明确发布 explanation unavailable |

`land_sea_mask` 是 RiskFrame validity/hard-mask 依赖，不是加权 contributor。若它非有限，即使 11
个公式 component 可计算，producer 也把该格解释设为 `UNAVAILABLE`、不发布 contributor，并在
`missing_data` 中明确记录 `land_sea_mask`；不能把“公式可算”误写成“RiskFrame 风险有效”。

缺失 component 的 `normalized_value` 与 `contribution` 在 trace 中保持 NaN；JSON 中该 component
缺席，风险快照按 RiskFrame 表达为 `score=null`、`level=5`、`confidence=0`。任何层都禁止将其
转换为 `0`、静默重归一权重或猜测自然语言解释。

对当前加性公式，有限 `RiskFrame.risk_score` 必然要求全部 11 个 component 有限。因此“score
有限但 trace 缺 component”只能说明 trace 被破坏或不是同次计算，exporter 会拒绝整个 Sidecar，
不会把它降级成 `PARTIAL`。Schema 虽可表达更一般的 finite-score partial attribution，但该语义
留给未来经审阅的原生 attribution producer，不能由当前 exporter 猜测。

## 为什么不能由 D 生成解释（2026-08-23 21:40 +08:00）

RiskFrame 只保存最终 score、level、confidence 与摘要，数学上不能唯一反解 11 个 component。
D 若读取 A/B 私有输入或重复公式，会复制模型政策、造成版本漂移，并可能在缺测格点制造虚假
贡献。D 的合法职责只有：校验 Sidecar 与正在显示的 RiskFrame 身份，显示 producer 已发布的
contributors/reason/status；Sidecar 缺失或无效时关闭解释区域，不影响基础展示。

## 验证与复核（2026-08-23 21:40 +08:00）

核心自动化覆盖：

- 现有治理提案 Draft 2020-12 Schema 兼容；
- `sum(contribution) == risk_score` 正例与错配拒绝；
- window/frame/grid identity mismatch 拒绝整个 Sidecar；
- 单 component 缺测时 `PARTIAL` 且不补零；
- 全 component 缺测时 `UNAVAILABLE` 且不发布 contributors；
- trace component 缺项或乱序失败关闭；
- trace 完整性摘要拒绝同和异构的 component attribution 调换；
- `land_sea_mask` 缺测明确发布 `UNAVAILABLE`；零风险完整格显示“无主要贡献项”。

执行完整 B 门禁：

```bash
make lint
make test
make integration
make check
git diff --check
```

## Limitations（2026-08-23 21:40 +08:00）

- 仅支持当前 `deterministic_environment_components_v2` 加性公式；非加性模型没有经审阅的原生
  attribution 时必须保持不可用。
- 输出器是 research-only，没有独立 immutable Sidecar store、canonical sidecar ID、manifest、
  Orchestrator transport 或 D consumer。
- `normalized_value` / `weight` 目前只在 B trace API 中，不属于 strict v1 JSON；这是现有 Schema
  的明确兼容边界。
- 解释说明工程公式的数值构成，不证明权重科学有效、环境因素具有因果关系或航线适航。
- 完整逐格 JSON 的体积、压缩、分帧索引、按需加载和真实 Viewer 性能尚未验证。
- 格点风险解释不等于路线选择解释；完整回答仍需 C 已发布的 objective、route metrics、hard
  constraint 和候选比较证据，但 C 不消费本 Sidecar。
