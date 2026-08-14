> [!NOTE]
> **治理归档声明**
> - 文件角色：新交付 CNN 的治理前整合计划，仅供历史审计。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 现行去向：[DELIVERED_CNN_INTEGRATION_PLAN.md](DELIVERED_CNN_INTEGRATION_PLAN.md)。
> - 改造原因：旧计划混有“尚未实现”和“P1 已实现”两种时点，需要按当前阶段重新建立单一状态口径。
> - 内容保护：下方标记之后为归档前原正文，逐字保留。

<!-- ORIGINAL CONTENT START -->
# 新交付 CNN 与当前工作包 B 的整合方案

> 方案日期：2026-08-14
>
> 当前阶段：设计冻结，尚未实施模型代码接入
>
> 前置审计：[DELIVERED_CNN_MODEL_AUDIT.md](DELIVERED_CNN_MODEL_AUDIT.md)

## 1. 已冻结决策

1. 当前确定性规则基线继续作为 B 的默认正式工程主线；新 CNN 先作为 opt-in
   `experimental_unverified` 后端，不能替换或伪装成该规则模型。
2. A 的 `PreparedWindow/DatasetBundle v2/RunContext v2`、BC `RiskFrame v2`、B store 和 C
   ingress 均保持不变。模型差异只在 B 内部实现，诊断走 sidecar/run report。
3. 当前权重只允许 CPU、有界、单步 shadow 推理；不支持递归 168 h，不宣称原生逐小时。
4. 当前代码中的 `0.75° × 2.2°` 暂时保留为 **fast smoke 候选**，不是已冻结 formal 科研网格；
   模型 native grid、B publish grid 和 C fast grid必须分开管理。
5. 本方案完成后停在实施门前；只有用户确认后才新增后端代码、依赖、权重接收或测试。

## 2. 目标架构

```text
A PreparedWindow + payload attestations + RunContext
                         │
                         ▼
BInputEnvelope：身份 / coverage / as-of / generation / 内容证明
                         │
                         ▼
B 现有时间引擎：按可见支撑生成逐小时时刻输入
                         │
                         ▼
                  RiskModelBackend
       ┌─────────────────┴──────────────────┐
       │                                    │
RuleBaselineBackend                LegacyCnnOneStepBackend
当前默认、正式工程主线             opt-in、CPU、shadow、单步
       │                                    │
       └─────────────────┬──────────────────┘
                         ▼
B 公共后处理：land hard mask / risk_level / confidence / speed factor
                         │
                         ▼
RiskFrame v2 → canonical ID → PersistentRiskStore → C v2/v3 ingress
```

新模型不直接读取 A、不开私有路径、不自行发布 RiskFrame。旧 B NetCDF 的兼容只出现在受门禁的
适配器/评估夹具中，不能变成正式运行输入。

## 3. P1 后端接口（已实现，仍不接正式调用链）

以下接口已在 `arctic_route_risk.modeling` 实现；它们是 opt-in 适配面，不会被正式
`RiskBuildService` 自动调用：

```python
class RiskModelBackend(Protocol):
    backend_id: str
    model_version: str
    required_variables: tuple[str, ...]

    def infer(self, model_input: RiskModelInput) -> RiskModelOutput: ...
```

`RiskModelInput` 至少绑定：

- `valid_time` 和明确 lead/cadence 语义；
- 严格排序的变量、数组、dtype、单位、缺测掩膜；
- native/output 经纬度、CRS、坐标摘要和重采样政策；
- A 来源质量、时间处理方法、source references；
- 当前规则基线摘要（仅当 CNN 输入是规则 risk 时）。

`RiskModelOutput` 只允许返回 `risk_score`、可选 `applicability_confidence` 和诊断 sidecar。
模型不得直接决定 `hard_mask`，也不得绕过 B 的未知值、来源和 canonical 身份门禁。

兼容策略：

- 当前 `b.risk-build-configuration.v2` 隐式对应 `RuleBaselineBackend`，不改变其历史摘要；
- 后续新增严格 backend-aware 配置版本，显式选择后端和资产；
- 没有 backend 字段时不能暗中加载权重；
- 模型输出的任何诊断变量不得加入 `RiskFrame v2` payload；C 不感知后端类型。

## 4. 逐小时时间策略

### 4.1 当前权重允许的行为

当前权重只能证明“单张风险网格 → 下一步风险网格”，且步长未绑定。旧 B 数据谱系可能是
6 h，但不能据此把 checkpoint 标成已确认 6 h，更不能标成 1 h。

第一阶段只做 shadow：

1. 规则后端照常生成并发布完整逐小时 `RiskFrame`；
2. CNN 在明确选择的旧谱系/对照样本上做单步推理；
3. CNN 结果写独立诊断制品，不进入 C committed window；
4. 记录输入/输出摘要、耗时、峰值内存和与 persistence/rule target 的误差。

明确禁止：

- 把一张输出复制 169 次；
- 递归滚动 169 步并称为 168 h 预测；
- 把 6 h 或未知步长输出直接改时间戳为 1 h；
- 利用未来时刻的旧标签而不执行 as-of 门禁；
- 把当前 B 逐小时连续化误写成 CNN 原生逐小时预测。

### 4.2 晋级到逐小时后端的条件

只有补齐训练 cadence、源文件哈希和独立评估后，才从下列路径选择一个：

- **真正 1 h 训练/微调**：每个小时输入对应一个小时 target，最容易与 B 合同一致；
- **已确认 6 h anchor**：只在 6 h anchor 产生模型结果，小时化政策、置信度衰减和误差需单独
  版本化并由负责人确认；
- **每小时 A 环境先形成规则 risk，再做 CNN 单步后处理**：必须重新训练/验证，证明当前规则
  risk 的分布和旧标签兼容，不能直接套用现权重。

在任何路径中，169 帧是 B 按 RunContext 物化的发布数量，不代表网络本身一次输出了 169 步。

## 5. hard mask、等级、置信度和速度影响

- `hard_mask` 始终来自 B 现有正式 `land_sea_mask` 及未来明确批准的硬约束，不从 CNN 学习。
- `risk_level` 继续按 BC v2 已冻结的等宽规则从最终 `risk_score` 派生。
- `confidence` 组合 A 质量、时间方法、模型适用域和评估证据。当前权重没有可用模型置信度，
  因而只能 shadow，不能人为填 1。
- `environment_speed_factor` 仍由 B 的版本化后处理生成，最终速度/ETA 仍归 C。
- 任一必需输入未知时保持 NaN/零置信度；不得复用交付脚本的 `nan_to_num(..., nan=0)`。

## 6. 三网格策略

| 层次 | 当前决策 | 状态/用途 |
|---|---|---|
| model native grid | 旧谱系 `0.05° × 0.05°` | 只用于模型审计和 shadow；不得假定任意网格等价 |
| formal publish 候选 | `0.5° × 1.5°` | 约 55×57 km；待 C 性能优化、真实 mask 和路线对比后决定 |
| fast smoke 候选 | `0.75° × 2.2°` | 约 83×84 km；暂留当前配置用于合同/失败定位，不是路线质量基线 |

`TargetGridConfig.realize()` 通过 `ceil + linspace` 覆盖 bbox，所以配置值是最大角度步长政策，
实际坐标间距由 bbox/shape 决定。文档和报告不得把它误写成所有走廊完全相同的实际分辨率。

保留 `0.75° × 2.2°` 的依据只有：两条走廊在空 hard mask 下端点允许区均可物化，节点数量显著
下降。尚缺：真实 land mask 端点、完整 v2/v3 耗时、路线保真、风险混叠以及与
`0.5° × 1.5°` 的对照。条件满足前，它不能自动随当前代码成为 formal 默认。

新 CNN 不决定 C 规划网格。若在 native 细网格推理，必须显式记录 native→publish 的重采样方法、
unknown mask 处理和抗混叠政策；若在粗网格重训，必须生成新的权重身份和评估，不能复用现有
模型声明。

## 7. `model_config_digest` 必须绑定的内容

后端-aware 摘要至少覆盖：

- backend ID、架构版本、checkpoint SHA-256、受限加载/转换格式版本；
- 模型输入变量和顺序、单位、dtype、归一化、NaN/无穷处理；
- 模型 cadence、lead、递归/非递归政策、首次帧政策；
- native grid、输入重采样、native→publish 重采样；
- `risk_score` 校准映射、hard/confidence/speed 后处理；
- publish grid 政策和数值确定性/provider 政策。

实际 bbox 和实现后的坐标仍由每帧 `grid_id`/canonical risk ID 绑定。fast/formal 网格必须产生
不同摘要。运行报告另记录 provider、CPU/GPU 型号、Torch/native 库版本和实际耗时；若 provider
会改变可接受数值模式，provider policy 也进入摘要。

## 8. 环境与资产接收

首版只在 B 的 Mamba + uv 体系增加一个显式可选 CPU extra，不把未锁版本的 ZIP
`requirements.txt` 直接安装到主环境。实施阶段必须：

1. 校验 ZIP 和 checkpoint 哈希；
2. 确认许可/所有权；
3. 在隔离进程中进行受限 weights-only 读取，白名单校验 8 个键、shape 和 dtype；
4. 转换为审计友好的纯权重制品或保存可重复的转换记录；
5. 锁定 Torch/NumPy/xarray 版本并做 CPU 确定性复现；
6. 不提交训练数据、缓存、凭据或无审计 `.pyc`。

CUDA 暂不进入必需环境。只有 native 细网格批量 CPU 基准超过明确预算时，才增加独立可选
CUDA extra；CPU 仍保留合同 smoke 和可复现基准。

## 9. 分阶段实施与晋级门槛

| 阶段 | 允许动作 | 完成证据 |
|---|---|---|
| P0：本次设计 | 静态审计、架构、事故复盘、文档 | 本方案及 Handoff；无代码接入 |
| P1：安全 intake | 资产描述符、受限加载、CPU 单步适配、小数组测试 | 键/shape/hash/确定性/NaN 负例 |
| P2：shadow | 不改正式窗口，生成诊断 sidecar | cadence 已声明；holdout/persistence 对照；两走廊报告 |
| P3：候选后端 | opt-in 生成完整 RiskFrame，规则默认保留 | 169 帧、摘要、provenance、store、C ingress 全通过 |
| P4：正式晋级 | 经负责人批准才切换默认 | 独立验证、实源 A 回放、网格/性能/科学门槛均通过 |

P1/P2 不依赖真实 A 168 h bundle，可用明确标识的夹具/旧谱系审计数据完成。P3/P4 的正式实源
验收必须等待 A 的完整 12 类 `DatasetBundle v2`，旧 v1/9 类 bundle 永远不能替代。

## 10. 负例和错误语义

实施后至少覆盖：错 checkpoint 哈希、未知 key/shape、额外配置、错 backend、未知 cadence、
输入 grid/dtype/变量顺序不符、NaN 被当安全、输出越界/非有限、provider 数值不稳定、旧 corridor
ID、未来信息、缺帧、旧 generation、模型摘要与 store query 不一致。

失败必须返回明确错误并保持规则主线/已有 commit 不变。任何失败都不得回退到“复制最后一张”、
静默填零、自动换后端或重新归一化。

## 11. P1 完成状态与当前暂停点

P1 已完成：固定哈希 intake、manifest、safetensors 转换、CPU 单步后端、规则 parity 适配器、
负例和确定性短测试均已落盘。转换资产 SHA-256 为
`602d4849ce0b2f5eda90db2a020b8ec69a00dc929d5e764ed412d0d03e205103`；`make model-check` 为
10 passed。默认规则环境不安装 Torch，`RiskBuildService`、RiskFrame/store/C 链路、0.75°×2.2°
fast-smoke 网格和 golden digest 均未改变。

P2–P4 仍暂停：未生成 shadow sidecar，未证明 cadence/holdout/persistence，未进入逐小时
RiskFrame，未运行 169 步 CNN、C 长规划、真实 A 下载或 CUDA。完整任务状态、阻塞和下一步见
[工作包 B 新风险模型整合与续开发 Handoff](../../work_package_b_handoff/工作包B-新风险模型整合与续开发Handoff.md)。
