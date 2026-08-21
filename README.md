---
Overall Status: ACTIVE
Content Status:
  - COMPLETED
  - PLANNED
Document Role: CANONICAL
Scope: work package B entrypoint and public boundary
Branch: research-validation-system
Last Verified: 2026-08-21
---

> [!NOTE]
> **文档治理声明**
> - 文件角色：正式工作包 B 的现行入口与边界说明。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 原文件去向：[README.archive-20260814-pre-governance.md](README.archive-20260814-pre-governance.md)。
> - 改造原因：统一正式主线、可选 CNN 后端、验证证据和后续开发入口。

# 工作包 B：逐小时环境风险场

## Research Validation 定位（2026-08-21 23:18）

B 的阶段角色为 Risk Assessment and Forecast。当前 fixed-grid risk 与 hard reason 已实现，
但仍是 `demo_unvalidated`；Adaptive Grid 和科学标定均为 PLANNED。RC2 31×11 来自显式
Tromsø 配置，不是 `TargetGridConfig()` 的全局默认。

工作包 B `0.2.0` 是 A 与 C 之间的正式风险服务：只消费 A 的公共
`PreparedWindow` / `DatasetBundle v2` 及与其匹配的 `RunContext v2`，按小时发布
`bc.risk-frame.v2`。当前实现是可审计的工程基线，风险数值仍标记为
`demo_unvalidated`，不能用于真实航行决策。

## 当前状态

| 范围 | 状态 | 可复核证据 |
|---|---|---|
| 正式规则风险流水线 | 已实现、待真实源与科学评审 | `make check`：40 个 unit/contract + 8 个 integration 测试通过 |
| 旧 CNN 可选后端 P1 | 已实现、仅显式启用 | `make model-check`：10 个 model 测试通过 |
| CNN P2 sidecar 影子运行 | 未实现 | 不得描述为已接入正式流水线 |
| 真实 A→B→C→D 全链路（Demo RC1） | 已跑通（orchestrator r6/r7，v3 四层+6h 重规划） | 见 `../work_package_a/data/output/golden/mur-v3-smoke-20260816-r6/r7/output/` |
| RC1 hard-mask 策略 | `land_sea_mask_plus_unknown_v1`（source-unknown → hard） | B 风险帧 145 帧 unknown-navigable=0 |
| RC2 hard_reason | `hard_reason`（NONE/LAND/DATA_UNAVAILABLE/OTHER）+ `missing_input_variable_counts` | 每格原因可解释；hard_mask/fail-closed 不变（RC2 分支） |
| RC2 无冰语义 | `land_sea_mask_plus_unknown_ice_free_v1`：无冰水域 ice_type/edge 中性化为 0 | 解决 Tromsø 外海起点误判 DATA_UNAVAILABLE；RC1 策略不变 |
| 科学/真船风险标定 | 未完成（非 RC1 门槛） | 见 [风险基线](docs/RISK_MODEL.md) |

统一事实口径：`22_深度学习综合风险预测模型.zip` 已整合进当前 B 仓库的可选实验后端并完成
P1，但未完成 P2 sidecar，且未进入 formal build / RiskFrame / store / C。

## 数据流与责任边界

```text
A PreparedWindow + DatasetBundle v2 + RunContext v2
                         │
                         ▼
              B 时间对齐、风险、硬掩码、置信度
                         │
                         ▼
                 bc.risk-frame.v2
                         │
                         ▼
              C 速度、ETA、代价、路径与重规划
```

B 拥有输入一致性校验、逐小时风险、不可通行掩码、置信度和环境速度因子。C 拥有最终船速、
ETA、代价函数、路径搜索和重规划。B 不读取 A 私有目录，不从文件名或 mtime 猜测时间，不把
CNN 单步输出自动伪装成正式时序 RiskFrame，也不把 `synthetic`、`legacy_unverified` 或
`demo_unvalidated` 产物写成 `formal`。

## 快速验证

```bash
cd /root/my_project/work_package_b
make env-create     # 首次创建 Mamba 前缀
make sync
make check          # 正式路径
make model-sync     # 可选：安装 CPU 模型依赖
make model-check    # 可选：仅验证 CNN P1
```

正式环境由 `environment.yml`、`uv.lock` 和本仓 `Makefile` 共同约束。不要仅凭已有 `.venv`
的测试结果宣称 Mamba + uv 可复现。

## 文档与继续开发入口

- 唯一总交接入口：[work_package_b_handoff.md](../work_package_b_handoff/work_package_b_handoff.md)
- A→B 适配：[docs/AB_ADAPTER.md](docs/AB_ADAPTER.md)
- B→C 契约：[docs/BC_CONTRACT.md](docs/BC_CONTRACT.md)
- 风险与时序：[docs/RISK_MODEL.md](docs/RISK_MODEL.md)、[docs/TEMPORAL_MODEL.md](docs/TEMPORAL_MODEL.md)
- 当前验证：[docs/VALIDATION.md](docs/VALIDATION.md)
- CNN 模型审计与分阶段计划：[docs/DELIVERED_CNN_MODEL_AUDIT.md](docs/DELIVERED_CNN_MODEL_AUDIT.md)、[docs/DELIVERED_CNN_INTEGRATION_PLAN.md](docs/DELIVERED_CNN_INTEGRATION_PLAN.md)
- 系统级权威文档：[ARCTIC_ROUTE_SYSTEM.md](../ARCTIC_ROUTE_SYSTEM.md)、[ABC_10_DAY_SPRINT.md](../ABC_10_DAY_SPRINT.md)
- 共享契约：[arctic_route_contracts/README.md](../arctic_route_contracts/README.md)

## 下一步顺序

1. 用 A 正式发布且来源完整的 bundle 做冻结场景回放，保留 UTC 时间和来源字段。
2. 完成规则风险的科学标定与阈值评审，再讨论 `demo_unvalidated` 升级。
3. 获得明确批准后才实现 CNN P2 sidecar；保持 fail-open、不可写正式 store、不可进入 C。
4. P2 证据充分后再决定是否启动 P3 多时效改造；P4 正式接入必须另行评审。
