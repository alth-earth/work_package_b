---
Overall Status: ACTIVE
Content Status:
  - COMPLETED
  - IN_PROGRESS
Document Role: CANONICAL
Scope: Work Package B validation evidence and acceptance boundary
Canonical For: current B validation commands, results, and maturity limits
Branch: research-validation-system
Last Verified: 2026-09-02
---

> [!NOTE]
> **文档治理声明**
> - 文件角色：正式工作包 B 的现行验证记录与验收边界。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 原文件去向：[VALIDATION.archive-20260814-pre-governance.md](VALIDATION.archive-20260814-pre-governance.md)。
> - 改造原因：以本次实测结果纠正旧计数，并分开正式路径和 CNN 可选后端证据。

# 正式工作包 B 验证记录

## 2026-09-02 当前集成复核

本次复核仅核对现有 C endpoint fail-closed 语义和 B 集成夹具，未修改 C 的 endpoint 门禁、
corridor、网格定义或任何已发布航线制品。

| 命令 | 结果 | 说明 |
|---|---|---|
| `.venv/bin/python -m pytest -m integration -q` | `8 passed, 1 warning, EXIT_CODE=0` | 当前 B 集成通过；warning 为可选 ecCodes 后端未安装 |
| `allowed_region_has_no_grid_node` 负向用例 | PASS | 粗网格无允许区域节点时按预期拒绝，验证 fail-closed |

`allowed_region_has_no_grid_node` 只在专门的粗网格负向用例中预期出现；它表示该输入网格无法
覆盖允许区域，不表示 B 公式或当前 Winter Viewer 航线失败。下方 2026-08-23 的 `7 passed,
1 failed` 是当时的历史快照，保留用于审计，不得替代本次复核结果。

## 2026-08-23 Risk Explanation Research Exporter（2026-08-23 21:40 +08:00）

本轮 verdict：`UNIT_PASS / FULL MAKE GATE BLOCKED`。在仓库现有 Python `3.13.15` `.venv`
中，risk explanation 相关 11 项测试全部通过；排除并发、非本任务 `calibration_shadow` 文件后，
本轮相关 unit/contract 集为 `62 passed`。验证包括：

- 与治理仓库 Draft 2020-12 `risk-explanation.v1.schema.json` 兼容且可严格 JSON 序列化；
- 同次 B 公式求值 trace 的 `normalized_value * weight == contribution`；
- `sum(contribution) == RiskFrame.risk_score` 与故意错配失败关闭；
- RiskWindow / RiskFrame / frame time / grid identity binding；
- 单 component 缺测为 `PARTIAL` 且不补零；全部 component 缺测为 `UNAVAILABLE` 且无 contributor；
- `land_sea_mask` validity 缺测为 `UNAVAILABLE`；零风险不虚构主要 contributor；
- sealed build trace 拒绝同和异构 attribution 修改；
- research trace build 与原 `build_window()` 的 canonical RiskFrame bytes 完全一致。

执行证据：

| 命令 | 结果 | 说明 |
|---|---|---|
| `.venv/bin/ruff check src tests` | PASS | 当前工作树 Ruff 通过 |
| `.venv/bin/pytest -q tests/unit/test_risk_explanation.py` | `11 passed` | producer 专项 |
| 明确列举本轮相关 contract/unit 文件 | `62 passed` | 排除并发 `calibration_shadow` 文件 |
| `.venv/bin/pytest -m integration -q` | `7 passed, 1 failed`（历史快照） | 当时失败为既有 Murmansk–Dikson default grid 无 destination allowed-region node；本轮未改 C/走廊/网格 |
| `make lint/test/integration/check` | BLOCKED | `uv run --locked` 报 `uv.lock needs to be updated`；本轮未改 `uv.lock` 或上游 A 依赖 |

该验证成熟度为 `UNIT_PASS`，不是 `SMOKE_PASS`、`REAL_E2E_PASS` 或生产 Sidecar 发布证据。
未运行真实 committed Winter window exporter、Sidecar store、Orchestrator transport、D consumer、
Browser E2E、12h/24h replay 或性能/体积 benchmark。

## 2026-08-14 可复核结果

在 `${ARCTIC_ROUTE_ROOT}/work_package_b` 的现有有效 Mamba 前缀与锁定 uv 环境中执行：

| 命令 | 结果 | 覆盖边界 |
|---|---|---|
| `make check` | 通过 | Ruff；40 个 unit/contract；8 个 integration；lock、sync、CLI help |
| `make model-check` | 10 个通过 | 可选 CPU CNN 的装载、输入适配和单步后端检查 |
| `git diff --check` | 通过 | 当时工作树的空白/补丁格式检查 |

`make check` 的测试划分为 `40 passed, 18 deselected` 和
`8 passed, 50 deselected`；`make model-check` 为 `10 passed, 48 deselected`。

## CNN 整合边界

`22_深度学习综合风险预测模型.zip` 已整合进当前 B 仓库的可选实验后端并完成 P1，但未完成
P2 sidecar，且未进入 formal build / RiskFrame / store / C。

因此，10 个模型测试只证明受控输入下的权重装载、适配和单步推理，不证明以下事项：

- 不证明 CNN 是正式风险来源，也不证明其输出可写入正式 `PersistentRiskStore`；
- 不证明单步模型具备 169 个逐小时有效时刻或任何已知预报间隔；
- 不证明 CNN 结果已发送给 C；
- 不证明模型科学有效、完成标定或适航认证。

## 已验证与未验证矩阵

| 能力 | 工程状态 | 验收结论 |
|---|---|---|
| A v2 公共对象校验、时间/上下文绑定 | 自动测试覆盖 | 工程基线通过 |
| `bc.risk-frame.v2` 编解码与正式存储约束 | 自动测试覆盖 | 工程基线通过 |
| fixture A→B→C 窗口消费 | 8 个 integration 通过 | 仅受控 fixture |
| A 真实发布物全链路回放 | 未在本次验证中完成 | 待验收 |
| 规则风险科学标定 | 未完成 | `demo_unvalidated` |
| CNN P1 可选单步后端 | 10 个 model 测试通过 | 实验能力通过 |
| CNN P2 sidecar / P3 多时效 / P4 正式接入 | 未实现 | 不得宣称交付 |

## 复核命令

```bash
cd ${ARCTIC_ROUTE_ROOT}/work_package_b
make check
make model-check
```

若重新创建环境，先执行 `make env-create && make sync`；模型检查前执行 `make model-sync`。
任何环境、锁文件或上游版本变化都需要重新记录日期、解释器、命令和完整结果。

## 关联资料

- [正式入口](../README.md)
- [CNN 分阶段整合计划](DELIVERED_CNN_INTEGRATION_PLAN.md)
- [CNN 模型审计](DELIVERED_CNN_MODEL_AUDIT.md)
- [唯一总交接入口](../../work_package_b_handoff/work_package_b_handoff.md)
- [系统架构](../../ARCTIC_ROUTE_SYSTEM.md)
