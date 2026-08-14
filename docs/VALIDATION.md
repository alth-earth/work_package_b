> [!NOTE]
> **文档治理声明**
> - 文件角色：正式工作包 B 的现行验证记录与验收边界。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 原文件去向：[VALIDATION.archive-20260814-pre-governance.md](VALIDATION.archive-20260814-pre-governance.md)。
> - 改造原因：以本次实测结果纠正旧计数，并分开正式路径和 CNN 可选后端证据。

# 正式工作包 B 验证记录

## 2026-08-14 可复核结果

在 `/root/my_project/work_package_b` 的现有有效 Mamba 前缀与锁定 uv 环境中执行：

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
cd /root/my_project/work_package_b
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
