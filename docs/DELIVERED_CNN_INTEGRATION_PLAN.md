> [!NOTE]
> **文档治理声明**
> - 文件角色：旧 CNN 在当前工作包 B 中的现行分阶段整合计划与审批边界。
> - 改造时间：2026-08-14（Asia/Shanghai）。
> - 原文件去向：[DELIVERED_CNN_INTEGRATION_PLAN.archive-20260814-pre-governance.md](DELIVERED_CNN_INTEGRATION_PLAN.archive-20260814-pre-governance.md)。
> - 改造原因：消除“尚未实现”与“P1 已实现”的时点冲突，并锁定正式主线边界。

# 旧 CNN 分阶段整合计划

## 当前结论

`22_深度学习综合风险预测模型.zip` 已整合进当前 B 仓库的可选实验后端并完成 P1，但未完成
P2 sidecar，且未进入 formal build / RiskFrame / store / C。

当前 CNN 是一个旧版、单通道、单步实验模型。原生栅格约为 `0.05°`，其训练数据、训练代码、
预报步长语义和独立科学验证尚不完整。它不能通过重复单步输出伪造 169 小时预报，也不能覆盖
正式规则风险的硬掩码、置信度和来源审计职责。

## 分阶段状态

| 阶段 | 当前状态 | 产物与约束 | 进入下一阶段的门槛 |
|---|---|---|---|
| P0 资产审计与安全转换 | 已完成 | checkpoint 已审计并转为 safetensors；保留来源与哈希 | 资产清单可复核 |
| P1 可选后端 | 已完成 | 显式 opt-in；独立模型依赖；单步输出标为非正式 | `make model-check` 10 个通过 |
| P2 fail-open sidecar | 未实现，需批准 | 只能写独立实验记录；失败不得影响正式 build | 定义输入快照、指标、存储隔离和关闭开关 |
| P3 多时效/独立预报能力 | 未开始 | 禁止递归复制单步结果；必须重新定义时效语义 | 数据、训练、回测和漂移证据齐备 |
| P4 正式候选 | 未开始 | 才可讨论进入 RiskFrame/store/C | 契约、科学、安全和治理专项评审通过 |

## 已实现的 P1 边界

- 模型资产位于 `models/legacy_cnn_one_step_v1/`，策略位于
  `configs/models/legacy_cnn_one_step_v1_policy.json`。
- 适配器位于 `src/arctic_route_risk/modeling/legacy_cnn.py`，只提供显式调用的实验后端。
- 正式服务仍由 `src/arctic_route_risk/service.py` 构建 `bc.risk-frame.v2`；
  `src/arctic_route_risk/modeling/contracts.py` 明确隔离正式服务和 modeling 包。
- CNN 输出缺少可证明的 `valid_time` 与时间步语义，并携带 `formal_risk_frame=false`；
  不允许写入正式 store。

## P2 设计约束（尚未实施）

P2 若获批准，应作为与正式构建并行的 sidecar：读取同一份冻结输入快照，产生独立命名空间的
实验结果和对比指标；超时、缺依赖、模型错误或输出异常时 fail-open，正式 B 继续运行。P2 不得：

- 改变正式 `RiskFrame` 数值或 schema；
- 写入正式 `PersistentRiskStore`；
- 作为 C 的输入；
- 将未知预报间隔解释成 1 小时；
- 把实验来源标为 `formal`。

启动 P2 前至少需要用户确认运行预算、采样窗口、保留周期、比较指标与停用阈值。P2 结果只用于
研究判断，不自动授权 P3 或 P4。

## 资产与合规

用户已授权在本项目范围内处理和再分发交付资产；但原始模型缺少可独立核验的上游许可证、
训练数据清单和训练过程。两者不是同一事实：项目内授权允许当前工程处理，不等于已经证明所有
上游权利或适合对外发布。对外分发或正式部署前仍需人工确认。

## 验证与相关文档

- [当前验证记录](VALIDATION.md)
- [模型审计](DELIVERED_CNN_MODEL_AUDIT.md)
- [CNN 专题交接（保留原文）](../../work_package_b_handoff/工作包B-新风险模型整合与续开发Handoff.md)
- [唯一总交接入口](../../work_package_b_handoff/work_package_b_handoff.md)
- [正式工作包 B 入口](../README.md)
