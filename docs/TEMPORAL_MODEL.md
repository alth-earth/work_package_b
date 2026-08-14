# 逐小时时间与空间策略 v2

- 输出固定为 60 分钟闭区间，帧数为 `(end-start)/1h + 1`。
- 每个支撑帧都必须 `issue_time <= knowledge_as_of`。
- 连续量只在可见的前后支撑之间线性插值，不做无界外推。
- 分类层 `sea_ice_type`/`sea_ice_edge` 使用 nearest；相等距离优先较早帧。
- `land_sea_mask` 使用适用的静态版本并以 nearest 空间对齐。
- 时间方法置信度由模型配置 v2 显式给出：默认 exact/static 为 1.0、linear 为 0.9、类别
  nearest 为 0.85；A 质量默认 good/suspect/degraded 为 1.0/0.75/0.5。最终 confidence 取所有
  必需来源支撑值的下界。
- 目标网格为 EPSG:4326 严格递增 rectilinear 网格。v1 只接受覆盖整个目标 bbox 的
  rectilinear A 输入；不静默重投影 curvilinear/unstructured 数据。目标网格政策自身仍为 v1，
  当前未冻结配置暂用纬向 0.75°、经向 2.2° 的 fast smoke 步长。它保证两条走廊空 hard-mask
  端点允许区有节点，但尚无完整重规划、真实 mask 或路线保真证据；formal 候选与模型 native
  grid 必须独立管理。不扩张允许区、不做走廊专调；空间重采样仍沿用同一 linear/nearest 语义。
- 连续量空间线性插值，分类/掩膜空间 nearest；越界或缺坐标明确失败。

该策略是确定性连续化，不是训练完成的预测模型。

新交付 CNN 只证明未知时长的单步预测；旧数据谱系可能是 6 h，但 checkpoint 未绑定 cadence。
在 cadence、训练源和验证补齐前，它只能做单步 shadow，禁止复制一张结果或递归滚动 169 步
冒充逐小时长窗。完整时间晋级门槛见 [DELIVERED_CNN_INTEGRATION_PLAN.md](DELIVERED_CNN_INTEGRATION_PLAN.md)。
