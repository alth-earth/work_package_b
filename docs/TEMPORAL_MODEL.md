# 逐小时时间与空间策略 v1

- 输出固定为 60 分钟闭区间，帧数为 `(end-start)/1h + 1`。
- 每个支撑帧都必须 `issue_time <= knowledge_as_of`。
- 连续量只在可见的前后支撑之间线性插值，不做无界外推。
- 分类层 `sea_ice_type`/`sea_ice_edge` 使用 nearest；相等距离优先较早帧。
- `land_sea_mask` 使用适用的静态版本并以 nearest 空间对齐。
- 目标网格为 EPSG:4326 严格递增 rectilinear 网格。v1 只接受覆盖整个目标 bbox 的
  rectilinear A 输入；不静默重投影 curvilinear/unstructured 数据。
- 连续量空间线性插值，分类/掩膜空间 nearest；越界或缺坐标明确失败。

该策略是确定性连续化，不是训练完成的预测模型。
