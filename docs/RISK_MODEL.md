# `demo_unvalidated` 风险基线

首版用固定、版本化占位权重融合 11 个环境分量。它的用途是打通可复现的 A→B→C 工程链、
验证合同、时间门禁和缺测语义，不证明真实风险概率或船舶性能。

- `risk_score` 在 `[0,1]`；任一必需环境分量未知时为 NaN。
- `risk_level` 使用 C 当前等宽规则；未知点保守编码为 5，但仍以 NaN/置信度 0 表达未知。
- `hard_mask` v1 仅为 `land_sea_mask < 0.5` 或该分类本身未知；风险分量缺测但陆海分类有效时，
  仍通过 `risk_score=NaN`、`confidence=0` 和 `risk_level=5` fail closed，不误称安全。
- `confidence` 取实际支撑来源质量下界，再乘时间处理方法因子。
- `environment_speed_factor = clip(1 - 0.55*risk, 0.35, 1)`；未知点使用正的保守下界，
  但 C 会先按未知风险/零置信度拒绝把该点当作可规划安全点。

没有规则把 bathymetry、限制区名称或 `source_valid_mask` 自动转为 hard mask；没有输出最终航速。
