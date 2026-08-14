# `demo_unvalidated` 风险基线与外部模型边界

0.2.0 用严格、版本化占位配置融合 11 个环境分量。它的用途是打通可复现的 A→B→C 工程链、
验证合同、时间门禁和缺测语义，不证明真实风险概率或船舶性能。

配置真源为 `configs/models/demo_unvalidated_v2.json`。每项显式给出
`component_id/weight/transform/lower/upper`：冰密集度 0.24、冰厚 0.14、冰型 0.05、冰缘
0.02、冰漂模长 0.06、浪高 0.13、流速模长 0.07、风速模长 0.10、低温差 0.05、低能见度
0.10、水位绝对值 0.04。权重和必须为 1；分量集合和顺序固定，禁止缺项、重复或按可用变量
静默重归一化。

- `risk_score` 在 `[0,1]`；任一必需环境分量未知时为 NaN。
- `risk_level` 使用 C 当前等宽规则；未知点保守编码为 5，但仍以 NaN/置信度 0 表达未知。
- `hard_mask` 当前仅为 `land_sea_mask < land_sea_mask_land_threshold` 或该分类本身未知；默认
  阈值为 0.5。风险分量缺测但陆海分类有效时，
  仍通过 `risk_score=NaN`、`confidence=0` 和 `risk_level=5` fail closed，不误称安全。
- `confidence` 取实际支撑来源质量下界，再乘配置中的 exact、categorical-nearest、linear 或
  static 时间方法因子。
- `environment_speed_factor = clip(1 - speed_risk_coefficient*risk,
  minimum_speed_factor, 1)`；默认仍为 `0.55/0.35`。未知点使用正的保守下界，
  但 C 会先按未知风险/零置信度拒绝把该点当作可规划安全点。

没有规则把 bathymetry、限制区名称或 `source_valid_mask` 自动转为 hard mask；没有输出最终航速。
默认配置已用逐数组回归测试证明与 0.1.0 数值基线一致；这只是工程兼容性，不是参数正确性
或校准证据。
这里的“逐数组一致”指在相同输入数组上，11 项风险、hard mask、confidence 和速度公式数值
一致。当前未冻结工作树暂把网格从 Git HEAD 的 1° 改为纬向 0.75°、经向 2.2° fast 候选，
因此完整帧的坐标、shape、grid ID 和 risk ID 会发生预期变化；该网格尚未通过完整性能和路线
保真门槛，不能称为正式科研分辨率。

2026-08-14 新交付的 CNN 权重不属于本规则配置：它只输入旧 B 已生成的单通道综合风险并输出
未知步长的下一步二维场，缺 hard mask、confidence、speed 和正式来源身份。资产虽真实存在，
但训练标签仍是旧启发式风险，只有训练 loss，没有独立验证。它当前固定为
`experimental_unverified`。P1 已完成固定哈希 weights-only CPU 接收、safetensors 转换和单步
短测；它仍缺独立验证和 cadence，详细证据见 [静态审计](DELIVERED_CNN_MODEL_AUDIT.md)，后续
只能走 [隔离后端方案](DELIVERED_CNN_INTEGRATION_PLAN.md)，不能覆盖本规则基线的身份或摘要。
