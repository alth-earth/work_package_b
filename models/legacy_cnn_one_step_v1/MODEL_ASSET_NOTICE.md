# Legacy CNN asset notice

This directory contains a converted, pickle-free copy of the user-delivered
legacy one-step CNN checkpoint. It is an experimental, CPU-only shadow asset;
it is not the formal B rule backend and its cadence is intentionally unknown.

- Converted artifact: `model.safetensors`
- Safetensors SHA-256: `602d4849ce0b2f5eda90db2a020b8ec69a00dc929d5e764ed412d0d03e205103`
- Source ZIP SHA-256: `a2ee74fd70cb0735695d9cf25ae8907a7c6d7aab866b7549d8142e412190ad79`
- Source checkpoint SHA-256: `0390fac17de1f082652fc5851b6979fd771d98ff803880c6562c54921e081666`
- Source member: `22_深度学习综合风险预测模型/downloads/model/comprehensive_risk_cnn.pth`
- Conversion: `torch.load(weights_only=True, map_location="cpu")` followed by
  safetensors serialization and bitwise re-load verification.
- Upstream license: not provided with the delivery.
- Public redistribution: explicitly authorized by the user on 2026-08-14 for
  this public Work Package B repository.

The original ZIP, checkpoint, training data, scripts and bytecode are not
included in the repository. Do not infer an hourly forecast, calibration,
confidence, hard mask, route, or navigation guarantee from this asset.
