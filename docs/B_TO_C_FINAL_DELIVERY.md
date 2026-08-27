# Work Package B final delivery alignment

## Purpose

Work Package B consumes Work Package A `PreparedWindow`, `DatasetBundle.v2` and
`RunContext.v2`, then produces formal `bc.risk-frame.v2` risk fields for Work
Package C. The delivery goal is to keep B as the risk prediction and risk-field
generation layer: A owns environmental data preparation, C owns route planning,
and B owns risk scores, hard masks, confidence and speed impact.

## C-compatible RiskFrame output

Formal B output keeps the payload variables restricted to the fields accepted by
C:

- `risk_score`: continuous comprehensive risk in `[0, 1]`.
- `risk_level`: integer risk level derived from `risk_score`.
- `hard_mask`: non-navigable or hard-constrained grid cells.
- `confidence`: confidence of the risk value after source quality and temporal
  support are considered.
- `environment_speed_factor`: environmental speed factor used by C route-cost
  calculations.

Cell-level `hard_reason` is intentionally not included as a payload variable
because C's `bc.risk-frame.v2` schema does not allow additional payload
variables. B still keeps auditability through `payload.attrs["hard_reason_counts"]`.

## Mentor-aligned time sequence

The code labels every output frame with `payload.attrs["forecast_stage"]`.
The stage is computed from the valid-time offset relative to the run start:

- `0-2h`: high-confidence executable navigation window.
- `2-4h`: recommended executable navigation window.
- `4-6h`: predicted executable navigation window.
- `6-24h`: rolling dynamic route optimization window.
- `24-72h`: main corridor recognition and medium-range risk window.
- `>72h`: full-route reference window. This is based on the actual route horizon
  carried by `RunContext`, not a hard-coded nine-day assumption.

The helper functions are available from `arctic_route_risk.time_horizons`:

```python
from arctic_route_risk import mentor_required_offsets, stage_for_offset

stage = stage_for_offset(24)
offsets = mentor_required_offsets(total_route_hours=168)
```

## Final model parameter configuration

The final delivery parameter file is:

```text
configs/models/final_delivery_comprehensive_risk_v1.json
```

This configuration preserves the formal B interface while recording the
project-trained component weights and model identity:

- `model_version`: `final_delivery_comprehensive_risk.v1`
- `formula_version`: `trained_weighted_environment_components_v1`
- `calibration_status`: `project_trained_parameters_embedded`

The original demo baseline configuration remains available for regression tests
and historical comparison.

## Integration with Work Package A

B does not mutate A data. It reads the verified A envelope, checks the generation
identity, interpolates each required environmental variable onto the configured
B grid, and carries source references into `source_summary`. This preserves A's
data lineage and keeps B's risk output traceable.

## Integration with Work Package C

C can consume B output directly through the formal `RiskFrame` fields:

- Use `risk_score` for continuous route cost.
- Use `hard_mask` as the hard navigation barrier.
- Use `environment_speed_factor` for ETA and speed-loss calculations.
- Use `confidence` to express uncertainty in route recommendations.

No C-side parser should depend on `hard_reason` as a payload variable.

## Plotting outputs

B also provides report and review plotting helpers:

```python
from arctic_route_risk import render_risk_maps

paths = render_risk_maps(frame, "outputs/plots")
print(paths["color"])
print(paths["binary"])
```

These images are presentation products only. They do not replace RiskFrame data
and should not be used as the machine-readable handoff to C.

## Delivery status

This package is aligned with the A-B-C responsibility boundary:

- A: multi-source environmental data preparation.
- B: comprehensive risk prediction fields and risk visualization.
- C: route planning and dynamic replanning.

B's formal output has been constrained to the C schema, while retaining
engineering audit information in metadata.
