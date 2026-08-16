# Single-factor realtime risk outputs

This document describes the Work Package B single-factor realtime output path.
It is intentionally separate from the formal B -> C multi-hour `RiskFrame`
contract.

## Purpose

Single-factor products answer a narrow operational question:

> What does each individual risk factor look like at the latest valid time that
> is currently visible to Work Package B?

They are designed for situation awareness, quality review, quick route-context
discussion, and handoff plots. They are not the authoritative route-planning
input consumed by Work Package C. The formal planning path remains
`RiskBuildService`, which emits `bc.risk-frame.v2`.

## Input

The entry point accepts the same `RiskBuildRequest` used by the formal risk
service:

```python
from arctic_route_risk import (
    RiskBuildRequest,
    build_realtime_single_factor_layers,
    write_realtime_single_factor_outputs,
)

layers = build_realtime_single_factor_layers(request)
paths = write_realtime_single_factor_outputs(request, "single_factor_outputs")
```

The request must contain a verified `BInputEnvelope` built from Work Package A
`PreparedWindow` data and the shared `RunContext`.

## Time handling

Single-factor layers do not create a multi-horizon forecast sequence. For each
factor, Work Package B selects that factor's own latest valid time at or before
the input envelope `knowledge_as_of` time.

The output records three separate timestamps:

- `issue_time`: latest upstream issue time supporting the layer.
- `valid_time`: latest valid time used by that factor.
- `collect_time`: time when B generated the single-factor output.

This keeps the display honest: a wind layer, wave layer, sea-ice layer, and
static-mask-supported layer may have different latest available times.

## Processing logic

For each configured risk component:

1. Identify the source data types required by that component.
2. Select the latest visible support time for those source types.
3. Regrid the source variables onto the target corridor grid.
4. Convert the physical variable into a normalized risk score using the same
   configured component transform used by the rule baseline.
5. Clip risk values into `[0, 1]`.
6. Apply the land/sea hard mask so land cells are excluded from the displayed
   factor layer.
7. Export a compact summary and, optionally, NetCDF/PNG products.

The generated dataset contains:

- `risk_score`: processed single-factor risk in `[0, 1]`.
- `risk_level`: integer display level from 0 to 5, where 0 is masked/no-data.
- `hard_mask`: land or unavailable cells.
- `confidence`: inherited source-confidence support for the factor layer.

## Output files

`write_realtime_single_factor_outputs()` always writes:

- `single_factor_realtime_summary.json`
- one JSON summary per factor

Optional outputs:

- NetCDF files when `write_netcdf=True`
- PNG preview images when `write_png=True`

The optional outputs require the extra dependency group:

```bash
uv sync --extra single-factor-output
```

## Boundary with formal planning

These outputs are intentionally marked with:

- `risk_role = realtime_situation_awareness`
- `planning_contract = non_authoritative_display_layer`

Work Package C should not treat them as the formal risk source. For route
planning, C should continue to use the comprehensive formal `RiskFrame` stream.
Single-factor layers are useful for explaining why a risk field looks the way it
does, checking latest source status, and producing factor-specific maps.
