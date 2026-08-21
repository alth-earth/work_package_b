---
Overall Status: ACTIVE
Content Status:
  - COMPLETED
Document Role: SUPPORTING
Scope: bounded fixed-grid comparison through the real formal B build service
Canonical/Supporting: Supporting evidence for B grid research; production defaults remain canonical in versioned model configuration
Branch: research-validation-system
Last Verified: 2026-08-22
---

# B Formal Grid Comparison Report

## Verdict（2026-08-22 01:11 +08:00）

Three fixed-grid profiles were run sequentially through A's public
`PreparedWindow` and B's real `RiskBuildService`. All 234 outputs decoded as
`bc.risk-frame.v2`, carried `formal` provenance, and were written only as
unpublished experimental transport documents. No production configuration,
risk formula, level policy, hard-reason policy, store, or frozen artifact changed.

Status: `EXPERIMENTAL / REAL_DATA_PIPELINE_PASS`. This is engineering evidence,
not scientific calibration evidence.

## Experiment identity（2026-08-22 01:11 +08:00）

| Field | Value |
|---|---|
| Scenario | `tromso_isfjorden_august_2026_demo_v1` |
| Source bundle | `a-bundle-ed7e5595bd8fa7cc16ebe79d` |
| Bundle digest | `ed7e5595bd8fa7cc16ebe79d9ec013402f1e194549e11d10d27a3d531efb0d9c` |
| Provenance completeness | true for every coverage item |
| Window | 2026-08-15 10:00Z → 2026-08-18 15:00Z |
| Cadence / frames | hourly / 78 |
| B model | existing `demo_unvalidated_rule_baseline.v2` |
| Execution | one A prepare, then baseline → medium → fine sequentially |
| Publication | false; runtime evidence under `.runtime/experiments/b-formal-grid-round2/` |

The experiment uses the formal summer archive because the winter dataset is not
available. It does not claim a winter result.

## Formal build results（2026-08-22 01:11 +08:00）

| Profile | Angular steps lat × lon | Rows × cols | Cells/frame | B build | Sampled RSS peak | Risk JSON bytes |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.75° × 2.2° | 16 × 7 | 112 | 4.032 s | 920,720 KiB | 1,062,380 |
| medium | 0.375° × 1.25° | 31 × 11 | 341 | 3.861 s | 922,696 KiB | 2,202,408 |
| fine | 0.1875° × 0.625° | 60 × 21 | 1,260 | 3.914 s | 925,136 KiB | 6,744,836 |

The complete command, including A archive preparation and serialization, took
94.06 s wall time and 954,704 KiB maximum RSS with no swap. The measured B-only
section is almost flat at this bounded size; source preparation/xarray overhead
dominates. Output volume grows materially: medium is 2.07× and fine is 6.35× the
baseline JSON size.

RSS values are process-level sampled peaks and include the already-loaded A
window. They must not be interpreted as incremental memory attributable only to
the target grid.

## Risk and availability distributions（2026-08-22 01:11 +08:00）

| Profile | L1 | L2 | L3 | L4 | L5 | LAND | DATA_UNAVAILABLE | Finite score min / mean / max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 75.893% | 0% | 0% | 0% | 24.107% | 20.536% | 3.571% | 0.0150 / 0.0483 / 0.1637 |
| medium | 74.780% | 0% | 0% | 0% | 25.220% | 19.062% | 6.158% | 0.0147 / 0.0488 / 0.1626 |
| fine | 74.921% | 0% | 0% | 0% | 25.079% | 17.778% | 7.302% | 0.0141 / 0.0490 / 0.1630 |

All navigable finite cells are Level 1 for this summer artifact/model. Level 5
counts are the separate fail-closed hard cells. Grid refinement does not create
Level 2–4 signal and must not be presented as scientific-risk improvement.
Refinement mainly exposes more coastline and unavailable boundaries at their
sampled cell centers.

## Interpretation and limits（2026-08-22 01:11 +08:00）

- B can generate all three regular-grid profiles within roughly one second of
  each other after the A window is prepared; B generation is not the immediate
  bottleneck at 78 frames.
- Transport size scales enough to matter for storage and downstream ingestion.
- Grid resolution changes endpoint mapping and C state-space size, so quality
  cannot be assessed from B latency alone.
- The profile labels are experimental identities. `baseline` means the B code
  default 16×7 control; `medium` is the current 31×11 Tromsø presentation grid.
- No interpolation, adaptive grid, risk formula change, or scientific
  calibration was performed.

## Reproduction and evidence（2026-08-22 01:11 +08:00）

```bash
cd /root/my_project/work_package_b
uv run python scripts/benchmark_formal_grid_profiles.py \
  --a-data-root /root/my_project/work_package_a/data \
  --run-context /root/my_project/work_package_a/data/output/rc2-smoke/output-tromso-144h-r2/run-context.json \
  --output-root /root/my_project/.runtime/experiments/b-formal-grid-round2
```

Canonical runtime outputs remain untracked under `.runtime`; the report records
the stable summary. The harness writes each profile summary before moving to the
next profile, so an interrupted run retains bounded evidence without publishing
a partial BC window.
