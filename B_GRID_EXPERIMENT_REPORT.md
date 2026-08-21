---
Overall Status: ACTIVE
Content Status:
  - COMPLETED
  - PLANNED
Document Role: SUPPORTING
Scope: fixed-grid experiment framework and initial kernel benchmark
Canonical Current State: NO
Branch: research-validation-system
Last Verified: 2026-08-22
---

# B Grid Experiment Report

## Status and scope（2026-08-22 00:08）

Status: `EXPERIMENTAL + UNIT_VALIDATED`.

The framework compares fixed regular grids without changing production
`TargetGridConfig`, risk formula, risk level, hard reason or `bc.risk-frame.v2`.
The benchmark uses deterministic synthetic arrays and the same xarray
linear/nearest interpolation methods as B's spatial kernel. It is not a formal
A→B build, scientific validation or end-to-end replay benchmark.

## Profiles（2026-08-22 00:08）

Config: `configs/experiments/tromso_grid_profiles_v1.json` over the current
Tromsø bbox 10–22°E, 68.5–79.5°N.

| Profile | Max angular step | Realized grid | Cells | Relation |
|---|---|---:|---:|---|
| baseline | 0.75° × 2.2° | 16×7 | 112 | unchanged code default |
| medium | 0.375° × 1.25° | 31×11 | 341 | current Tromsø demo policy |
| fine | 0.1875° × 0.625° | 60×21 | 1260 | experimental half-step candidate |

Production model configs are unchanged. Profiles are loaded only by
`arctic_route_risk.grid_experiments` and the benchmark script.

## Initial kernel results（2026-08-22 00:08）

Command:

```bash
uv run python scripts/benchmark_grid_profiles.py \
  --output /root/my_project/.runtime/test-logs/b-grid-kernel-20260822.json
```

Eight measured iterations per profile after one warm-up:

| Grid | Cells | Mean kernel runtime | Kernel output | Python peak | Process peak RSS | Impact |
|---|---:|---:|---:|---:|---:|---|
| baseline | 112 | 24.610 ms | 9,856 B | 932,650 B | 131,880 KiB | reference |
| medium | 341 | 24.622 ms | 30,008 B | 782,868 B | 132,136 KiB | 3.04× cells/output |
| fine | 1260 | 24.534 ms | 110,880 B | 830,386 B | 132,136 KiB | 11.25× cells/output |

Runtime is dominated by fixed xarray overhead at these small dimensions, so
the similar timings do not prove that full B runtime is grid-independent.
Output bytes scale exactly with cells. Process RSS is an absolute process peak
and cumulative within one run; it is not isolated per-profile memory.

Each profile produced a deterministic output digest across repeated benchmark
runs. The JSON evidence is runtime-only and is not committed.

## Validation（2026-08-22 00:08）

```text
ruff: PASS
targeted pytest: 2 passed
default grid unchanged: PASS
profile shapes: PASS
repeat output digest equality: PASS
```

## Next experiment gate（2026-08-22 00:08）

Before proposing Adaptive Grid or selecting a finer formal grid:

1. run a formal, bounded B build from one exact A `PreparedWindow` for each
   profile in isolated processes;
2. record wall time, process peak RSS, risk/hard/unknown distributions, grid and
   model digests;
3. run C route/ETA/integrity comparison against each committed experimental window;
4. reject any profile that changes unknown/hard semantics or exceeds the agreed memory budget;
5. keep all outputs under a new experimental identity and outside frozen stores.

Formal build benchmark: `PLANNED`, not run in this round.

