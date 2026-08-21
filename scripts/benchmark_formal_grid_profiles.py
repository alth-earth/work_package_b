#!/usr/bin/env python3
"""Run fixed-grid experiments through A's formal archive and B's real builder."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from arctic_route_contracts import canonical_sha256, load_run_context
from arctic_route_data import PartitionedABCache, SimulationClock, WorkPackageA
from arctic_route_data.sources import LocalArchiveSource

from arctic_route_risk.config import load_risk_build_configuration
from arctic_route_risk.context import REQUIRED_FORMAL_DATA_TYPES, BInputEnvelope
from arctic_route_risk.formal_grid_experiments import (
    build_formal_grid_profile,
    write_frame_documents,
)
from arctic_route_risk.grid_experiments import load_grid_experiment_suite


def _parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-data-root", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/models/demo_unvalidated_tromso_smoke_grid_v1.json"),
    )
    parser.add_argument(
        "--profiles-config",
        type=Path,
        default=Path("configs/experiments/tromso_grid_profiles_v1.json"),
    )
    parser.add_argument("--start", default="2026-08-15T10:00:00Z")
    parser.add_argument("--horizon-hours", type=int, default=77)
    parser.add_argument("--cache-memory-mb", type=float, default=2048.0)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon_hours <= 0:
        parser.error("--horizon-hours must be positive")

    start = _parse_utc(args.start)
    end = start + timedelta(hours=args.horizon_hours)
    suite = load_grid_experiment_suite(args.profiles_config)
    risk_configuration = load_risk_build_configuration(args.model_config)
    base_context = load_run_context(args.run_context)

    source = LocalArchiveSource(args.a_data_root)
    clock = SimulationClock(start)
    work = WorkPackageA(
        source=source,
        clock=clock,
        cache=PartitionedABCache(max_memory_mb=args.cache_memory_mb),
    )
    try:
        prepared = work.prepare_window_for_b(
            route_id=base_context.corridor_id,
            data_types=tuple(sorted(REQUIRED_FORMAL_DATA_TYPES)),
            start_time=start,
            target_horizon_hours=args.horizon_hours,
            minimum_complete_horizon_hours=args.horizon_hours,
            knowledge_as_of=start,
        )
        context = replace(
            base_context,
            run_id="run-00000000-0000-4000-8000-0000000000b2",
            created_at=start,
            simulation_start=start,
            simulation_end=end,
            dataset_bundle_id=prepared.dataset_bundle.bundle_id,
            dataset_bundle_digest=prepared.dataset_bundle.bundle_digest,
            config_digest=canonical_sha256(
                {
                    "status": "EXPERIMENTAL",
                    "purpose": "formal_grid_comparison",
                    "base_config_digest": base_context.config_digest,
                    "dataset_bundle_id": prepared.dataset_bundle.bundle_id,
                    "dataset_bundle_digest": prepared.dataset_bundle.bundle_digest,
                    "start": start,
                    "end": end,
                }
            ),
        )
        envelope = BInputEnvelope.from_prepared_window(
            run_context=context,
            prepared_window=prepared,
            generation_id=prepared.generation_id,
            knowledge_as_of=prepared.as_of_time,
        )
        summaries = []
        for profile in suite.profiles:
            result = build_formal_grid_profile(
                envelope=envelope,
                target_bbox=suite.bbox,
                model_config=risk_configuration.model_config,
                profile=profile,
                utc_now=lambda: start,
            )
            summaries.append(result.summary)
            profile_root = args.output_root / profile.name
            profile_root.mkdir(parents=True, exist_ok=True)
            (profile_root / "summary.json").write_text(
                json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            write_frame_documents(
                result.frames,
                profile_root / "risk-frames.json",
            )
            del result
            gc.collect()
    finally:
        work.close()

    report = {
        "schema_version": "b.formal-grid-comparison.v1",
        "status": "EXPERIMENTAL",
        "formal_risk_build": True,
        "published": False,
        "source": {
            "kind": "local_formal_a_archive",
            "dataset_bundle_id": prepared.dataset_bundle.bundle_id,
            "dataset_bundle_digest": prepared.dataset_bundle.bundle_digest,
            "provenance_complete": all(
                item.provenance_complete for item in prepared.dataset_bundle.coverage
            ),
        },
        "window": {
            "start": start.isoformat().replace("+00:00", "Z"),
            "end": end.isoformat().replace("+00:00", "Z"),
            "hours": args.horizon_hours,
        },
        "profiles": summaries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "comparison.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
