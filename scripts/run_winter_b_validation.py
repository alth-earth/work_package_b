#!/usr/bin/env python3
"""Run one formal Winter B RiskFrame validation window.

This runner is intentionally a thin experiment boundary.  It restores the
immutable A bundle through A's public exact-bundle resolver, builds the existing
``bc.risk-frame.v2`` frames with the existing rule baseline, and commits the
window to a new ``PersistentRiskStore``.  It does not alter B model semantics,
contract versions, or production defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from arctic_route_contracts import load_run_context, verify_dataset_bundle
from arctic_route_data import DatasetBundle, PartitionedABCache, SimulationClock, WorkPackageA
from arctic_route_data.sources import LocalArchiveSource
from jsonschema import Draft202012Validator, FormatChecker

from arctic_route_risk import (
    BInputEnvelope,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
    RiskExplanationArtifactStore,
    RiskExplanationResearchExporter,
    load_risk_build_configuration,
)
from arctic_route_risk.bc_codec import risk_frame_to_document
from arctic_route_risk.formal_grid_experiments import PeakRssSampler
from arctic_route_risk.grid_experiments import load_grid_experiment_suite


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"UTC timestamp requires timezone: {value}")
    return parsed.astimezone(UTC)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _validate_execution_spec(document: dict[str, Any], *, context: Any) -> dict[str, Any]:
    expected = {
        "schema_version",
        "run_id",
        "scenario_id",
        "generation_id",
        "input_revision",
        "generated_at",
        "planning_contract",
        "max_snap_km",
        "replan_after_hours",
        "per_stage_timeout_seconds",
    }
    if set(document) != expected:
        raise ValueError("ExecutionSpec fields do not match strict v1")
    if document["schema_version"] != "orchestrator.execution-spec.v1":
        raise ValueError("unsupported ExecutionSpec schema")
    if document["run_id"] != context.run_id:
        raise ValueError("ExecutionSpec run_id does not match RunContext")
    if document["scenario_id"] != context.scenario_id:
        raise ValueError("ExecutionSpec scenario_id does not match RunContext")
    if document["generation_id"] != 0:
        raise ValueError("Winter first validation requires generation_id=0")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_frames(frames: Sequence[Any], schema_path: Path) -> None:
    schema = _load_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for frame in frames:
        document = risk_frame_to_document(frame)
        validator.validate(document)
        if frame.schema_version != "bc.risk-frame.v2":
            raise ValueError("RiskFrame schema version mismatch")
        if frame.provenance.value != "formal":
            raise ValueError("Winter B output must have formal provenance")


def _distribution(frames: Sequence[Any]) -> dict[str, Any]:
    if not frames:
        raise ValueError("RiskFrame window must not be empty")
    first = frames[0]
    payload = first.payload
    latitude = np.asarray(payload.coords["latitude"].values, dtype=np.float64)
    longitude = np.asarray(payload.coords["longitude"].values, dtype=np.float64)
    rows = int(latitude.size)
    cols = int(longitude.size)
    total_cells = 0
    level_counts: Counter[int] = Counter()
    reason_counts: Counter[str] = Counter()
    finite_scores: list[np.ndarray] = []
    unknown_count = 0
    unknown_navigable_count = 0
    hard_mask_count = 0
    hard_reason_mismatch_count = 0
    per_frame: list[dict[str, Any]] = []

    for frame in frames:
        current = frame.payload
        scores = np.asarray(current["risk_score"].values, dtype=np.float64)
        levels = np.asarray(current["risk_level"].values, dtype=np.uint8)
        hard_mask = np.asarray(current["hard_mask"].values, dtype=bool)
        reasons = (
            np.asarray(current["hard_reason"].values, dtype=np.str_)
            if "hard_reason" in current
            else None
        )
        finite = np.isfinite(scores)
        unknown = ~finite
        hard = hard_mask
        level_counts.update(int(value) for value in levels.ravel())
        frame_reason_counts = (
            Counter(str(value) for value in reasons.ravel())
            if reasons is not None
            else Counter(
                {
                    str(key): int(value)
                    for key, value in current.attrs.get("hard_reason_counts", {}).items()
                }
            )
        )
        reason_counts.update(frame_reason_counts)
        finite_scores.append(scores[finite])
        total_cells += int(scores.size)
        unknown_count += int(np.count_nonzero(unknown))
        unknown_navigable_count += int(np.count_nonzero(unknown & ~hard))
        hard_mask_count += int(np.count_nonzero(hard))
        if reasons is not None:
            hard_reason_mismatch_count += int(
                np.count_nonzero((reasons == "NONE") != ~hard)
            )
        frame_finite = scores[finite]
        per_frame.append(
            {
                "valid_time": frame.valid_time.isoformat().replace("+00:00", "Z"),
                "risk_levels": {
                    str(level): int(np.count_nonzero(levels == level))
                    for level in range(1, 6)
                },
                "hard_reasons": dict(sorted(frame_reason_counts.items())),
                "unknown_count": int(np.count_nonzero(unknown)),
                "unknown_navigable_count": int(np.count_nonzero(unknown & ~hard)),
                "finite_risk_score_min": float(np.min(frame_finite)) if frame_finite.size else None,
                "finite_risk_score_mean": (
                    float(np.mean(frame_finite)) if frame_finite.size else None
                ),
                "finite_risk_score_max": float(np.max(frame_finite)) if frame_finite.size else None,
            }
        )

    finite_parts = [values for values in finite_scores if values.size]
    combined = np.concatenate(finite_parts) if finite_parts else np.array([], dtype=float)
    level_distribution = {
        str(level): {
            "count": level_counts.get(level, 0),
            "percentage": round(100.0 * level_counts.get(level, 0) / total_cells, 6),
        }
        for level in range(1, 6)
    }
    return {
        "frame_count": len(frames),
        "valid_time_start": frames[0].valid_time.isoformat().replace("+00:00", "Z"),
        "valid_time_end": frames[-1].valid_time.isoformat().replace("+00:00", "Z"),
        "cadence_minutes": int(
            (frames[1].valid_time - frames[0].valid_time).total_seconds() / 60
        )
        if len(frames) > 1
        else None,
        "grid": {
            "rows": rows,
            "cols": cols,
            "cells_per_frame": rows * cols,
            "latitude_min": float(latitude[0]),
            "latitude_max": float(latitude[-1]),
            "longitude_min": float(longitude[0]),
            "longitude_max": float(longitude[-1]),
            "latitude_step_actual": float(np.diff(latitude).mean()) if rows > 1 else None,
            "longitude_step_actual": float(np.diff(longitude).mean()) if cols > 1 else None,
            "grid_id": str(payload.attrs.get("grid_id", "")),
        },
        "total_cells": total_cells,
        "navigable_cells": total_cells - hard_mask_count,
        "hard_mask_count": hard_mask_count,
        "unknown_count": unknown_count,
        "unknown_navigable_nodes": unknown_navigable_count,
        "hard_reason_distribution": dict(sorted(reason_counts.items())),
        "hard_reason_consistency_mismatches": hard_reason_mismatch_count,
        "risk_level_distribution": level_distribution,
        "finite_risk_score": {
            "count": int(combined.size),
            "min": float(np.min(combined)) if combined.size else None,
            "mean": float(np.mean(combined)) if combined.size else None,
            "max": float(np.max(combined)) if combined.size else None,
        },
        "per_frame": per_frame,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-data-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run-context", type=Path, required=True)
    parser.add_argument("--execution-spec", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--profiles-config", type=Path, required=True)
    parser.add_argument("--profile", choices=("baseline", "medium", "fine"), default="medium")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--emit-risk-explanation",
        action="store_true",
        help=(
            "capture the same-call B formula trace and publish an immutable "
            "risk-explanation.v1 sidecar + manifest"
        ),
    )
    parser.add_argument(
        "--generated-at",
        type=_parse_utc,
        help=(
            "override the RiskFrame generated_at timestamp for deterministic "
            "replay binding; defaults to the current UTC second"
        ),
    )
    parser.add_argument("--cache-memory-mb", type=float, default=2048.0)
    parser.add_argument(
        "--risk-schema",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "work_package_c"
        / "schemas"
        / "risk-frame-v2.schema.json",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    context = load_run_context(args.run_context)
    execution_spec = _validate_execution_spec(_load_json(args.execution_spec), context=context)
    bundle_document = _load_json(args.bundle)
    bundle = DatasetBundle.from_dict(bundle_document)
    verified_bundle = verify_dataset_bundle(bundle.to_dict())
    if verified_bundle.bundle_id != context.dataset_bundle_id:
        raise ValueError("DatasetBundle ID does not match RunContext")
    if verified_bundle.bundle_digest != context.dataset_bundle_digest:
        raise ValueError("DatasetBundle digest does not match RunContext")
    if verified_bundle.formal_run_eligible is not True:
        raise ValueError("DatasetBundle is not formal-run eligible")

    suite = load_grid_experiment_suite(args.profiles_config)
    profile = next((item for item in suite.profiles if item.name == args.profile), None)
    if profile is None:
        raise ValueError(f"grid profile not found: {args.profile}")
    configuration = load_risk_build_configuration(args.model_config)
    grid = profile.target_grid()
    if configuration.grid_config != grid:
        raise ValueError(
            "model config grid does not match selected experiment profile; "
            "refusing implicit grid change"
        )

    source = LocalArchiveSource(args.a_data_root)
    clock = SimulationClock(context.simulation_start)
    work = WorkPackageA(
        source=source,
        clock=clock,
        cache=PartitionedABCache(max_memory_mb=args.cache_memory_mb),
    )
    resolve_started = time.perf_counter()
    try:
        prepared = work.resolve_dataset_bundle_for_b(
            bundle,
            generation_id=int(execution_spec["generation_id"]),
            knowledge_as_of=bundle.as_of_time,
        )
        resolve_seconds = time.perf_counter() - resolve_started
        envelope = BInputEnvelope.from_prepared_window(
            run_context=context,
            prepared_window=prepared,
            generation_id=int(execution_spec["generation_id"]),
            knowledge_as_of=bundle.as_of_time,
        )
        generated_at = args.generated_at or datetime.now(UTC).replace(microsecond=0)
        build_started = time.perf_counter()
        build_request = RiskBuildRequest(
            envelope=envelope,
            target_bbox=suite.bbox,
            grid_config=grid,
            model_config=configuration.model_config,
        )
        with PeakRssSampler() as memory:
            if args.emit_risk_explanation:
                build_result = RiskBuildService(
                    utc_now=lambda: generated_at
                ).build_window_with_explanation_trace(build_request)
                frames = build_result.frames
            else:
                build_result = None
                frames = RiskBuildService(utc_now=lambda: generated_at).build_window(build_request)
        build_seconds = time.perf_counter() - build_started
        build_peak_rss_kib = memory.peak_kib
    finally:
        work.close()

    if len(frames) != 145:
        raise ValueError(f"Winter 144-hour window must contain 145 frames, got {len(frames)}")
    if (
        frames[0].valid_time != context.simulation_start
        or frames[-1].valid_time != context.simulation_end
    ):
        raise ValueError("RiskFrame valid-time window does not match RunContext")
    _validate_frames(frames, args.risk_schema)
    distribution = _distribution(frames)

    store_root = args.output_root / "risk-store"
    store = PersistentRiskStore(store_root)
    store.activate_generation(context.run_id, int(execution_spec["generation_id"]))
    publish_started = time.perf_counter()
    committed = store.publish_window(
        frames,
        start=context.simulation_start,
        end=context.simulation_end,
        interval_minutes=60,
    )
    publish_seconds = time.perf_counter() - publish_started
    verified_commit = store.get_committed_window(committed.query)
    if verified_commit.content_digest != committed.content_digest:
        raise ValueError("committed RiskFrame window digest changed during verification")

    frame_index = {
        "artifact_kind": "winter-b-risk-frame-index",
        "status": "FORMAL_VALIDATED",
        "frame_schema": "bc.risk-frame.v2",
        "run_id": context.run_id,
        "scenario_id": context.scenario_id,
        "dataset_bundle_id": bundle.bundle_id,
        "dataset_bundle_digest": bundle.bundle_digest,
        "model_config_digest": frames[0].model_config_digest,
        "grid_profile": args.profile,
        "risk_store": "risk-store",
        "commit_id": committed.commit_id,
        "content_digest": committed.content_digest,
        "frame_ids": [frame.risk_id for frame in committed.frames],
    }
    _write_json(args.output_root / "frame-index.json", frame_index)
    _write_json(args.output_root / "distribution.json", distribution)

    explanation_publication = None
    if build_result is not None:
        sidecar = RiskExplanationResearchExporter(utc_now=lambda: generated_at).export(
            committed_window=verified_commit,
            build_result=build_result,
        )
        explanation_store = RiskExplanationArtifactStore(args.output_root / "risk-explanation")
        publication = explanation_store.publish(sidecar)
        verified_manifest, verified_sidecar = explanation_store.read(
            publication["manifest_path"]
        )
        if verified_sidecar != sidecar:
            raise ValueError("risk explanation sidecar changed during publication readback")
        if verified_manifest["artifact_sha256"] != publication["manifest"]["artifact_sha256"]:
            raise ValueError("risk explanation manifest digest changed during readback")
        explanation_publication = {
            "schema_version": "risk-explanation-transport.v1",
            "status": "PUBLISHED",
            "manifest_path": str(
                publication["manifest_path"].relative_to(args.output_root)
            ),
            "artifact_path": str(
                publication["artifact_path"].relative_to(args.output_root)
            ),
            "artifact_id": publication["manifest"]["artifact_id"],
            "artifact_sha256": publication["manifest"]["artifact_sha256"],
            "risk_window_id": committed.commit_id,
        }
        _write_json(args.output_root / "risk-explanation-publication.json", explanation_publication)

    metadata = {
        "artifact_kind": "winter-b-risk-validation",
        "status": "FORMAL_VALIDATED",
        "experiment_identity": {
            "bundle_id": bundle.bundle_id,
            "bundle_digest": bundle.bundle_digest,
            "run_id": context.run_id,
            "execution_spec": str(args.execution_spec),
            "generation_id": int(execution_spec["generation_id"]),
        },
        "input": {
            "bundle_path": str(args.bundle),
            "bundle_sha256": _sha256(args.bundle),
            "run_context_path": str(args.run_context),
            "execution_spec_path": str(args.execution_spec),
            "scenario_start": context.simulation_start.isoformat().replace("+00:00", "Z"),
            "scenario_end": context.simulation_end.isoformat().replace("+00:00", "Z"),
            "knowledge_as_of": bundle.as_of_time.isoformat().replace("+00:00", "Z"),
            "records": len(bundle.records),
            "required_data_types": sorted(bundle.requested_data_types),
        },
        "b_configuration": {
            "model_config_path": str(args.model_config),
            "model_version": configuration.model_config.model_version,
            "model_calibration_status": configuration.model_config.calibration_status,
            "risk_formula": configuration.model_config.formula_version,
            "risk_level_policy": configuration.model_config.risk_level_policy,
            "hard_mask_policy": configuration.model_config.hard_mask_policy,
            "unknown_policy": configuration.model_config.unknown_policy,
            "grid_profile": args.profile,
            "grid": distribution["grid"],
            "model_config_digest": frames[0].model_config_digest,
        },
        "output": {
            "risk_frame_schema": "bc.risk-frame.v2",
            "risk_store": str(store_root),
            "commit_id": committed.commit_id,
            "content_digest": committed.content_digest,
            "frame_count": committed.count,
            "frame_schema_validation": "PASS",
            "commit_readback_validation": "PASS",
            "provenance": sorted({frame.provenance.value for frame in frames}),
            "risk_explanation": explanation_publication,
        },
        "performance_observation": {
            "resolve_seconds": round(resolve_seconds, 6),
            "build_seconds": round(build_seconds, 6),
            "publish_seconds": round(publish_seconds, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "build_peak_sampled_rss_kib": build_peak_rss_kib,
            "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    }
    _write_json(args.output_root / "run-metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
