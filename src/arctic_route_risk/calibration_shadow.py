"""Research-only comparison of risk calibration policies.

The module consumes immutable ``bc.risk-frame.v2`` JSON artifacts and emits an
independent comparison sidecar.  It never mutates RiskFrames and never publishes
candidate levels into the formal B store.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from statistics import fmean, median
from typing import Any

_SIDECAR_VERSION = "research.risk-calibration-shadow-comparison.v1"
_CONFIG_VERSION = "b.risk-calibration-shadow-experiment.v1"
_RISK_FRAME_VERSION = "bc.risk-frame.v2"
_RISK_WINDOW_VERSION = "bc.risk-window-commit.v1"
_PRODUCER_VERSION = "calibration-shadow-builder.v1"
_RISK_ID = re.compile(r"^risk-sha256-([0-9a-f]{64})$")
_WINDOW_ID = re.compile(r"^risk-window-sha256-([0-9a-f]{64})$")
_HARD_REASONS = frozenset({"NONE", "LAND", "DATA_UNAVAILABLE", "OTHER"})
_FRAME_FIELDS = {
    "schema_version",
    "risk_id",
    "run_id",
    "scenario_id",
    "corridor_id",
    "vessel_profile_id",
    "config_digest",
    "model_config_digest",
    "generation_id",
    "valid_time",
    "as_of_time",
    "generated_at",
    "model_version",
    "payload",
    "source_summary",
    "provenance",
}
_MANIFEST_FIELDS = {
    "schema_version",
    "commit_id",
    "content_digest",
    "start",
    "end",
    "interval_seconds",
    "count",
    "run_id",
    "scenario_id",
    "corridor_id",
    "generation_id",
    "vessel_profile_id",
    "config_digest",
    "model_config_digest",
    "as_of",
    "frames",
}


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One immutable RiskFrame cell used by the shadow experiment."""

    frame_index: int
    frame_id: str
    valid_time: str
    row: int
    column: int
    risk_score: float | None
    baseline_level: int
    hard_reason: str
    split: str


def build_shadow_comparison(
    *,
    frame_index_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Build a deterministic, read-only calibration comparison document."""

    config = _load_json(config_path)
    _validate_config(config)
    config_sha256 = _sha256_file(config_path)
    producer_source_sha256 = _sha256_file(Path(__file__))
    index = _load_json(frame_index_path)
    index_sha256 = _sha256_file(frame_index_path)
    _validate_index(index, config=config, index_sha256=index_sha256)
    store_root = _safe_store_root(frame_index_path, index["risk_store"])
    manifest_path = store_root / "commits" / f"{index['commit_id']}.json"
    manifest = _load_json(manifest_path)
    manifest_sha256 = _sha256_file(manifest_path)
    _validate_manifest(
        manifest,
        index=index,
        config=config,
        manifest_sha256=manifest_sha256,
    )

    samples, frame_set_sha256 = _load_samples(
        store_root=store_root,
        index=index,
        manifest=manifest,
        split_policy=config["split_policy"],
    )
    finite = tuple(sample for sample in samples if sample.risk_score is not None)
    fit = tuple(sample for sample in finite if sample.split == "fit")
    validation = tuple(sample for sample in finite if sample.split == "validation")
    test = tuple(sample for sample in finite if sample.split == "test")
    if not fit or not validation or not test:
        raise ValueError("calibration split must contain finite fit, validation, and test samples")

    baseline_thresholds = tuple(float(value) for value in config["baseline"]["thresholds"])
    _validate_strict_thresholds(baseline_thresholds, field="baseline thresholds")
    for sample in finite:
        expected = _level_for_score(float(sample.risk_score), baseline_thresholds)
        if sample.baseline_level != expected:
            raise ValueError(
                "formal RiskFrame level does not match frozen baseline policy: "
                f"{sample.frame_id}[{sample.row},{sample.column}]"
            )

    quantiles = tuple(float(value) for value in config["statistical_candidate"]["quantiles"])
    statistical_thresholds = tuple(
        _percentile(sorted(float(sample.risk_score) for sample in fit), quantile)
        for quantile in quantiles
    )
    _validate_strict_thresholds(statistical_thresholds, field="statistical thresholds")

    dataset_digest_input = {
        "risk_window_id": index["commit_id"],
        "risk_window_content_digest": manifest["content_digest"],
        "risk_window_manifest_sha256": manifest_sha256,
        "frame_index_sha256": index_sha256,
        "frame_set_sha256": frame_set_sha256,
        "config_sha256": config_sha256,
        "producer_version": _PRODUCER_VERSION,
        "producer_source_sha256": producer_source_sha256,
        "split_policy": config["split_policy"],
    }
    dataset_digest = _canonical_digest(dataset_digest_input)
    comparison = {
        "schema_version": _SIDECAR_VERSION,
        "publication_status": "RESEARCH_ONLY",
        "contract_status": "EXPERIMENTAL_UNREGISTERED",
        "authority": {
            "shadow_only": True,
            "formal_riskframe_write": False,
            "formal_threshold_write": False,
            "feeds_c": False,
            "feeds_d": False,
            "baseline_riskframe_remains_authoritative": True,
        },
        "experiment_id": config["experiment_id"],
        "identity": {
            "risk_window_id": index["commit_id"],
            "risk_window_content_digest": manifest["content_digest"],
            "risk_window_manifest_sha256": manifest_sha256,
            "run_id": index["run_id"],
            "scenario_id": index["scenario_id"],
            "dataset_bundle_id": index["dataset_bundle_id"],
            "dataset_bundle_digest": index["dataset_bundle_digest"],
            "model_config_digest": index["model_config_digest"],
            "frame_index_sha256": index_sha256,
            "frame_set_sha256": frame_set_sha256,
            "config_sha256": config_sha256,
            "producer_version": _PRODUCER_VERSION,
            "producer_source_sha256": producer_source_sha256,
            "immutable_dataset_id": f"shadow-dataset-sha256-{dataset_digest}",
        },
        "semantics": config["semantics"],
        "dataset": _dataset_summary(samples, split_policy=config["split_policy"]),
        "metric_definitions": config["metric_definitions"],
        "candidate_comparison": [
            _executed_candidate(
                candidate_id="equal_width_baseline",
                status="EXECUTED_REFERENCE",
                semantics="discretized_hazard_index",
                thresholds=baseline_thresholds,
                fit=fit,
                validation=validation,
                test=test,
                stability={
                    "status": "FIXED_BY_POLICY",
                    "temporal_threshold_max_range": 0.0,
                    "spatial_threshold_max_range": 0.0,
                },
            ),
            _executed_candidate(
                candidate_id="fit_quantile_statistical",
                status="EXECUTED_DESCRIPTIVE",
                semantics="distribution_relative_hazard_stratum",
                thresholds=statistical_thresholds,
                fit=fit,
                validation=validation,
                test=test,
                stability=_statistical_stability(
                    fit,
                    quantiles=quantiles,
                    split_policy=config["split_policy"],
                ),
                comparison_thresholds=baseline_thresholds,
            ),
            _blocked_candidate(
                candidate_id="expert_rule_calibration",
                semantics="operational_action_level",
                blocker="BLOCKED_BY_APPROVED_EXPERT_RULESET",
                required_evidence=(
                    "versioned expert rulebook",
                    "inter-rater agreement",
                    "vessel and corridor applicability",
                ),
            ),
            _blocked_candidate(
                candidate_id="physics_constraint_gate",
                semantics="non_compensatory_severity_guard",
                blocker="BLOCKED_BY_COMPONENT_ATTRIBUTION_AND_APPROVED_LIMITS",
                required_evidence=(
                    "B-owned component contribution sidecar",
                    "approved vessel-specific physical limits",
                    "monotonic and fail-closed gate rules",
                ),
            ),
            _blocked_candidate(
                candidate_id="ordinal_calibration",
                semantics="ordinal_operational_severity",
                blocker="BLOCKED_BY_ORDINAL_LABELS",
                required_evidence=(
                    "immutable ordinal labels",
                    "label provenance and adjudication",
                    "grouped external validation scenarios",
                ),
            ),
        ],
        "route_utility": {
            "status": "NOT_EVALUATED",
            "reason": (
                "candidate levels are shadow-only and C was not run; existing route metrics "
                "remain reference evidence, not calibration utility evidence"
            ),
            "future_metrics": [
                "selected-route stability",
                "risk-time Pareto separation",
                "decision regret against adjudicated operational labels",
                "fail-closed route success rate",
            ],
        },
        "restrictions": [
            "does_not_modify_risk_score",
            "does_not_modify_formal_risk_level",
            "does_not_publish_to_risk_store",
            "does_not_run_c_or_d",
            "does_not_claim_probability_or_operational_calibration",
        ],
    }
    content_digest = _canonical_digest(comparison)
    return {
        **comparison,
        "content_digest": content_digest,
        "artifact_id": f"risk-calibration-shadow-sha256-{content_digest}",
    }


def write_shadow_comparison(
    *,
    frame_index_path: Path,
    config_path: Path,
    output_path: Path,
    approved_output_root: Path,
) -> dict[str, Any]:
    """Atomically create a no-clobber sidecar below an approved output root."""

    source_root = frame_index_path.resolve().parent
    approved_root = approved_output_root.resolve(strict=True)
    resolved_output = output_path.resolve(strict=False)
    if not resolved_output.is_relative_to(approved_root) or resolved_output == approved_root:
        raise ValueError("shadow output must be inside the approved output root")
    if resolved_output.is_relative_to(source_root):
        raise ValueError("shadow output must not be written inside the source experiment")
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"shadow output already exists: {output_path}")
    document = build_shadow_comparison(
        frame_index_path=frame_index_path,
        config_path=config_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = output_path.parent.resolve(strict=True)
    if not resolved_parent.is_relative_to(approved_root):
        raise ValueError("shadow output parent escapes the approved output root")
    temporary = output_path.with_name(f".{output_path.name}.{document['content_digest']}.part")
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return document


def _validate_config(config: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "source",
        "semantics",
        "split_policy",
        "baseline",
        "statistical_candidate",
        "metric_definitions",
    }
    if set(config) != required:
        raise ValueError("shadow experiment config fields do not match v1")
    if config["schema_version"] != _CONFIG_VERSION:
        raise ValueError("unsupported shadow experiment config")
    source_required = {
        "risk_window_id",
        "risk_window_content_digest",
        "risk_window_manifest_sha256",
        "run_id",
        "scenario_id",
        "frame_index_sha256",
        "frame_count",
        "interval_seconds",
    }
    if set(config["source"]) != source_required:
        raise ValueError("shadow source identity fields do not match v1")
    split = config["split_policy"]
    split_required = {
        "policy_version",
        "fit_frame_end_exclusive",
        "validation_frame_start",
        "validation_frame_end_exclusive",
        "test_frame_start",
        "time_purge_frames",
        "fit_row_range",
        "validation_row_range",
        "test_row_range",
        "spatial_purge_rows",
        "stability_time_block_frames",
        "stability_spatial_block_rows",
    }
    if set(split) != split_required:
        raise ValueError("shadow split policy fields do not match v1")
    fit_end = int(split["fit_frame_end_exclusive"])
    validation_start = int(split["validation_frame_start"])
    validation_end = int(split["validation_frame_end_exclusive"])
    test_start = int(split["test_frame_start"])
    time_purge = int(split["time_purge_frames"])
    if not 0 < fit_end <= validation_start < validation_end <= test_start:
        raise ValueError("calibration time ranges must be ordered and non-empty")
    if validation_start - fit_end < time_purge or test_start - validation_end < time_purge:
        raise ValueError("calibration time ranges do not satisfy purge gap")
    row_ranges = tuple(
        tuple(int(value) for value in split[field])
        for field in ("fit_row_range", "validation_row_range", "test_row_range")
    )
    if any(len(values) != 2 or not 0 <= values[0] < values[1] for values in row_ranges):
        raise ValueError("calibration row ranges must be ordered [start, end) pairs")
    spatial_purge = int(split["spatial_purge_rows"])
    if (
        row_ranges[1][0] - row_ranges[0][1] < spatial_purge
        or row_ranges[2][0] - row_ranges[1][1] < spatial_purge
    ):
        raise ValueError("calibration row ranges do not satisfy spatial purge gap")
    if int(split["stability_time_block_frames"]) < 1:
        raise ValueError("stability time block must be positive")
    if int(split["stability_spatial_block_rows"]) < 1:
        raise ValueError("stability spatial block must be positive")
    quantiles = tuple(float(value) for value in config["statistical_candidate"]["quantiles"])
    if quantiles != tuple(sorted(quantiles)) or len(quantiles) != 4:
        raise ValueError("statistical candidate requires four sorted quantiles")
    if any(value <= 0 or value >= 1 for value in quantiles):
        raise ValueError("statistical quantiles must be within (0, 1)")
    if config["baseline"] != {
        "policy_id": "c_equal_width_floor_v1",
        "thresholds": [0.2, 0.4, 0.6, 0.8],
    }:
        raise ValueError("shadow baseline must match the frozen equal-width policy")
    if config["statistical_candidate"] != {
        "policy_id": "fit_quantile_strata_v1",
        "quantiles": [0.2, 0.4, 0.6, 0.8],
        "status": "DESCRIPTIVE_ONLY",
    }:
        raise ValueError("unsupported statistical shadow candidate")


def _validate_index(index: dict[str, Any], *, config: dict[str, Any], index_sha256: str) -> None:
    source = config["source"]
    expected = {
        "commit_id": source["risk_window_id"],
        "run_id": source["run_id"],
        "scenario_id": source["scenario_id"],
        "frame_schema": _RISK_FRAME_VERSION,
    }
    mismatched = [key for key, value in expected.items() if index.get(key) != value]
    if index_sha256 != source["frame_index_sha256"]:
        mismatched.append("frame_index_sha256")
    if len(index.get("frame_ids", ())) != int(source["frame_count"]):
        mismatched.append("frame_count")
    if index.get("content_digest") != source["risk_window_content_digest"]:
        mismatched.append("content_digest")
    frame_ids = index.get("frame_ids", ())
    if len(frame_ids) != len(set(frame_ids)) or any(
        not isinstance(frame_id, str) or _RISK_ID.fullmatch(frame_id) is None
        for frame_id in frame_ids
    ):
        mismatched.append("frame_ids")
    if _WINDOW_ID.fullmatch(str(index.get("commit_id"))) is None:
        mismatched.append("commit_id")
    if mismatched:
        raise ValueError("shadow source identity mismatch: " + ", ".join(mismatched))


def _safe_store_root(frame_index_path: Path, raw_store: Any) -> Path:
    if not isinstance(raw_store, str) or not raw_store:
        raise ValueError("risk_store must be a non-empty relative path")
    store_path = Path(raw_store)
    if store_path.is_absolute() or ".." in store_path.parts:
        raise ValueError("risk_store must stay inside the source experiment")
    source_root = frame_index_path.resolve().parent
    resolved = (source_root / store_path).resolve(strict=True)
    if not resolved.is_relative_to(source_root):
        raise ValueError("risk_store escapes the source experiment")
    return resolved


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    index: dict[str, Any],
    config: dict[str, Any],
    manifest_sha256: str,
) -> None:
    source = config["source"]
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("formal RiskWindow manifest fields mismatch")
    if manifest_sha256 != source["risk_window_manifest_sha256"]:
        raise ValueError("formal RiskWindow manifest SHA-256 mismatch")
    window_match = _WINDOW_ID.fullmatch(str(manifest.get("commit_id")))
    if window_match is None or window_match.group(1) != manifest.get("content_digest"):
        raise ValueError("formal RiskWindow identity is not canonical")
    expected = {
        "schema_version": _RISK_WINDOW_VERSION,
        "commit_id": index["commit_id"],
        "content_digest": source["risk_window_content_digest"],
        "run_id": index["run_id"],
        "scenario_id": index["scenario_id"],
        "model_config_digest": index["model_config_digest"],
        "count": int(source["frame_count"]),
        "interval_seconds": int(source["interval_seconds"]),
    }
    mismatched = [key for key, value in expected.items() if manifest.get(key) != value]
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) != expected["count"]:
        mismatched.append("frames")
    else:
        manifest_ids = []
        for item in raw_frames:
            if not isinstance(item, dict) or set(item) != {"risk_id", "content_digest"}:
                mismatched.append("frame_entry")
                break
            match = _RISK_ID.fullmatch(str(item["risk_id"]))
            if match is None or match.group(1) != item["content_digest"]:
                mismatched.append("frame_content_digest")
                break
            manifest_ids.append(item["risk_id"])
        if manifest_ids != index["frame_ids"]:
            mismatched.append("frame_order")
    if mismatched:
        raise ValueError("formal RiskWindow manifest mismatch: " + ", ".join(mismatched))


def _load_samples(
    *,
    store_root: Path,
    index: dict[str, Any],
    manifest: dict[str, Any],
    split_policy: dict[str, Any],
) -> tuple[tuple[CalibrationSample, ...], str]:
    frame_root = store_root / "frames"
    samples: list[CalibrationSample] = []
    frame_hasher = hashlib.sha256()
    expected_shape: tuple[int, int] | None = None
    expected_coordinates: tuple[tuple[float, ...], tuple[float, ...]] | None = None
    previous_valid_time: datetime | None = None
    interval = timedelta(seconds=int(manifest["interval_seconds"]))
    start = _parse_utc(manifest["start"], field="RiskWindow.start")
    end = _parse_utc(manifest["end"], field="RiskWindow.end")
    for frame_number, frame_id in enumerate(index["frame_ids"]):
        path = frame_root / f"{frame_id}.json"
        raw = path.read_bytes()
        frame_hasher.update(frame_id.encode("utf-8"))
        frame_hasher.update(b"\0")
        frame_hasher.update(raw)
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError(f"RiskFrame JSON object required: {frame_id}")
        if set(document) != _FRAME_FIELDS:
            raise ValueError(f"RiskFrame top-level fields mismatch: {frame_id}")
        content_document = {key: value for key, value in document.items() if key != "risk_id"}
        expected_digest = frame_id.removeprefix("risk-sha256-")
        if _canonical_digest(content_document) != expected_digest:
            raise ValueError(f"RiskFrame canonical content digest mismatch: {frame_id}")
        if document.get("schema_version") != _RISK_FRAME_VERSION:
            raise ValueError(f"unsupported RiskFrame schema: {frame_id}")
        for key in ("risk_id", "run_id", "scenario_id", "model_config_digest"):
            expected = frame_id if key == "risk_id" else index[key]
            if document.get(key) != expected:
                raise ValueError(f"RiskFrame {key} mismatch: {frame_id}")
        manifest_identity = {
            "corridor_id": manifest["corridor_id"],
            "vessel_profile_id": manifest["vessel_profile_id"],
            "config_digest": manifest["config_digest"],
            "generation_id": manifest["generation_id"],
            "as_of_time": manifest["as_of"],
            "provenance": "formal",
        }
        if any(document.get(key) != value for key, value in manifest_identity.items()):
            raise ValueError(f"RiskFrame formal window identity mismatch: {frame_id}")
        _parse_utc(document["generated_at"], field=f"{frame_id}.generated_at")
        payload = document["payload"]
        if set(payload) != {"coordinates", "variables", "attributes"}:
            raise ValueError(f"RiskFrame payload fields mismatch: {frame_id}")
        latitudes = payload["coordinates"]["latitude"]
        longitudes = payload["coordinates"]["longitude"]
        variables = payload["variables"]
        required_variables = {"risk_score", "risk_level", "hard_mask", "hard_reason", "confidence"}
        if not required_variables.issubset(variables):
            raise ValueError(f"RiskFrame required variables missing: {frame_id}")
        scores = variables["risk_score"]
        levels = variables["risk_level"]
        hard_masks = variables["hard_mask"]
        reasons = variables["hard_reason"]
        confidences = variables["confidence"]
        shape = (len(latitudes), len(longitudes))
        coordinates = (
            tuple(float(value) for value in latitudes),
            tuple(float(value) for value in longitudes),
        )
        if any(not math.isfinite(value) for axis in coordinates for value in axis):
            raise ValueError("RiskFrame coordinates must be finite")
        if any(right <= left for axis in coordinates for left, right in pairwise(axis)):
            raise ValueError("RiskFrame coordinates must be strictly increasing")
        if expected_shape is None:
            expected_shape = shape
            expected_coordinates = coordinates
            for field in ("fit_row_range", "validation_row_range", "test_row_range"):
                if int(split_policy[field][1]) > shape[0]:
                    raise ValueError(f"{field} exceeds RiskFrame row count")
        if shape != expected_shape:
            raise ValueError("RiskFrame grid shape changed within committed window")
        if coordinates != expected_coordinates:
            raise ValueError("RiskFrame coordinates changed within committed window")
        arrays = (scores, levels, hard_masks, reasons, confidences)
        if any(len(array) != shape[0] for array in arrays) or any(
            len(row_values) != shape[1]
            for array in arrays
            for row_values in array
        ):
            raise ValueError(f"RiskFrame variable shape mismatch: {frame_id}")
        valid_time = _parse_utc(document["valid_time"], field=f"{frame_id}.valid_time")
        expected_time = start + frame_number * interval
        if valid_time != expected_time or (
            previous_valid_time is not None and valid_time <= previous_valid_time
        ):
            raise ValueError("RiskFrame valid_time order or cadence mismatch")
        previous_valid_time = valid_time
        for row in range(shape[0]):
            for column in range(shape[1]):
                raw_score = scores[row][column]
                score = None if raw_score is None else float(raw_score)
                reason = str(reasons[row][column])
                level = int(levels[row][column])
                hard_mask = hard_masks[row][column]
                confidence = float(confidences[row][column])
                if reason not in _HARD_REASONS:
                    raise ValueError("RiskFrame hard_reason is not canonical")
                if level not in range(1, 6):
                    raise ValueError("RiskFrame risk_level must be within [1, 5]")
                if not isinstance(hard_mask, bool):
                    raise ValueError("RiskFrame hard_mask must be boolean")
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("RiskFrame confidence must be within [0, 1]")
                if score is None and (
                    reason == "NONE" or not hard_mask or level != 5 or confidence != 0
                ):
                    raise ValueError("non-finite risk must remain fail-closed")
                if score is not None and (
                    not math.isfinite(score)
                    or reason != "NONE"
                    or hard_mask
                    or confidence <= 0
                ):
                    raise ValueError("finite risk and hard reason semantics are inconsistent")
                samples.append(
                    CalibrationSample(
                        frame_index=frame_number,
                        frame_id=frame_id,
                        valid_time=document["valid_time"],
                        row=row,
                        column=column,
                        risk_score=score,
                        baseline_level=level,
                        hard_reason=reason,
                        split=_assign_split(
                            frame_index=frame_number,
                            row=row,
                            column=column,
                            policy=split_policy,
                        ),
                    )
                )
    if previous_valid_time != end:
        raise ValueError("RiskFrame window end does not match formal manifest")
    return tuple(samples), frame_hasher.hexdigest()


def _parse_utc(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{field} must be UTC")
    return parsed.astimezone(UTC)


def _assign_split(*, frame_index: int, row: int, column: int, policy: dict[str, Any]) -> str:
    del column
    fit_rows = tuple(int(value) for value in policy["fit_row_range"])
    validation_rows = tuple(int(value) for value in policy["validation_row_range"])
    test_rows = tuple(int(value) for value in policy["test_row_range"])
    if frame_index < int(policy["fit_frame_end_exclusive"]) and _within(row, fit_rows):
        return "fit"
    if (
        int(policy["validation_frame_start"])
        <= frame_index
        < int(policy["validation_frame_end_exclusive"])
        and _within(row, validation_rows)
    ):
        return "validation"
    if frame_index >= int(policy["test_frame_start"]) and _within(row, test_rows):
        return "test"
    return "guard_excluded"


def _within(value: int, bounds: tuple[int, ...]) -> bool:
    return bounds[0] <= value < bounds[1]


def _dataset_summary(
    samples: tuple[CalibrationSample, ...], *, split_policy: dict[str, Any]
) -> dict[str, Any]:
    availability = Counter(sample.hard_reason for sample in samples)
    split_counts: dict[str, dict[str, int]] = {}
    for split in ("fit", "validation", "test", "guard_excluded"):
        selected = tuple(sample for sample in samples if sample.split == split)
        split_counts[split] = {
            "total": len(selected),
            "finite": sum(sample.risk_score is not None for sample in selected),
            "hard": sum(sample.risk_score is None for sample in selected),
        }
    return {
        "source_cell_count": len(samples),
        "finite_cell_count": sum(sample.risk_score is not None for sample in samples),
        "split_validity": "DIAGNOSTIC_ONLY_NOT_EXTERNAL_CALIBRATION",
        "availability": dict(sorted(availability.items())),
        "split_counts": split_counts,
        "split_policy": split_policy,
        "leakage_controls": {
            "time": "disjoint contiguous frame ranges with explicit purge gaps",
            "space": "disjoint latitude-row regions with explicit guard rows",
            "guard": "time/space purge samples and cross-region samples are excluded",
            "external_validity": (
                "single-scenario shadow evidence only; independent scenarios required"
            ),
        },
    }


def _executed_candidate(
    *,
    candidate_id: str,
    status: str,
    semantics: str,
    thresholds: tuple[float, ...],
    fit: tuple[CalibrationSample, ...],
    validation: tuple[CalibrationSample, ...],
    test: tuple[CalibrationSample, ...],
    stability: dict[str, Any],
    comparison_thresholds: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    result = {
        "candidate_id": candidate_id,
        "status": status,
        "semantics": semantics,
        "thresholds": list(thresholds),
        "fit": _evaluation(fit, thresholds),
        "validation": _evaluation(validation, thresholds),
        "test": _evaluation(test, thresholds),
        "stability": stability,
        "route_utility_status": "NOT_EVALUATED",
    }
    if comparison_thresholds is not None:
        result["agreement_with_baseline"] = {
            "validation": _agreement(validation, thresholds, comparison_thresholds),
            "test": _agreement(test, thresholds, comparison_thresholds),
        }
    return result


def _blocked_candidate(
    *, candidate_id: str, semantics: str, blocker: str, required_evidence: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": "BLOCKED",
        "semantics": semantics,
        "blocker": blocker,
        "required_evidence": list(required_evidence),
        "thresholds": None,
        "fit": None,
        "validation": None,
        "test": None,
        "route_utility_status": "NOT_EVALUATED",
    }


def _evaluation(
    samples: tuple[CalibrationSample, ...], thresholds: tuple[float, ...]
) -> dict[str, Any]:
    scores = [float(sample.risk_score) for sample in samples]
    levels = [_level_for_score(score, thresholds) for score in scores]
    grouped: dict[int, list[float]] = defaultdict(list)
    for score, level in zip(scores, levels, strict=True):
        grouped[level].append(score)
    total_mean = fmean(scores)
    total_ss = math.fsum((score - total_mean) ** 2 for score in scores)
    between_ss = math.fsum(
        len(values) * (fmean(values) - total_mean) ** 2 for values in grouped.values()
    )
    sorted_pairs = sorted(zip(scores, levels, strict=True))
    violations = sum(
        right_level < left_level
        for (_, left_level), (_, right_level) in pairwise(sorted_pairs)
    )
    medians = {level: median(values) for level, values in sorted(grouped.items())}
    adjacent_deltas = {
        f"L{left}_to_L{right}": medians[right] - medians[left]
        for left, right in pairwise(medians)
    }
    counts = Counter(levels)
    return {
        "sample_count": len(samples),
        "level_counts": {str(level): counts.get(level, 0) for level in range(1, 6)},
        "level_share": {
            str(level): counts.get(level, 0) / len(samples) for level in range(1, 6)
        },
        "score_mean": total_mean,
        "score_min": min(scores),
        "score_max": max(scores),
        "separation": {
            "between_level_variance_ratio": between_ss / total_ss if total_ss else None,
            "adjacent_level_median_delta": adjacent_deltas,
            "interpretation": "descriptive score separation, not outcome discrimination",
        },
        "monotonicity": {
            "mapping_violations": violations,
            "interpretation": "mapping property only; not physical-response validation",
        },
    }


def _agreement(
    samples: tuple[CalibrationSample, ...],
    candidate_thresholds: tuple[float, ...],
    baseline_thresholds: tuple[float, ...],
) -> dict[str, Any]:
    absolute_differences = []
    exact = 0
    for sample in samples:
        score = float(sample.risk_score)
        candidate = _level_for_score(score, candidate_thresholds)
        baseline = _level_for_score(score, baseline_thresholds)
        exact += candidate == baseline
        absolute_differences.append(abs(candidate - baseline))
    return {
        "exact_level_agreement": exact / len(samples),
        "mean_absolute_level_difference": fmean(absolute_differences),
        "interpretation": "behavioral delta only; higher disagreement is not improvement",
    }


def _statistical_stability(
    fit: tuple[CalibrationSample, ...],
    *,
    quantiles: tuple[float, ...],
    split_policy: dict[str, Any],
) -> dict[str, Any]:
    time_block_size = int(split_policy["stability_time_block_frames"])
    spatial_block_size = int(split_policy["stability_spatial_block_rows"])
    time_groups: dict[int, list[float]] = defaultdict(list)
    spatial_groups: dict[int, list[float]] = defaultdict(list)
    for sample in fit:
        score = float(sample.risk_score)
        time_groups[sample.frame_index // time_block_size].append(score)
        spatial_groups[sample.row // spatial_block_size].append(score)
    temporal = _group_thresholds(time_groups, quantiles)
    spatial = _group_thresholds(spatial_groups, quantiles)
    return {
        "status": "DESCRIPTIVE_ONLY",
        "temporal_fit_blocks": temporal,
        "spatial_fit_blocks": spatial,
        "temporal_threshold_max_range": _maximum_threshold_range(temporal),
        "spatial_threshold_max_range": _maximum_threshold_range(spatial),
        "interpretation": "internal sensitivity only; independent scenarios remain required",
    }


def _group_thresholds(
    groups: dict[int, list[float]], quantiles: tuple[float, ...]
) -> dict[str, list[float]]:
    return {
        str(group): [_percentile(sorted(values), quantile) for quantile in quantiles]
        for group, values in sorted(groups.items())
        if values
    }


def _maximum_threshold_range(groups: dict[str, list[float]]) -> float | None:
    if len(groups) < 2:
        return None
    columns = tuple(zip(*groups.values(), strict=True))
    return max(max(column) - min(column) for column in columns)


def _level_for_score(score: float, thresholds: tuple[float, ...]) -> int:
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError("finite risk score must be within [0, 1]")
    return bisect.bisect_right(thresholds, score) + 1


def _validate_strict_thresholds(thresholds: tuple[float, ...], *, field: str) -> None:
    if len(thresholds) != 4 or any(not 0 < value < 1 for value in thresholds):
        raise ValueError(f"{field} must contain four values within (0, 1)")
    if any(right <= left for left, right in pairwise(thresholds)):
        raise ValueError(f"{field} must be strictly increasing")


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value
