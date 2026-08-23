from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from arctic_route_risk.calibration_shadow import (
    build_shadow_comparison,
    write_shadow_comparison,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _baseline_level(score: float) -> int:
    if score < 0.2:
        return 1
    if score < 0.4:
        return 2
    if score < 0.6:
        return 3
    if score < 0.8:
        return 4
    return 5


def _fixture(
    tmp_path: Path,
    *,
    test_score: float | None = None,
    invalid_semantics: str | None = None,
    invalid_time_order: bool = False,
) -> tuple[Path, Path]:
    experiment = tmp_path / "source"
    frame_ids = []
    frame_entries = []
    for frame_index in range(9):
        scores = []
        levels = []
        hard_masks = []
        reasons = []
        confidences = []
        for row in range(9):
            score_row = []
            level_row = []
            hard_mask_row = []
            reason_row = []
            confidence_row = []
            for column in range(6):
                if row == 0 and column == 0:
                    score_row.append(None)
                    level_row.append(5)
                    hard_mask_row.append(True)
                    reason_row.append("LAND")
                    confidence_row.append(0.0)
                else:
                    score = min(0.95, 0.03 + 0.02 * row + 0.015 * column + 0.03 * frame_index)
                    if test_score is not None and frame_index == 8 and row == 8 and column == 5:
                        score = test_score
                    score_row.append(score)
                    level_row.append(_baseline_level(score))
                    hard_mask_row.append(False)
                    reason_row.append("NONE")
                    confidence_row.append(0.75)
            scores.append(score_row)
            levels.append(level_row)
            hard_masks.append(hard_mask_row)
            reasons.append(reason_row)
            confidences.append(confidence_row)
        if invalid_semantics == "hard_mask" and frame_index == 0:
            hard_masks[0][0] = False
        if invalid_semantics == "confidence" and frame_index == 0:
            confidences[0][0] = 0.5
        if invalid_semantics == "hard_reason" and frame_index == 0:
            reasons[0][0] = "INVENTED"
        if invalid_semantics == "unknown_safe" and frame_index == 0:
            reasons[0][0] = "NONE"
        if invalid_semantics == "level" and frame_index == 0:
            levels[1][1] = 5
        valid_hour = frame_index
        if invalid_time_order and frame_index == 5:
            valid_hour = 4
        document = {
            "schema_version": "bc.risk-frame.v2",
            "run_id": "run-test",
            "scenario_id": "scenario-test",
            "corridor_id": "corridor-test",
            "vessel_profile_id": "vessel-test",
            "config_digest": "4" * 64,
            "model_config_digest": "1" * 64,
            "generation_id": 0,
            "valid_time": f"2026-02-15T0{valid_hour}:00:00Z",
            "as_of_time": "2026-02-16T00:00:00Z",
            "generated_at": "2026-02-16T01:00:00Z",
            "model_version": "demo_unvalidated_rule_baseline.v2",
            "payload": {
                "coordinates": {
                    "latitude": [float(value) for value in range(9)],
                    "longitude": [float(value) for value in range(6)],
                },
                "variables": {
                    "risk_score": scores,
                    "risk_level": levels,
                    "hard_mask": hard_masks,
                    "hard_reason": reasons,
                    "confidence": confidences,
                },
                "attributes": {"calibration_status": "demo_unvalidated"},
            },
            "source_summary": [],
            "provenance": "formal",
        }
        content_digest = hashlib.sha256(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        frame_id = f"risk-sha256-{content_digest}"
        document["risk_id"] = frame_id
        frame_ids.append(frame_id)
        frame_entries.append({"risk_id": frame_id, "content_digest": content_digest})
        _write_json(experiment / "risk-store/frames" / f"{frame_id}.json", document)
    window_digest = "2" * 64
    commit_id = "risk-window-sha256-" + window_digest
    manifest = {
        "schema_version": "bc.risk-window-commit.v1",
        "commit_id": commit_id,
        "content_digest": window_digest,
        "start": "2026-02-15T00:00:00Z",
        "end": "2026-02-15T08:00:00Z",
        "interval_seconds": 3600,
        "count": 9,
        "run_id": "run-test",
        "scenario_id": "scenario-test",
        "corridor_id": "corridor-test",
        "generation_id": 0,
        "vessel_profile_id": "vessel-test",
        "config_digest": "4" * 64,
        "model_config_digest": "1" * 64,
        "as_of": "2026-02-16T00:00:00Z",
        "frames": frame_entries,
    }
    manifest_path = experiment / "risk-store/commits" / f"{commit_id}.json"
    _write_json(manifest_path, manifest)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    index = {
        "status": "FORMAL_VALIDATED",
        "commit_id": commit_id,
        "content_digest": window_digest,
        "run_id": "run-test",
        "scenario_id": "scenario-test",
        "dataset_bundle_id": "bundle-test",
        "dataset_bundle_digest": "3" * 64,
        "model_config_digest": "1" * 64,
        "frame_schema": "bc.risk-frame.v2",
        "risk_store": "risk-store",
        "frame_ids": frame_ids,
    }
    index_path = experiment / "frame-index.json"
    _write_json(index_path, index)
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()
    config = {
        "schema_version": "b.risk-calibration-shadow-experiment.v1",
        "experiment_id": "test-shadow",
        "source": {
            "risk_window_id": index["commit_id"],
            "risk_window_content_digest": window_digest,
            "risk_window_manifest_sha256": manifest_sha,
            "run_id": index["run_id"],
            "scenario_id": index["scenario_id"],
            "frame_index_sha256": index_sha,
            "frame_count": 9,
            "interval_seconds": 3600,
        },
        "semantics": {
            "risk_score": "hazard_index",
            "baseline_risk_level": "discretized_hazard_index",
            "recommended_research_target": "operational_action_level",
            "probability_claim": "NOT_SUPPORTED",
            "severity_claim": "NOT_CALIBRATED",
        },
        "split_policy": {
            "policy_version": "test_split_v1",
            "fit_frame_end_exclusive": 3,
            "validation_frame_start": 4,
            "validation_frame_end_exclusive": 6,
            "test_frame_start": 7,
            "time_purge_frames": 1,
            "fit_row_range": [0, 3],
            "validation_row_range": [4, 6],
            "test_row_range": [7, 9],
            "spatial_purge_rows": 1,
            "stability_time_block_frames": 1,
            "stability_spatial_block_rows": 1,
        },
        "baseline": {
            "policy_id": "c_equal_width_floor_v1",
            "thresholds": [0.2, 0.4, 0.6, 0.8],
        },
        "statistical_candidate": {
            "policy_id": "fit_quantile_strata_v1",
            "quantiles": [0.2, 0.4, 0.6, 0.8],
            "status": "DESCRIPTIVE_ONLY",
        },
        "metric_definitions": {
            "separation": "test",
            "monotonicity": "test",
            "route_utility": "test",
            "stability": "test",
        },
    }
    config_path = tmp_path / "config.json"
    _write_json(config_path, config)
    return index_path, config_path


def test_shadow_comparison_is_deterministic_and_keeps_blocked_candidates(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path)

    first = build_shadow_comparison(frame_index_path=index_path, config_path=config_path)
    second = build_shadow_comparison(frame_index_path=index_path, config_path=config_path)

    assert first == second
    assert first["publication_status"] == "RESEARCH_ONLY"
    assert first["contract_status"] == "EXPERIMENTAL_UNREGISTERED"
    assert first["authority"]["formal_riskframe_write"] is False
    assert first["authority"]["feeds_c"] is False
    assert first["dataset"]["split_validity"] == "DIAGNOSTIC_ONLY_NOT_EXTERNAL_CALIBRATION"
    assert [item["status"] for item in first["candidate_comparison"]] == [
        "EXECUTED_REFERENCE",
        "EXECUTED_DESCRIPTIVE",
        "BLOCKED",
        "BLOCKED",
        "BLOCKED",
    ]
    assert first["route_utility"]["status"] == "NOT_EVALUATED"
    assert first["dataset"]["availability"]["LAND"] == 9


def test_fit_thresholds_do_not_use_test_values(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path / "before")
    before = build_shadow_comparison(frame_index_path=index_path, config_path=config_path)
    index_path, config_path = _fixture(tmp_path / "after", test_score=0.99)
    after = build_shadow_comparison(frame_index_path=index_path, config_path=config_path)

    assert before["candidate_comparison"][1]["thresholds"] == after[
        "candidate_comparison"
    ][1]["thresholds"]
    assert before["identity"]["frame_set_sha256"] != after["identity"]["frame_set_sha256"]


def test_formal_level_mismatch_fails_closed(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path, invalid_semantics="level")

    with pytest.raises(ValueError, match="frozen baseline policy"):
        build_shadow_comparison(frame_index_path=index_path, config_path=config_path)


def test_unknown_is_never_calibrated_as_safe(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path, invalid_semantics="unknown_safe")

    with pytest.raises(ValueError, match="fail-closed"):
        build_shadow_comparison(frame_index_path=index_path, config_path=config_path)


def test_writer_rejects_output_inside_source_experiment(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path)

    with pytest.raises(ValueError, match="must not be written inside"):
        write_shadow_comparison(
            frame_index_path=index_path,
            config_path=config_path,
            output_path=index_path.parent / "comparison.json",
            approved_output_root=tmp_path,
        )


def test_writer_emits_atomic_sidecar_outside_source(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path)
    output = tmp_path / "output/comparison.json"

    document = write_shadow_comparison(
        frame_index_path=index_path,
        config_path=config_path,
        output_path=output,
        approved_output_root=tmp_path,
    )

    assert output.is_file()
    assert not output.with_suffix(".json.part").exists()
    assert json.loads(output.read_text(encoding="utf-8")) == document

    with pytest.raises(FileExistsError):
        write_shadow_comparison(
            frame_index_path=index_path,
            config_path=config_path,
            output_path=output,
            approved_output_root=tmp_path,
        )


@pytest.mark.parametrize("invalid_semantics", ["hard_mask", "confidence", "hard_reason"])
def test_formal_hard_semantics_fail_closed(
    tmp_path: Path, invalid_semantics: str
) -> None:
    index_path, config_path = _fixture(
        tmp_path, invalid_semantics=invalid_semantics
    )

    with pytest.raises(ValueError, match=r"RiskFrame|fail-closed"):
        build_shadow_comparison(frame_index_path=index_path, config_path=config_path)


def test_valid_time_order_and_cadence_fail_closed(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path, invalid_time_order=True)

    with pytest.raises(ValueError, match="valid_time order or cadence"):
        build_shadow_comparison(frame_index_path=index_path, config_path=config_path)


def test_writer_rejects_output_outside_approved_root(tmp_path: Path) -> None:
    index_path, config_path = _fixture(tmp_path / "fixture")
    approved = tmp_path / "approved"
    approved.mkdir()

    with pytest.raises(ValueError, match="approved output root"):
        write_shadow_comparison(
            frame_index_path=index_path,
            config_path=config_path,
            output_path=tmp_path / "outside.json",
            approved_output_root=approved,
        )
