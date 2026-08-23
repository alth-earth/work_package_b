from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from arctic_route_data import semantic_payload_digest
from arctic_route_planning.contracts import canonical_risk_frame_bytes
from jsonschema import Draft202012Validator, FormatChecker

from arctic_route_risk import (
    BInputEnvelope,
    InputIdentityError,
    PersistentRiskStore,
    RiskBuildRequest,
    RiskBuildService,
    RiskBuildTraceResult,
    RiskExplanationResearchExporter,
    RiskPipelineError,
)
from arctic_route_risk.risk_explanation import (
    RiskComponentTrace,
    RiskExplanationFrameTrace,
    RiskExplanationTraceWindow,
)

GENERATED = datetime(2026, 8, 2, tzinfo=UTC)
EXPORTED = datetime(2026, 8, 3, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).parents[2]
SCHEMA_PATH = (
    PROJECT_ROOT.parent
    / "arctic_route_governance"
    / "current"
    / "proposals"
    / "risk-explanation.v1.schema.json"
)


def _build_and_export(
    tmp_path,
    formal_fixture,
    *,
    missing_data_types=(),
    prepared_override=None,
):
    prepared = prepared_override or _with_missing_inputs(
        formal_fixture.prepared,
        missing_data_types,
    )
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=prepared,
        generation_id=prepared.generation_id,
        knowledge_as_of=prepared.as_of_time,
    )
    request = RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)
    result = RiskBuildService(
        utc_now=lambda: GENERATED
    ).build_window_with_explanation_trace(request)
    store = PersistentRiskStore(tmp_path / "risk-store")
    store.activate_generation(result.frames[0].run_id, result.frames[0].generation_id)
    committed = store.publish_window(result.frames)
    document = RiskExplanationResearchExporter(utc_now=lambda: EXPORTED).export(
        committed_window=committed,
        build_result=result,
    )
    return result, committed, document


def _with_missing_inputs(prepared, missing_data_types):
    if not missing_data_types:
        return prepared
    frames = dict(prepared.frames)
    attestations = dict(prepared.payload_attestations)
    for data_type in missing_data_types:
        changed = []
        for source in frames[data_type]:
            copied = source.consumer_copy()
            for variable in copied.payload.data_vars:
                copied.payload[variable] = copied.payload[variable] * np.nan
            changed.append(copied)
            attestations[copied.record.data_id] = semantic_payload_digest(
                copied.record,
                copied.payload,
            )
        frames[data_type] = tuple(changed)
    return replace(prepared, frames=frames, payload_attestations=attestations)


def _with_zero_risk_inputs(prepared):
    values = {
        "land_sea_mask": 1.0,
        "ocean_current_u": 0.0,
        "ocean_current_v": 0.0,
        "ice_concentration": 0.0,
        "ice_drift_u": 0.0,
        "ice_drift_v": 0.0,
        "ice_edge": 0.0,
        "ice_thickness": 0.0,
        "ice_type": 0.0,
        "air_temperature_2m": 273.15,
        "visibility": 10_000.0,
        "sea_surface_height": 0.0,
        "significant_wave_height": 0.0,
        "wind_u10": 0.0,
        "wind_v10": 0.0,
    }
    frames = dict(prepared.frames)
    attestations = dict(prepared.payload_attestations)
    for data_type, source_frames in frames.items():
        changed = []
        for source in source_frames:
            copied = source.consumer_copy()
            for variable in copied.payload.data_vars:
                copied.payload[variable] = copied.payload[variable] * 0 + values[variable]
            changed.append(copied)
            attestations[copied.record.data_id] = semantic_payload_digest(
                copied.record,
                copied.payload,
            )
        frames[data_type] = tuple(changed)
    return replace(prepared, frames=frames, payload_attestations=attestations)


def _validate_schema(document) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(
        schema,
        format_checker=FormatChecker(),
    ).validate(document)
    json.loads(json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _replace_first_component(trace_window, replacement):
    first_frame = trace_window.frames[0]
    changed_frame = replace(
        first_frame,
        components=(replacement, *first_frame.components[1:]),
    )
    return replace(trace_window, frames=(changed_frame, *trace_window.frames[1:]))


def test_complete_export_matches_schema_and_contribution_sum(
    tmp_path, formal_fixture
) -> None:
    result, committed, document = _build_and_export(tmp_path, formal_fixture)

    _validate_schema(document)
    assert document["publication_status"] == "COMPLETE"
    assert document["identity"]["risk_window_id"] == committed.commit_id
    assert [frame["risk_frame_id"] for frame in document["frames"]] == [
        frame.risk_id for frame in committed.frames
    ]
    for frame in document["frames"]:
        for cell in frame["cells"]:
            assert cell["explanation_status"] == "COMPLETE"
            assert math.isclose(
                math.fsum(item["contribution"] for item in cell["contributors"]),
                cell["risk"]["score"],
                rel_tol=0.0,
                abs_tol=1e-6,
            )

    expected_ids = result.explanation_trace.formula_component_ids
    for trace_frame, sidecar_frame in zip(
        result.explanation_trace.frames,
        document["frames"],
        strict=True,
    ):
        for component in trace_frame.components:
            np.testing.assert_allclose(
                component.contribution,
                component.normalized_value * component.weight,
                rtol=0.0,
                atol=1e-12,
            )
        first_cell = sidecar_frame["cells"][0]
        published_ids = tuple(
            component_id
            for contributor in first_cell["contributors"]
            for component_id in contributor["component_ids"]
        )
        assert published_ids == expected_ids


def test_research_trace_build_preserves_formal_riskframe_bytes(formal_fixture) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    request = RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)
    service = RiskBuildService(utc_now=lambda: GENERATED)

    formal = service.build_window(request)
    research = service.build_window_with_explanation_trace(request)

    assert [frame.risk_id for frame in formal] == [frame.risk_id for frame in research.frames]
    for expected, observed in zip(formal, research.frames, strict=True):
        assert canonical_risk_frame_bytes(expected) == canonical_risk_frame_bytes(observed)


def test_contribution_sum_mismatch_fails_closed(tmp_path, formal_fixture) -> None:
    result, committed, _ = _build_and_export(tmp_path, formal_fixture)
    original = result.explanation_trace.frames[0].components[0]
    normalized = np.clip(original.normalized_value + 0.01, 0.0, 1.0)
    changed = RiskComponentTrace(
        component_id=original.component_id,
        normalized_value=normalized,
        weight=original.weight,
        contribution=normalized * original.weight,
    )
    trace = _replace_first_component(result.explanation_trace, changed)
    changed_result = RiskBuildTraceResult._from_pipeline(
        frames=result.frames,
        explanation_trace=trace,
    )

    with pytest.raises(RiskPipelineError, match="contribution sum"):
        RiskExplanationResearchExporter(utc_now=lambda: EXPORTED).export(
            committed_window=committed,
            build_result=changed_result,
        )


def test_identity_mismatch_rejects_entire_sidecar(tmp_path, formal_fixture) -> None:
    result, committed, _ = _build_and_export(tmp_path, formal_fixture)
    first = result.explanation_trace.frames[0]
    mismatched_frame = replace(first, grid_id="wrong-grid")
    trace = replace(
        result.explanation_trace,
        frames=(mismatched_frame, *result.explanation_trace.frames[1:]),
    )
    mismatched_result = RiskBuildTraceResult._from_pipeline(
        frames=result.frames,
        explanation_trace=trace,
    )

    with pytest.raises(InputIdentityError, match="grid_id"):
        RiskExplanationResearchExporter(utc_now=lambda: EXPORTED).export(
            committed_window=committed,
            build_result=mismatched_result,
        )


def test_modified_trace_cannot_reuse_pipeline_integrity_seal(
    tmp_path, formal_fixture
) -> None:
    result, _, _ = _build_and_export(tmp_path, formal_fixture)
    original = result.explanation_trace.frames[0].components[0]
    normalized = np.clip(original.normalized_value + 0.01, 0.0, 1.0)
    changed = RiskComponentTrace(
        component_id=original.component_id,
        normalized_value=normalized,
        weight=original.weight,
        contribution=normalized * original.weight,
    )
    trace = _replace_first_component(result.explanation_trace, changed)

    with pytest.raises(RiskPipelineError, match="integrity mismatch"):
        replace(result, explanation_trace=trace)


def test_missing_component_is_partial_and_never_filled_with_zero(
    tmp_path, formal_fixture
) -> None:
    result, _, document = _build_and_export(
        tmp_path,
        formal_fixture,
        missing_data_types=("wave",),
    )

    _validate_schema(document)
    assert document["publication_status"] == "PARTIAL"
    cell = document["frames"][0]["cells"][0]
    assert cell["explanation_status"] == "PARTIAL"
    assert cell["risk"] == {"score": None, "level": 5, "confidence": 0.0}
    assert "wave_height" in cell["uncertainty"]["explanation_gaps"]
    assert all(item["contributor_id"] != "wave" for item in cell["contributors"])
    wave = next(
        component
        for component in result.explanation_trace.frames[0].components
        if component.component_id == "wave_height"
    )
    assert np.all(np.isnan(wave.normalized_value))
    assert np.all(np.isnan(wave.contribution))


def test_missing_land_sea_validity_is_unavailable(tmp_path, formal_fixture) -> None:
    _, _, document = _build_and_export(
        tmp_path,
        formal_fixture,
        missing_data_types=("land_sea_mask",),
    )

    _validate_schema(document)
    assert document["publication_status"] == "UNAVAILABLE"
    cell = document["frames"][0]["cells"][0]
    assert cell["explanation_status"] == "UNAVAILABLE"
    assert cell["contributors"] == []
    assert cell["risk"] == {"score": None, "level": 5, "confidence": 0.0}
    assert cell["uncertainty"]["missing_data"] == [
        {"data_type": "land_sea_mask", "cause": "NON_FINITE_INPUT"}
    ]


def test_all_missing_components_publish_unavailable_without_contributors(
    tmp_path, formal_fixture
) -> None:
    missing = (
        "ocean_current",
        "sea_ice_concentration",
        "sea_ice_drift",
        "sea_ice_edge",
        "sea_ice_thickness",
        "sea_ice_type",
        "temperature",
        "visibility",
        "water_level",
        "wave",
        "wind_field",
    )
    _, _, document = _build_and_export(
        tmp_path,
        formal_fixture,
        missing_data_types=missing,
    )

    _validate_schema(document)
    assert document["publication_status"] == "UNAVAILABLE"
    for frame in document["frames"]:
        assert frame["coverage"]["unavailable_cell_count"] == len(frame["cells"])
        for cell in frame["cells"]:
            assert cell["explanation_status"] == "UNAVAILABLE"
            assert cell["contributors"] == []
            assert cell["reason"]["code"] == "EXPLANATION_UNAVAILABLE"
            assert cell["risk"] == {"score": None, "level": 5, "confidence": 0.0}


def test_complete_zero_risk_has_no_invented_dominant_contributor(
    tmp_path, formal_fixture
) -> None:
    prepared = _with_zero_risk_inputs(formal_fixture.prepared)
    _, _, document = _build_and_export(
        tmp_path,
        formal_fixture,
        prepared_override=prepared,
    )

    _validate_schema(document)
    cell = document["frames"][0]["cells"][0]
    assert cell["explanation_status"] == "COMPLETE"
    assert cell["risk"]["score"] == 0.0
    assert cell["reason"]["main_contributor_ids"] == []
    assert cell["reason"]["text"] == "所有已验证风险贡献均为 0，无主要贡献项"


def test_window_identity_mismatch_fails_before_frame_export(
    tmp_path, formal_fixture
) -> None:
    result, committed, _ = _build_and_export(tmp_path, formal_fixture)
    trace = replace(result.explanation_trace, model_config_digest="f" * 64)
    mismatched_result = RiskBuildTraceResult._from_pipeline(
        frames=result.frames,
        explanation_trace=trace,
    )

    with pytest.raises(InputIdentityError, match="model_config_digest"):
        RiskExplanationResearchExporter(utc_now=lambda: EXPORTED).export(
            committed_window=committed,
            build_result=mismatched_result,
        )


def test_trace_rejects_missing_formula_component() -> None:
    normalized = np.ones((1, 1), dtype=np.float64)
    component = RiskComponentTrace(
        component_id="ice_concentration",
        normalized_value=normalized,
        weight=0.24,
        contribution=normalized * 0.24,
    )
    frame = RiskExplanationFrameTrace(
        risk_frame_id="risk-id",
        frame_time=GENERATED,
        grid_id="grid-id",
        latitude=np.array([70.0]),
        longitude=np.array([10.0]),
        land_sea_valid=np.ones((1, 1), dtype=np.bool_),
        components=(component,),
    )

    with pytest.raises(RiskPipelineError, match="canonical formula order"):
        RiskExplanationTraceWindow(
            run_id="run-id",
            scenario_id="scenario",
            corridor_id="corridor",
            vessel_profile_id="vessel",
            config_digest="1" * 64,
            model_config_digest="2" * 64,
            generation_id=0,
            as_of_time=GENERATED,
            formula_version="deterministic_environment_components_v2",
            calibration_status="demo_unvalidated",
            formula_component_ids=("ice_concentration", "wave_height"),
            frames=(frame,),
        )
