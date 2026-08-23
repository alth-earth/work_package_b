"""Research-only producer for ``risk-explanation.v1`` sidecars.

The contributor evidence consumed here is captured at B's formula evaluation
point.  It is never reconstructed from a ``RiskFrame`` or a D artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
from arctic_route_planning.contracts import CommittedRiskWindow, RiskFrame

from arctic_route_risk.errors import InputIdentityError, RiskPipelineError

_FORMULA_VERSION = "deterministic_environment_components_v2"
_SUM_ABS_TOLERANCE = 1e-6
_PIPELINE_SEAL = object()
_CONTRIBUTOR_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ice",
        "海冰",
        (
            "ice_concentration",
            "ice_thickness",
            "ice_type",
            "ice_edge",
            "ice_drift_speed",
        ),
    ),
    ("wave", "海浪", ("wave_height",)),
    ("current", "海流", ("ocean_current_speed",)),
    ("wind", "风", ("wind_speed",)),
    ("freezing", "低温", ("freezing_deficit",)),
    ("visibility", "低能见度", ("visibility_deficit",)),
    ("water_level", "水位", ("water_level_magnitude",)),
)


@dataclass(frozen=True, slots=True)
class RiskComponentTrace:
    """Exact component evidence captured during one B formula evaluation."""

    component_id: str
    normalized_value: np.ndarray
    weight: float
    contribution: np.ndarray

    def __post_init__(self) -> None:
        if not self.component_id:
            raise RiskPipelineError("risk explanation component_id cannot be empty")
        if not math.isfinite(self.weight) or not 0 < self.weight <= 1:
            raise RiskPipelineError("risk explanation component weight must be in (0, 1]")
        normalized = _frozen_array(self.normalized_value, field="normalized_value")
        contribution = _frozen_array(self.contribution, field="contribution")
        if normalized.shape != contribution.shape:
            raise RiskPipelineError("risk explanation component arrays must have one shape")
        if np.any(np.isinf(normalized)) or np.any(np.isinf(contribution)):
            raise RiskPipelineError("risk explanation component arrays cannot contain infinity")
        finite = np.isfinite(normalized)
        if np.any((normalized[finite] < 0) | (normalized[finite] > 1)):
            raise RiskPipelineError("normalized component values must be within [0, 1]")
        if not np.array_equal(finite, np.isfinite(contribution)):
            raise RiskPipelineError(
                "normalized component and contribution availability must match"
            )
        if not np.allclose(
            contribution[finite],
            normalized[finite] * self.weight,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RiskPipelineError("component contribution must equal normalized_value * weight")
        object.__setattr__(self, "normalized_value", normalized)
        object.__setattr__(self, "contribution", contribution)


@dataclass(frozen=True, slots=True)
class RiskExplanationFrameTrace:
    """Formula trace and grid identity for one generated RiskFrame."""

    risk_frame_id: str
    frame_time: datetime
    grid_id: str
    latitude: np.ndarray
    longitude: np.ndarray
    land_sea_valid: np.ndarray
    components: tuple[RiskComponentTrace, ...]

    def __post_init__(self) -> None:
        if not self.risk_frame_id or not self.grid_id:
            raise RiskPipelineError("risk explanation frame identity cannot be empty")
        frame_time = _require_utc(self.frame_time, field="frame_time")
        latitude = _frozen_coordinate(self.latitude, field="latitude")
        longitude = _frozen_coordinate(self.longitude, field="longitude")
        land_sea_valid = _frozen_boolean_array(
            self.land_sea_valid,
            field="land_sea_valid",
        )
        if not self.components:
            raise RiskPipelineError("risk explanation frame must contain component traces")
        component_ids = tuple(component.component_id for component in self.components)
        if len(component_ids) != len(set(component_ids)):
            raise RiskPipelineError("risk explanation component IDs must be unique")
        expected_shape = (latitude.size, longitude.size)
        if land_sea_valid.shape != expected_shape:
            raise RiskPipelineError("risk explanation validity shape does not match grid")
        if any(component.normalized_value.shape != expected_shape for component in self.components):
            raise RiskPipelineError("risk explanation component shape does not match grid")
        object.__setattr__(self, "frame_time", frame_time)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "land_sea_valid", land_sea_valid)


@dataclass(frozen=True, slots=True)
class RiskExplanationTraceWindow:
    """B-owned trace identity produced alongside one uncommitted frame sequence."""

    run_id: str
    scenario_id: str
    corridor_id: str
    vessel_profile_id: str
    config_digest: str
    model_config_digest: str
    generation_id: int
    as_of_time: datetime
    formula_version: str
    calibration_status: str
    formula_component_ids: tuple[str, ...]
    frames: tuple[RiskExplanationFrameTrace, ...]

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "scenario_id",
            "corridor_id",
            "vessel_profile_id",
            "config_digest",
            "model_config_digest",
            "formula_version",
            "calibration_status",
        ):
            if not getattr(self, name):
                raise RiskPipelineError(f"risk explanation trace {name} cannot be empty")
        if (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 0
        ):
            raise RiskPipelineError("risk explanation generation_id must be non-negative")
        as_of = _require_utc(self.as_of_time, field="as_of_time")
        if not self.formula_component_ids:
            raise RiskPipelineError("formula_component_ids cannot be empty")
        if len(self.formula_component_ids) != len(set(self.formula_component_ids)):
            raise RiskPipelineError("formula_component_ids must be unique")
        if not self.frames:
            raise RiskPipelineError("risk explanation trace window cannot be empty")
        expected = self.formula_component_ids
        for frame in self.frames:
            actual = tuple(component.component_id for component in frame.components)
            if actual != expected:
                raise RiskPipelineError(
                    "risk explanation frame components must use canonical formula order"
                )
        object.__setattr__(self, "as_of_time", as_of)


@dataclass(frozen=True, slots=True)
class RiskBuildTraceResult:
    """Research build output: unchanged RiskFrames plus B-owned component trace."""

    frames: tuple[RiskFrame, ...]
    explanation_trace: RiskExplanationTraceWindow
    _trace_digest: str
    _pipeline_seal: object = field(repr=False, compare=False)

    @classmethod
    def _from_pipeline(
        cls,
        *,
        frames: tuple[RiskFrame, ...],
        explanation_trace: RiskExplanationTraceWindow,
    ) -> RiskBuildTraceResult:
        """Create a sealed result at B's formula evaluation boundary."""

        return cls(
            frames=frames,
            explanation_trace=explanation_trace,
            _trace_digest=_risk_build_trace_digest(frames, explanation_trace),
            _pipeline_seal=_PIPELINE_SEAL,
        )

    def __post_init__(self) -> None:
        if not self.frames:
            raise RiskPipelineError("risk build trace result cannot be empty")
        frame_ids = tuple(frame.risk_id for frame in self.frames)
        trace_ids = tuple(
            frame.risk_frame_id for frame in self.explanation_trace.frames
        )
        if frame_ids != trace_ids:
            raise InputIdentityError("risk build frames and explanation trace IDs mismatch")
        self._assert_pipeline_integrity()

    def _assert_pipeline_integrity(self) -> None:
        """Recheck the sealed trace immediately before an export boundary."""

        if self._pipeline_seal is not _PIPELINE_SEAL:
            raise RiskPipelineError("risk explanation trace was not sealed by B pipeline")
        expected_digest = _risk_build_trace_digest(self.frames, self.explanation_trace)
        if not hmac.compare_digest(self._trace_digest, expected_digest):
            raise RiskPipelineError("risk explanation trace integrity mismatch")


class RiskExplanationResearchExporter:
    """Export a schema-compatible optional sidecar from exact B formula traces."""

    def __init__(self, *, utc_now: Callable[[], datetime] | None = None) -> None:
        self._utc_now = utc_now or (lambda: datetime.now(UTC))

    def export(
        self,
        *,
        committed_window: CommittedRiskWindow,
        build_result: RiskBuildTraceResult,
    ) -> dict[str, Any]:
        """Return one ``risk-explanation.v1`` JSON-ready document.

        The committed window supplies authoritative identity and risk snapshots.
        Contributor values are accepted only from a sealed B ``build_result`` and must bind
        exactly to that window before any sidecar is returned.
        """

        generated_at = _require_utc(self._utc_now(), field="generated_at")
        build_result._assert_pipeline_integrity()
        trace_window = build_result.explanation_trace
        committed_frame_ids = tuple(frame.risk_id for frame in committed_window.frames)
        build_frame_ids = tuple(frame.risk_id for frame in build_result.frames)
        if committed_frame_ids != build_frame_ids:
            raise InputIdentityError(
                "risk explanation committed window and sealed build frames mismatch"
            )
        self._validate_window_identity(committed_window, trace_window)
        frame_documents = tuple(
            self._frame_document(frame=frame, trace=trace)
            for frame, trace in zip(
                committed_window.frames,
                trace_window.frames,
                strict=True,
            )
        )
        statuses = tuple(
            cell["explanation_status"]
            for frame in frame_documents
            for cell in frame["cells"]
        )
        publication_status = _aggregate_status(statuses)
        return {
            "schema_version": "risk-explanation.v1",
            "publication_status": publication_status,
            "identity": {
                "risk_window_id": committed_window.commit_id,
                "run_id": committed_window.run_id,
                "scenario_id": committed_window.scenario_id,
                "corridor_id": committed_window.corridor_id,
                "vessel_profile_id": committed_window.vessel_profile_id,
                "config_digest": committed_window.config_digest,
                "model_config_digest": committed_window.model_config_digest,
                "generation_id": committed_window.generation_id,
                "as_of_time": _iso_z(committed_window.as_of),
            },
            "producer": {
                "producer_id": "work_package_b.risk_explanation_research_exporter.v1",
                "generated_at": _iso_z(generated_at),
                "formula_version": trace_window.formula_version,
                "formula_component_ids": list(trace_window.formula_component_ids),
                "decomposition_method": "weighted_additive_decomposition_v1",
                "calibration_status": trace_window.calibration_status,
                "source_risk_provenance": committed_window.frames[0].provenance.value,
                "sidecar_maturity": "research_unvalidated",
            },
            "frames": list(frame_documents),
        }

    def _validate_window_identity(
        self,
        window: CommittedRiskWindow,
        trace: RiskExplanationTraceWindow,
    ) -> None:
        mismatched = [
            name
            for name in (
                "run_id",
                "scenario_id",
                "corridor_id",
                "vessel_profile_id",
                "config_digest",
                "model_config_digest",
                "generation_id",
            )
            if getattr(window, name) != getattr(trace, name)
        ]
        if window.as_of != trace.as_of_time:
            mismatched.append("as_of_time")
        if len(window.frames) != len(trace.frames):
            mismatched.append("frames")
        if trace.formula_version != _FORMULA_VERSION:
            mismatched.append("formula_version")
        if trace.calibration_status != "demo_unvalidated":
            mismatched.append("calibration_status")
        expected_components = tuple(
            component_id
            for _, _, component_ids in _CONTRIBUTOR_GROUPS
            for component_id in component_ids
        )
        if trace.formula_component_ids != expected_components:
            mismatched.append("formula_component_ids")
        if mismatched:
            raise InputIdentityError(
                "risk explanation window identity mismatch: " + ", ".join(mismatched)
            )

    def _frame_document(
        self,
        *,
        frame: RiskFrame,
        trace: RiskExplanationFrameTrace,
    ) -> dict[str, Any]:
        grid = frame.grid
        latitude = np.asarray(frame.payload["latitude"].values, dtype=np.float64)
        longitude = np.asarray(frame.payload["longitude"].values, dtype=np.float64)
        mismatched = []
        if frame.risk_id != trace.risk_frame_id:
            mismatched.append("risk_frame_id")
        if frame.valid_time != trace.frame_time:
            mismatched.append("frame_time")
        if grid.grid_id != trace.grid_id:
            mismatched.append("grid_id")
        if not np.array_equal(latitude, trace.latitude):
            mismatched.append("latitude")
        if not np.array_equal(longitude, trace.longitude):
            mismatched.append("longitude")
        if frame.payload.attrs.get("risk_formula") != _FORMULA_VERSION:
            mismatched.append("risk_formula")
        if frame.payload.attrs.get("calibration_status") != "demo_unvalidated":
            mismatched.append("calibration_status")
        if mismatched:
            raise InputIdentityError(
                f"risk explanation frame identity mismatch for {frame.risk_id}: "
                + ", ".join(mismatched)
            )

        risk = np.asarray(frame.payload["risk_score"].values, dtype=np.float64)
        level = np.asarray(frame.payload["risk_level"].values)
        confidence = np.asarray(frame.payload["confidence"].values, dtype=np.float64)
        cells = tuple(
            self._cell_document(
                row=row,
                column=column,
                latitude=float(latitude[row]),
                longitude=float(longitude[column]),
                risk_score=float(risk[row, column]),
                risk_level=int(level[row, column]),
                confidence=float(confidence[row, column]),
                land_sea_valid=bool(trace.land_sea_valid[row, column]),
                components=trace.components,
            )
            for row in range(latitude.size)
            for column in range(longitude.size)
        )
        counts = {
            status: sum(cell["explanation_status"] == status for cell in cells)
            for status in ("COMPLETE", "PARTIAL", "UNAVAILABLE")
        }
        return {
            "risk_frame_id": frame.risk_id,
            "frame_time": _iso_z(frame.valid_time),
            "grid": {
                "grid_id": grid.grid_id,
                "crs": grid.crs,
                "rows": latitude.size,
                "columns": longitude.size,
            },
            "coverage": {
                "expected_cell_count": len(cells),
                "published_cell_count": len(cells),
                "complete_cell_count": counts["COMPLETE"],
                "partial_cell_count": counts["PARTIAL"],
                "unavailable_cell_count": counts["UNAVAILABLE"],
                "omitted_cell_count": 0,
            },
            "cells": list(cells),
        }

    def _cell_document(
        self,
        *,
        row: int,
        column: int,
        latitude: float,
        longitude: float,
        risk_score: float,
        risk_level: int,
        confidence: float,
        land_sea_valid: bool,
        components: tuple[RiskComponentTrace, ...],
    ) -> dict[str, Any]:
        by_id = {component.component_id: component for component in components}
        available_ids = tuple(
            component.component_id
            for component in components
            if math.isfinite(float(component.contribution[row, column]))
        )
        missing_component_ids = tuple(
            component.component_id
            for component in components
            if component.component_id not in available_ids
        )
        if not land_sea_valid:
            status = "UNAVAILABLE"
            available_ids = ()
            explanation_gap_ids = tuple(
                component.component_id for component in components
            )
        elif len(available_ids) == len(components):
            status = "COMPLETE"
            explanation_gap_ids = ()
        elif available_ids:
            status = "PARTIAL"
            explanation_gap_ids = missing_component_ids
        else:
            status = "UNAVAILABLE"
            explanation_gap_ids = missing_component_ids

        contributors: list[dict[str, Any]] = []
        for contributor_id, display_name, component_ids in _CONTRIBUTOR_GROUPS:
            published_ids = tuple(
                component_id for component_id in component_ids if component_id in available_ids
            )
            if not published_ids:
                continue
            contribution = math.fsum(
                float(by_id[component_id].contribution[row, column])
                for component_id in published_ids
            )
            contributors.append(
                {
                    "contributor_id": contributor_id,
                    "display_name": display_name,
                    "contribution": contribution,
                    "component_ids": list(published_ids),
                }
            )

        finite_risk = math.isfinite(risk_score)
        if not 1 <= risk_level <= 5:
            raise RiskPipelineError("risk explanation level must be within [1, 5]")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RiskPipelineError("risk explanation confidence must be within [0, 1]")
        if finite_risk and not 0 <= risk_score <= 1:
            raise RiskPipelineError("risk explanation score must be within [0, 1]")
        if not finite_risk and (risk_level != 5 or confidence != 0):
            raise RiskPipelineError(
                "unknown RiskFrame score must bind level 5 and confidence 0"
            )
        if status == "COMPLETE":
            if not finite_risk:
                raise RiskPipelineError(
                    "complete explanation cannot bind a non-finite RiskFrame score"
                )
            contribution_sum = math.fsum(
                contributor["contribution"] for contributor in contributors
            )
            if not math.isclose(
                contribution_sum,
                risk_score,
                rel_tol=0.0,
                abs_tol=_SUM_ABS_TOLERANCE,
            ):
                raise RiskPipelineError(
                    "risk explanation contribution sum does not match RiskFrame score"
                )
        elif finite_risk:
            raise RiskPipelineError(
                "partial or unavailable explanation cannot bind a finite RiskFrame score"
            )

        validity_missing_ids = () if land_sea_valid else ("land_sea_mask",)
        reason = _reason(
            status=status,
            contributors=contributors,
            missing_ids=(*validity_missing_ids, *missing_component_ids),
        )
        uncertainty_status = "NONE" if status == "COMPLETE" else "MISSING_DATA"
        return {
            "cell": {
                "row_index": row,
                "column_index": column,
                "latitude": latitude,
                "longitude": longitude,
            },
            "explanation_status": status,
            "risk": {
                "score": risk_score if finite_risk else None,
                "level": risk_level,
                "confidence": confidence,
            },
            "contributors": contributors,
            "reason": reason,
            "uncertainty": {
                "status": uncertainty_status,
                "missing_data": [
                    {
                        "data_type": component_id,
                        "cause": "NON_FINITE_INPUT",
                    }
                    for component_id in (
                        *validity_missing_ids,
                        *missing_component_ids,
                    )
                ],
                "explanation_gaps": list(explanation_gap_ids),
            },
        }


def _reason(
    *,
    status: str,
    contributors: list[dict[str, Any]],
    missing_ids: tuple[str, ...],
) -> dict[str, Any]:
    if status == "UNAVAILABLE":
        return {
            "code": "EXPLANATION_UNAVAILABLE",
            "text": "风险解释不可用：所有公式分量均缺少有效数据",
            "locale": "zh-CN",
            "main_contributor_ids": [],
        }
    if status == "PARTIAL":
        return {
            "code": "MISSING_DATA",
            "text": "仅能提供部分风险贡献；缺少：" + "、".join(missing_ids),
            "locale": "zh-CN",
            "main_contributor_ids": [],
        }
    maximum = max(float(item["contribution"]) for item in contributors)
    if math.isclose(maximum, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return {
            "code": "MULTIPLE_CONTRIBUTORS",
            "text": "所有已验证风险贡献均为 0，无主要贡献项",
            "locale": "zh-CN",
            "main_contributor_ids": [],
        }
    main = [
        str(item["contributor_id"])
        for item in contributors
        if math.isclose(
            float(item["contribution"]), maximum, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    names = [
        str(item["display_name"])
        for item in contributors
        if item["contributor_id"] in main
    ]
    multiple = len(main) > 1
    return {
        "code": "MULTIPLE_CONTRIBUTORS" if multiple else "DOMINANT_CONTRIBUTOR",
        "text": ("主要风险贡献并列来自" if multiple else "主要风险贡献来自")
        + "、".join(names),
        "locale": "zh-CN",
        "main_contributor_ids": main,
    }


def _aggregate_status(statuses: tuple[str, ...]) -> str:
    if statuses and all(status == "COMPLETE" for status in statuses):
        return "COMPLETE"
    if statuses and all(status == "UNAVAILABLE" for status in statuses):
        return "UNAVAILABLE"
    return "PARTIAL"


def _frozen_array(value: np.ndarray, *, field: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2:
        raise RiskPipelineError(f"risk explanation {field} must be two-dimensional")
    array.setflags(write=False)
    return array


def _frozen_coordinate(value: np.ndarray, *, field: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if (
        array.ndim != 1
        or array.size < 1
        or not np.all(np.isfinite(array))
        or (array.size > 1 and not np.all(np.diff(array) > 0))
    ):
        raise RiskPipelineError(f"risk explanation {field} coordinate is invalid")
    array.setflags(write=False)
    return array


def _frozen_boolean_array(value: np.ndarray, *, field: str) -> np.ndarray:
    source = np.asarray(value)
    if source.ndim != 2 or source.dtype != np.bool_:
        raise RiskPipelineError(f"risk explanation {field} must be a boolean matrix")
    array = np.array(source, dtype=np.bool_, copy=True)
    array.setflags(write=False)
    return array


def _risk_build_trace_digest(
    frames: tuple[RiskFrame, ...],
    trace: RiskExplanationTraceWindow,
) -> str:
    hasher = hashlib.sha256()

    def update_text(value: object) -> None:
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\0")

    update_text("b.risk-explanation-build-trace.v1")
    for value in (
        trace.run_id,
        trace.scenario_id,
        trace.corridor_id,
        trace.vessel_profile_id,
        trace.config_digest,
        trace.model_config_digest,
        trace.generation_id,
        _iso_z(trace.as_of_time),
        trace.formula_version,
        trace.calibration_status,
        *trace.formula_component_ids,
    ):
        update_text(value)
    for frame in frames:
        update_text(frame.risk_id)
    for frame in trace.frames:
        for value in (
            frame.risk_frame_id,
            _iso_z(frame.frame_time),
            frame.grid_id,
        ):
            update_text(value)
        hasher.update(np.asarray(frame.latitude, dtype="<f8").tobytes(order="C"))
        hasher.update(np.asarray(frame.longitude, dtype="<f8").tobytes(order="C"))
        hasher.update(np.asarray(frame.land_sea_valid, dtype=np.uint8).tobytes(order="C"))
        for component in frame.components:
            update_text(component.component_id)
            update_text(component.weight.hex())
            hasher.update(
                np.asarray(component.normalized_value, dtype="<f8").tobytes(order="C")
            )
            hasher.update(
                np.asarray(component.contribution, dtype="<f8").tobytes(order="C")
            )
    return hasher.hexdigest()


def _require_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RiskPipelineError(f"{field} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise RiskPipelineError(f"{field} must use UTC")
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
