"""Deterministic, hourly, demo-unvalidated A-to-RiskFrame build service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import xarray as xr
from arctic_route_data import StandardDataFrame
from arctic_route_data.derivations import (
    ICE_EDGE_CONCENTRATION_THRESHOLD as ICE_FREE_CONCENTRATION_THRESHOLD,
)
from arctic_route_planning.contracts import (
    ProvenanceKind,
    RiskFrame,
    SourceReference,
)

from arctic_route_risk.bc_codec import with_canonical_risk_id
from arctic_route_risk.config import (
    DemoRiskModelConfig,
    RiskComponentConfig,
    TargetGridConfig,
    model_config_digest,
)
from arctic_route_risk.context import REQUIRED_FORMAL_DATA_TYPES, BInputEnvelope
from arctic_route_risk.errors import CoverageError, GridCompatibilityError, RiskPipelineError
from arctic_route_risk.risk_explanation import (
    RiskBuildTraceResult,
    RiskComponentTrace,
    RiskExplanationFrameTrace,
    RiskExplanationTraceWindow,
)

_VARIABLES: dict[str, tuple[str, ...]] = {
    "land_sea_mask": ("land_sea_mask",),
    "ocean_current": ("ocean_current_u", "ocean_current_v"),
    "sea_ice_concentration": ("ice_concentration",),
    "sea_ice_drift": ("ice_drift_u", "ice_drift_v"),
    "sea_ice_edge": ("ice_edge",),
    "sea_ice_thickness": ("ice_thickness",),
    "sea_ice_type": ("ice_type",),
    "temperature": ("air_temperature_2m",),
    "visibility": ("visibility",),
    "water_level": ("sea_surface_height",),
    "wave": ("significant_wave_height",),
    "wind_field": ("wind_u10", "wind_v10"),
}
_CATEGORICAL = frozenset({"sea_ice_type", "sea_ice_edge"})
_STATIC = frozenset({"land_sea_mask"})
_SPATIAL_NEAREST = _CATEGORICAL | _STATIC
_COMPONENT_INPUTS: dict[str, tuple[str, ...]] = {
    "ice_concentration": ("ice_concentration",),
    "ice_thickness": ("ice_thickness",),
    "ice_type": ("ice_type",),
    "ice_edge": ("ice_edge",),
    "ice_drift_speed": ("ice_drift_u", "ice_drift_v"),
    "wave_height": ("significant_wave_height",),
    "ocean_current_speed": ("ocean_current_u", "ocean_current_v"),
    "wind_speed": ("wind_u10", "wind_v10"),
    "freezing_deficit": ("air_temperature_2m",),
    "visibility_deficit": ("visibility",),
    "water_level_magnitude": ("sea_surface_height",),
}


@dataclass(frozen=True, slots=True)
class RiskBuildRequest:
    envelope: BInputEnvelope
    target_bbox: tuple[float, float, float, float]
    grid_config: TargetGridConfig = field(default_factory=TargetGridConfig)
    model_config: DemoRiskModelConfig = field(default_factory=DemoRiskModelConfig)

    def __post_init__(self) -> None:
        if len(self.target_bbox) != 4:
            raise GridCompatibilityError("grid_incompatible: target_bbox must have four values")
        if self.model_config.interval_minutes != 60:
            raise RiskPipelineError("formal B output interval must be 60 minutes")

    @property
    def model_config_digest(self) -> str:
        return model_config_digest(grid=self.grid_config, model=self.model_config)


@dataclass(frozen=True, slots=True)
class _ResolvedField:
    variables: dict[str, np.ndarray]
    support_frames: tuple[StandardDataFrame, ...]
    confidence: float


@dataclass(frozen=True, slots=True)
class _RiskEvaluation:
    risk: np.ndarray
    hard: np.ndarray
    confidence: np.ndarray
    speed_factor: np.ndarray
    hard_reason: np.ndarray
    land_sea_valid: np.ndarray
    normalized_components: tuple[tuple[RiskComponentConfig, np.ndarray], ...]


@dataclass(frozen=True, slots=True)
class _BuiltFrame:
    frame: RiskFrame
    land_sea_valid: np.ndarray
    components: tuple[RiskComponentTrace, ...]


class RiskBuildService:
    """Build a complete full-RunContext window without changing A payloads."""

    def __init__(self, *, utc_now: Callable[[], datetime] | None = None) -> None:
        self._utc_now = utc_now or (lambda: datetime.now(UTC))

    def build_window(self, request: RiskBuildRequest) -> tuple[RiskFrame, ...]:
        """Build the unchanged formal RiskFrame sequence."""

        frames, trace = self._build_window(request, capture_explanation=False)
        assert trace is None
        return frames

    def build_window_with_explanation_trace(
        self, request: RiskBuildRequest
    ) -> RiskBuildTraceResult:
        """Research-only build that also captures exact formula component evidence."""

        frames, trace = self._build_window(request, capture_explanation=True)
        assert trace is not None
        return RiskBuildTraceResult._from_pipeline(
            frames=frames,
            explanation_trace=trace,
        )

    def _build_window(
        self,
        request: RiskBuildRequest,
        *,
        capture_explanation: bool,
    ) -> tuple[tuple[RiskFrame, ...], RiskExplanationTraceWindow | None]:
        # Recheck A's semantic payload attestations immediately before reading
        # values, then work only from a fresh deep snapshot. This closes the
        # validation-to-build aliasing gap of inspectable xarray containers.
        request = replace(request, envelope=request.envelope.verified_build_snapshot())
        envelope = request.envelope
        generated_at = _ensure_utc(self._utc_now(), field="generated_at")
        latitude, longitude, grid_id = request.grid_config.realize(request.target_bbox)
        interval = timedelta(minutes=request.model_config.interval_minutes)
        duration = envelope.requested_end - envelope.requested_start
        if duration.total_seconds() % interval.total_seconds():
            raise CoverageError(
                "forecast_coverage_insufficient: RunContext window is not hourly aligned"
            )
        count = int(duration / interval) + 1
        built = tuple(
            self._build_frame(
                request=request,
                valid_time=envelope.requested_start + index * interval,
                generated_at=generated_at,
                latitude=latitude,
                longitude=longitude,
                grid_id=grid_id,
                capture_explanation=capture_explanation,
            )
            for index in range(count)
        )
        frames = tuple(item.frame for item in built)
        if not frames or frames[-1].valid_time != envelope.requested_end:
            raise CoverageError("forecast_coverage_insufficient: output window is incomplete")
        if not capture_explanation:
            return frames, None
        first = frames[0]
        trace = RiskExplanationTraceWindow(
            run_id=first.run_id,
            scenario_id=first.scenario_id,
            corridor_id=first.corridor_id,
            vessel_profile_id=first.vessel_profile_id,
            config_digest=first.config_digest,
            model_config_digest=first.model_config_digest,
            generation_id=first.generation_id,
            as_of_time=first.as_of_time,
            formula_version=request.model_config.formula_version,
            calibration_status=request.model_config.calibration_status,
            formula_component_ids=tuple(
                component.component_id for component in request.model_config.components
            ),
            frames=tuple(
                RiskExplanationFrameTrace(
                    risk_frame_id=item.frame.risk_id,
                    frame_time=item.frame.valid_time,
                    grid_id=item.frame.grid.grid_id,
                    latitude=np.asarray(item.frame.payload["latitude"].values),
                    longitude=np.asarray(item.frame.payload["longitude"].values),
                    land_sea_valid=item.land_sea_valid,
                    components=item.components,
                )
                for item in built
            ),
        )
        return frames, trace

    def _build_frame(
        self,
        *,
        request: RiskBuildRequest,
        valid_time: datetime,
        generated_at: datetime,
        latitude: np.ndarray,
        longitude: np.ndarray,
        grid_id: str,
        capture_explanation: bool,
    ) -> _BuiltFrame:
        envelope = request.envelope
        resolved = {
            data_type: _resolve_field(
                data_type=data_type,
                frames=envelope.frames[data_type],
                target_time=valid_time,
                knowledge_as_of=envelope.knowledge_as_of,
                latitude=latitude,
                longitude=longitude,
                target_bbox=request.target_bbox,
                model_config=request.model_config,
            )
            for data_type in sorted(REQUIRED_FORMAL_DATA_TYPES)
        }
        arrays = {
            variable: values
            for field in resolved.values()
            for variable, values in field.variables.items()
        }
        if (
            request.model_config.hard_mask_policy
            == "land_sea_mask_plus_unknown_ice_free_v1"
        ):
            arrays, ice_free_neutralized_counts = _apply_ice_free_neutral_fill(arrays)
        else:
            ice_free_neutralized_counts = {
                "ice_type": 0,
                "ice_edge": 0,
            }
        evaluation = _evaluate_demo_unvalidated_risk(
            arrays,
            source_confidence=min(field.confidence for field in resolved.values()),
            model_config=request.model_config,
        )
        risk = evaluation.risk
        hard = evaluation.hard
        confidence = evaluation.confidence
        speed_factor = evaluation.speed_factor
        reason = evaluation.hard_reason
        missing_input_variable_counts = {
            variable: int(np.count_nonzero(~np.isfinite(values)))
            for variable, values in arrays.items()
        }
        level = np.full(risk.shape, 5, dtype=np.uint8)
        finite = np.isfinite(risk)
        level[finite] = np.clip(np.floor(risk[finite] * 5) + 1, 1, 5).astype(np.uint8)
        payload = xr.Dataset(
            data_vars={
                "risk_score": (("latitude", "longitude"), risk.astype(np.float32)),
                "risk_level": (("latitude", "longitude"), level),
                "hard_mask": (("latitude", "longitude"), hard.astype(np.bool_)),
                "confidence": (("latitude", "longitude"), confidence.astype(np.float32)),
                "environment_speed_factor": (
                    ("latitude", "longitude"),
                    speed_factor.astype(np.float32),
                ),
                "hard_reason": (("latitude", "longitude"), reason.astype(np.str_)),
            },
            coords={"latitude": latitude, "longitude": longitude},
            attrs={
                "crs": "EPSG:4326",
                "grid_id": grid_id,
                "development_only": True,
                "calibration_status": "demo_unvalidated",
                "risk_formula": request.model_config.formula_version,
                "temporal_policy": request.model_config.temporal_policy_version,
                "hard_mask_policy": request.model_config.hard_mask_policy,
                "missing_input_variable_counts": missing_input_variable_counts,
                "ice_free_neutralized_input_counts": ice_free_neutralized_counts,
                "ice_free_predicate": {
                    "variable": "ice_concentration",
                    "operator": "<",
                    "threshold": ICE_FREE_CONCENTRATION_THRESHOLD,
                    "source": "arctic_route_data.derivations.ICE_EDGE_CONCENTRATION_THRESHOLD",
                    "semantics": (
                        "open water per A ice-type/edge derivation; "
                        "ice_type/ice_edge are not-applicable in open water"
                    ),
                },
                "dataset_bundle_id": envelope.dataset_bundle.bundle_id,
                "dataset_bundle_digest": envelope.dataset_bundle.bundle_digest,
            },
        )
        sources = _source_references(
            frame
            for field in resolved.values()
            for frame in field.support_frames
        )
        provisional = RiskFrame(
            schema_version="bc.risk-frame.v2",
            risk_id="risk-pending-canonical-content-id",
            run_id=envelope.run_context.run_id,
            scenario_id=envelope.run_context.scenario_id,
            corridor_id=envelope.run_context.corridor_id,
            vessel_profile_id=envelope.run_context.vessel_profile_id,
            config_digest=envelope.run_context.config_digest,
            model_config_digest=request.model_config_digest,
            generation_id=envelope.generation_id,
            valid_time=valid_time,
            as_of_time=envelope.knowledge_as_of,
            generated_at=generated_at,
            model_version=request.model_config.model_version,
            payload=payload,
            source_summary=sources,
            provenance=ProvenanceKind.FORMAL,
        )
        frame = replace(provisional, risk_id=with_canonical_risk_id(provisional))
        components = (
            tuple(
                RiskComponentTrace(
                    component_id=component.component_id,
                    normalized_value=normalized,
                    weight=component.weight,
                    contribution=component.weight * normalized,
                )
                for component, normalized in evaluation.normalized_components
            )
            if capture_explanation
            else ()
        )
        return _BuiltFrame(
            frame=frame,
            land_sea_valid=evaluation.land_sea_valid,
            components=components,
        )


def _resolve_field(
    *,
    data_type: str,
    frames: tuple[StandardDataFrame, ...],
    target_time: datetime,
    knowledge_as_of: datetime,
    latitude: np.ndarray,
    longitude: np.ndarray,
    target_bbox: tuple[float, float, float, float],
    model_config: DemoRiskModelConfig,
) -> _ResolvedField:
    if not frames:
        raise CoverageError(f"forecast_coverage_insufficient: no {data_type} frames")
    visible = tuple(frame for frame in frames if frame.record.issue_time <= knowledge_as_of)
    if not visible:
        raise CoverageError(
            f"future_information_leakage: no visible {data_type} support at knowledge cutoff"
        )
    ordered = tuple(sorted(visible, key=lambda item: (item.record.valid_time, item.record.data_id)))
    if data_type in _STATIC:
        support = _static_support(ordered, target_time)
        fraction = 0.0
        method_confidence = model_config.temporal_method_confidence.static
    elif data_type in _CATEGORICAL:
        lower, upper = _bracket(ordered, target_time, data_type)
        support = (_nearest(lower, upper, target_time),)
        fraction = 0.0
        method_confidence = (
            model_config.temporal_method_confidence.exact
            if support[0].record.valid_time == target_time
            else model_config.temporal_method_confidence.categorical_nearest
        )
    else:
        lower, upper = _bracket(ordered, target_time, data_type)
        support = (lower,) if lower is upper else (lower, upper)
        total = (upper.record.valid_time - lower.record.valid_time).total_seconds()
        fraction = (
            0.0
            if total == 0
            else (target_time - lower.record.valid_time).total_seconds() / total
        )
        method_confidence = (
            model_config.temporal_method_confidence.exact
            if len(support) == 1
            else model_config.temporal_method_confidence.linear_interpolation
        )
    for frame in support:
        if not _bbox_contains(frame.record.bbox, target_bbox):
            raise GridCompatibilityError(
                f"grid_incompatible: {frame.record.data_id} does not cover target bbox"
            )
    variables: dict[str, np.ndarray] = {}
    for variable in _VARIABLES[data_type]:
        lower_values = _regrid_variable(
            support[0],
            variable,
            latitude,
            longitude,
            categorical=data_type in _SPATIAL_NEAREST,
        )
        if len(support) == 1:
            variables[variable] = lower_values
        else:
            upper_values = _regrid_variable(
                support[1], variable, latitude, longitude, categorical=False
            )
            variables[variable] = lower_values + (upper_values - lower_values) * fraction
    quality = min(
        getattr(model_config.quality_confidence, frame.record.quality_flag.value)
        for frame in support
    )
    return _ResolvedField(
        variables=variables,
        support_frames=support,
        confidence=quality * method_confidence,
    )


def _static_support(
    frames: tuple[StandardDataFrame, ...], target: datetime
) -> tuple[StandardDataFrame, ...]:
    prior = [frame for frame in frames if frame.record.valid_time <= target]
    if prior:
        return (prior[-1],)
    if len(frames) == 1:
        return (frames[0],)
    raise CoverageError("forecast_coverage_insufficient: static layer has no applicable version")


def _bracket(
    frames: tuple[StandardDataFrame, ...], target: datetime, data_type: str
) -> tuple[StandardDataFrame, StandardDataFrame]:
    exact = [frame for frame in frames if frame.record.valid_time == target]
    if exact:
        return exact[-1], exact[-1]
    lower = [frame for frame in frames if frame.record.valid_time < target]
    upper = [frame for frame in frames if frame.record.valid_time > target]
    if not lower or not upper:
        raise CoverageError(
            f"forecast_coverage_insufficient: {data_type} lacks interpolation support at {target}"
        )
    return lower[-1], upper[0]


def _nearest(
    lower: StandardDataFrame, upper: StandardDataFrame, target: datetime
) -> StandardDataFrame:
    lower_delta = target - lower.record.valid_time
    upper_delta = upper.record.valid_time - target
    return lower if lower_delta <= upper_delta else upper


def _regrid_variable(
    frame: StandardDataFrame,
    variable: str,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    categorical: bool,
) -> np.ndarray:
    if not isinstance(frame.payload, xr.Dataset):
        raise GridCompatibilityError(
            f"grid_incompatible: {frame.record.data_id} is not an xarray Dataset"
        )
    dataset = frame.payload
    if variable not in dataset:
        raise RiskPipelineError(
            f"required_variable_missing: {frame.record.data_id} has no {variable}"
        )
    array = dataset[variable]
    rename: dict[str, str] = {}
    for canonical, aliases in (
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "lon")),
    ):
        matches = [name for name in aliases if name in array.coords or name in array.dims]
        if not matches:
            raise GridCompatibilityError(
                f"grid_incompatible: {frame.record.data_id} lacks {canonical} coordinate"
            )
        if matches[0] != canonical:
            rename[matches[0]] = canonical
    array = array.rename(rename)
    for dimension in tuple(array.dims):
        if dimension not in {"latitude", "longitude"}:
            if array.sizes[dimension] != 1:
                raise GridCompatibilityError(
                    f"grid_incompatible: {frame.record.data_id} has unsupported {dimension}"
                )
            array = array.isel({dimension: 0}, drop=True)
    if array.dims != ("latitude", "longitude"):
        try:
            array = array.transpose("latitude", "longitude")
        except ValueError as exc:
            raise GridCompatibilityError(
                f"grid_incompatible: {frame.record.data_id} is not rectilinear"
            ) from exc
    for coordinate in ("latitude", "longitude"):
        values = np.asarray(array[coordinate].values, dtype=float)
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise GridCompatibilityError(
                f"grid_incompatible: invalid {coordinate} in {frame.record.data_id}"
            )
        differences = np.diff(values)
        if np.all(differences < 0):
            array = array.sortby(coordinate)
        elif not np.all(differences > 0):
            raise GridCompatibilityError(
                f"grid_incompatible: non-monotonic {coordinate} in {frame.record.data_id}"
            )
    method = "nearest" if categorical else "linear"
    aligned = array.interp(latitude=latitude, longitude=longitude, method=method)
    values = np.asarray(aligned.values, dtype=np.float64)
    if values.shape != (latitude.size, longitude.size):
        raise GridCompatibilityError(
            f"grid_incompatible: unexpected aligned shape for {frame.record.data_id}"
        )
    return values


def _demo_unvalidated_risk(
    values: dict[str, np.ndarray],
    *,
    source_confidence: float,
    model_config: DemoRiskModelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the frozen formal outputs without exposing research trace state."""

    evaluation = _evaluate_demo_unvalidated_risk(
        values,
        source_confidence=source_confidence,
        model_config=model_config,
    )
    return (
        evaluation.risk,
        evaluation.hard,
        evaluation.confidence,
        evaluation.speed_factor,
        evaluation.hard_reason,
    )


def _evaluate_demo_unvalidated_risk(
    values: dict[str, np.ndarray],
    *,
    source_confidence: float,
    model_config: DemoRiskModelConfig,
) -> _RiskEvaluation:
    normalized_components = tuple(
        (component, _risk_component(values, component))
        for component in model_config.components
    )
    risk = sum(
        component.weight * normalized
        for component, normalized in normalized_components
    )
    valid = np.logical_and.reduce(
        [np.isfinite(normalized) for _, normalized in normalized_components]
    )
    land_sea = values["land_sea_mask"]
    land_sea_valid = np.isfinite(land_sea)
    hard = ~np.isfinite(land_sea) | (
        land_sea < model_config.land_sea_mask_land_threshold
    )
    valid &= np.isfinite(land_sea)
    if model_config.hard_mask_policy in {
        "land_sea_mask_plus_unknown_v1",
        "land_sea_mask_plus_unknown_ice_free_v1",
    }:
        # Conservative fail-closed rule: planning nodes whose risk inputs are
        # not fully finite are unavailable (hard), never treated as safe.
        hard = hard | (~valid)
    risk = np.where(valid, np.clip(risk, 0.0, 1.0), np.nan)
    confidence = np.where(valid, source_confidence, 0.0)
    speed = np.where(
        valid,
        np.clip(
            1.0 - model_config.speed_risk_coefficient * risk,
            model_config.minimum_speed_factor,
            1.0,
        ),
        model_config.minimum_speed_factor,
    )
    reason = _hard_reason_values(
        values=values,
        hard=hard,
        valid=valid,
        model_config=model_config,
    )
    return _RiskEvaluation(
        risk=risk,
        hard=hard,
        confidence=confidence,
        speed_factor=speed,
        hard_reason=reason,
        land_sea_valid=land_sea_valid,
        normalized_components=normalized_components,
    )


def _hard_reason_values(
    *,
    values: dict[str, np.ndarray],
    hard: np.ndarray,
    valid: np.ndarray,
    model_config: DemoRiskModelConfig,
) -> np.ndarray:
    """Return per-cell hard reason with documented precedence.

    Precedence: physical land first, then data-unavailable under the
    ``land_sea_mask_plus_unknown_v1`` policy, then OTHER for any remaining
    unexplained hard cell.  Non-hard cells always read ``NONE``.
    """

    land_sea = values["land_sea_mask"]
    land = ~np.isfinite(land_sea) | (
        land_sea < model_config.land_sea_mask_land_threshold
    )
    reason = np.full(hard.shape, "NONE", dtype="U32")
    reason[land] = "LAND"
    if model_config.hard_mask_policy in {
        "land_sea_mask_plus_unknown_v1",
        "land_sea_mask_plus_unknown_ice_free_v1",
    }:
        reason[(~valid) & (~land)] = "DATA_UNAVAILABLE"
    unexplained = hard & (reason == "NONE")
    reason[unexplained] = "OTHER"
    reason[~hard] = "NONE"
    return reason


def _apply_ice_free_neutral_fill(
    values: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Neutralise NEXTsim ice type/edge where a trusted input proves open water.

    ``ice_type`` and ``ice_edge`` are NOT_APPLICABLE in open water, not unknown.
    The trusted predicate is a finite, physically valid sea-ice concentration
    in ``[0, threshold)`` (project authority: A's
    ``ICE_EDGE_CONCENTRATION_THRESHOLD``, used by ``derive_sea_ice_type`` /
    ``derive_sea_ice_edge``).  NaN, negative, or above-threshold concentration
    cannot prove open water and stays fail-closed.  Only NEXTsim-NaN cells are
    neutralised to 0.0 (zero risk contribution); finite values are preserved.
    Returns updated arrays and per-variable neutralised counts for provenance.
    """

    concentration = values["ice_concentration"]
    ice_free = (
        np.isfinite(concentration)
        & (concentration >= 0.0)
        & (concentration < ICE_FREE_CONCENTRATION_THRESHOLD)
    )
    neutralized_counts: dict[str, int] = {}
    for name in ("ice_type", "ice_edge"):
        array = values[name]
        fill = ice_free & ~np.isfinite(array)
        neutralized_counts[name] = int(np.count_nonzero(fill))
        if np.any(fill):
            values[name] = np.where(fill, 0.0, array)
    return values, neutralized_counts


def _risk_component(
    values: dict[str, np.ndarray], component: RiskComponentConfig
) -> np.ndarray:
    inputs = tuple(values[name] for name in _COMPONENT_INPUTS[component.component_id])
    if component.transform == "identity":
        transformed = inputs[0]
        return _bounded(transformed, component.lower, component.upper)
    if component.transform == "absolute":
        transformed = np.abs(inputs[0])
        return _bounded(transformed, component.lower, component.upper)
    if component.transform == "vector_magnitude":
        transformed = np.hypot(inputs[0], inputs[1])
        return _bounded(transformed, component.lower, component.upper)
    if component.transform == "inverse_linear":
        return np.clip(
            (component.upper - inputs[0]) / (component.upper - component.lower),
            0.0,
            1.0,
        )
    raise RiskPipelineError(
        f"unsupported transform at runtime: {component.component_id}/{component.transform}"
    )


def _bounded(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _source_references(frames) -> tuple[SourceReference, ...]:
    unique = {frame.record.data_id: frame.record for frame in frames}
    return tuple(
        SourceReference(
            source_id=record.source,
            data_id=record.data_id,
            issue_time=record.issue_time,
            valid_time=record.valid_time,
            version=record.version,
            quality_flag=record.quality_flag.value,
            checksum=record.checksum,
        )
        for record in sorted(
            unique.values(),
            key=lambda item: (item.data_type, item.valid_time, item.data_id),
        )
    )


def _bbox_contains(
    outer: tuple[float, float, float, float], inner: tuple[float, float, float, float]
) -> bool:
    west, south, east, north = outer
    target_west, target_south, target_east, target_north = inner
    tolerance = 1e-9
    return (
        west <= target_west + tolerance
        and south <= target_south + tolerance
        and east >= target_east - tolerance
        and north >= target_north - tolerance
    )


def _ensure_utc(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RiskPipelineError(f"{field} must be timezone-aware UTC")
    utc = value.astimezone(UTC)
    if value.utcoffset() != timedelta(0):
        raise RiskPipelineError(f"{field} must use UTC")
    return utc
