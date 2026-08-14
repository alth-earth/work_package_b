"""Deterministic, hourly, demo-unvalidated A-to-RiskFrame build service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import numpy as np
import xarray as xr
from arctic_route_data import StandardDataFrame
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


class RiskBuildService:
    """Build a complete full-RunContext window without changing A payloads."""

    def __init__(self, *, utc_now: Callable[[], datetime] | None = None) -> None:
        self._utc_now = utc_now or (lambda: datetime.now(UTC))

    def build_window(self, request: RiskBuildRequest) -> tuple[RiskFrame, ...]:
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
        frames = tuple(
            self._build_frame(
                request=request,
                valid_time=envelope.requested_start + index * interval,
                generated_at=generated_at,
                latitude=latitude,
                longitude=longitude,
                grid_id=grid_id,
            )
            for index in range(count)
        )
        if not frames or frames[-1].valid_time != envelope.requested_end:
            raise CoverageError("forecast_coverage_insufficient: output window is incomplete")
        return frames

    def _build_frame(
        self,
        *,
        request: RiskBuildRequest,
        valid_time: datetime,
        generated_at: datetime,
        latitude: np.ndarray,
        longitude: np.ndarray,
        grid_id: str,
    ) -> RiskFrame:
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
        risk, hard, confidence, speed_factor = _demo_unvalidated_risk(
            arrays,
            source_confidence=min(field.confidence for field in resolved.values()),
            model_config=request.model_config,
        )
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
        return replace(provisional, risk_id=with_canonical_risk_id(provisional))


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    components = tuple(
        (
            component.weight,
            _risk_component(values, component),
        )
        for component in model_config.components
    )
    risk = sum(weight * component for weight, component in components)
    valid = np.logical_and.reduce([np.isfinite(component) for _, component in components])
    land_sea = values["land_sea_mask"]
    hard = ~np.isfinite(land_sea) | (
        land_sea < model_config.land_sea_mask_land_threshold
    )
    valid &= np.isfinite(land_sea)
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
    return risk, hard, confidence, speed


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
