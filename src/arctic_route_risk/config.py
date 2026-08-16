"""Versioned B-only temporal, target-grid, and demo risk policy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from arctic_route_contracts import canonical_sha256

from arctic_route_risk.errors import GridCompatibilityError, RiskPipelineError

_COMPONENT_POLICIES: tuple[tuple[str, str], ...] = (
    ("ice_concentration", "identity"),
    ("ice_thickness", "identity"),
    ("ice_type", "identity"),
    ("ice_edge", "identity"),
    ("ice_drift_speed", "vector_magnitude"),
    ("wave_height", "identity"),
    ("ocean_current_speed", "vector_magnitude"),
    ("wind_speed", "vector_magnitude"),
    ("freezing_deficit", "inverse_linear"),
    ("visibility_deficit", "inverse_linear"),
    ("water_level_magnitude", "absolute"),
)


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        raise RiskPipelineError(f"{field} must be a finite number")
    return float(value)


def _unit_interval(value: object, *, field: str, positive: bool = False) -> float:
    number = _finite_number(value, field=field)
    lower_valid = number > 0 if positive else number >= 0
    if not lower_valid or number > 1:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise RiskPipelineError(f"{field} must be within {interval}")
    return number


@dataclass(frozen=True, slots=True)
class TargetGridConfig:
    schema_version: str = "b.target-grid-policy.v1"
    crs: str = "EPSG:4326"
    latitude_step_degrees: float = 0.75
    longitude_step_degrees: float = 2.2
    shape_rule: str = "cover_bbox_endpoints_v1"
    continuous_method: str = "linear_v1"
    categorical_method: str = "nearest_v1"

    def __post_init__(self) -> None:
        if self.schema_version != "b.target-grid-policy.v1" or self.crs != "EPSG:4326":
            raise GridCompatibilityError("grid_incompatible: unsupported target grid policy")
        expected_policies = {
            "shape_rule": "cover_bbox_endpoints_v1",
            "continuous_method": "linear_v1",
            "categorical_method": "nearest_v1",
        }
        if any(getattr(self, name) != value for name, value in expected_policies.items()):
            raise GridCompatibilityError(
                "grid_incompatible: unsupported target grid method"
            )
        for name in ("latitude_step_degrees", "longitude_step_degrees"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or value <= 0
            ):
                raise GridCompatibilityError(f"grid_incompatible: {name} must be positive")

    def realize(
        self, bbox: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if len(bbox) != 4:
            raise GridCompatibilityError("grid_incompatible: bbox must have four values")
        west, south, east, north = (float(item) for item in bbox)
        if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
            raise GridCompatibilityError("grid_incompatible: invalid EPSG:4326 bbox")
        ny = max(2, math.ceil((north - south) / self.latitude_step_degrees) + 1)
        nx = max(2, math.ceil((east - west) / self.longitude_step_degrees) + 1)
        latitude = np.linspace(south, north, ny, dtype=np.float64)
        longitude = np.linspace(west, east, nx, dtype=np.float64)
        coordinate_digest = _coordinate_digest(latitude, longitude)
        return latitude, longitude, f"b-grid-{coordinate_digest[:24]}"


@dataclass(frozen=True, slots=True)
class RiskComponentConfig:
    """One fixed-input component in the uncalibrated rule baseline."""

    component_id: str
    weight: float
    transform: str
    lower: float
    upper: float

    def __post_init__(self) -> None:
        expected = dict(_COMPONENT_POLICIES).get(self.component_id)
        if expected is None:
            raise RiskPipelineError(f"unsupported risk component: {self.component_id}")
        if self.transform != expected:
            raise RiskPipelineError(
                f"unsupported transform for {self.component_id}: {self.transform}"
            )
        weight = _unit_interval(
            self.weight,
            field=f"risk component {self.component_id} weight",
            positive=True,
        )
        lower = _finite_number(
            self.lower,
            field=f"risk component {self.component_id} lower",
        )
        upper = _finite_number(
            self.upper,
            field=f"risk component {self.component_id} upper",
        )
        if lower >= upper:
            raise RiskPipelineError(
                f"risk component {self.component_id} requires lower < upper"
            )
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True, slots=True)
class QualityConfidenceConfig:
    """Confidence factors for A's three formal quality flags."""

    good: float = 1.0
    suspect: float = 0.75
    degraded: float = 0.5

    def __post_init__(self) -> None:
        values = {
            name: _unit_interval(getattr(self, name), field=f"quality_confidence.{name}")
            for name in ("good", "suspect", "degraded")
        }
        if not values["good"] >= values["suspect"] >= values["degraded"]:
            raise RiskPipelineError(
                "quality confidence must be monotonic: good >= suspect >= degraded"
            )
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TemporalMethodConfidenceConfig:
    """Confidence factors for the deterministic temporal support methods."""

    exact: float = 1.0
    categorical_nearest: float = 0.85
    linear_interpolation: float = 0.9
    static: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "exact",
            "categorical_nearest",
            "linear_interpolation",
            "static",
        ):
            object.__setattr__(
                self,
                name,
                _unit_interval(
                    getattr(self, name),
                    field=f"temporal_method_confidence.{name}",
                ),
            )


def _default_components() -> tuple[RiskComponentConfig, ...]:
    return (
        RiskComponentConfig("ice_concentration", 0.24, "identity", 0.0, 1.0),
        RiskComponentConfig("ice_thickness", 0.14, "identity", 0.0, 3.0),
        RiskComponentConfig("ice_type", 0.05, "identity", 0.0, 4.0),
        RiskComponentConfig("ice_edge", 0.02, "identity", 0.0, 1.0),
        RiskComponentConfig("ice_drift_speed", 0.06, "vector_magnitude", 0.0, 1.5),
        RiskComponentConfig("wave_height", 0.13, "identity", 0.0, 8.0),
        RiskComponentConfig(
            "ocean_current_speed", 0.07, "vector_magnitude", 0.0, 2.0
        ),
        RiskComponentConfig("wind_speed", 0.10, "vector_magnitude", 0.0, 30.0),
        RiskComponentConfig("freezing_deficit", 0.05, "inverse_linear", 243.15, 273.15),
        RiskComponentConfig("visibility_deficit", 0.10, "inverse_linear", 0.0, 10_000.0),
        RiskComponentConfig("water_level_magnitude", 0.04, "absolute", 0.0, 3.0),
    )


@dataclass(frozen=True, slots=True)
class DemoRiskModelConfig:
    """Uncalibrated deterministic baseline; values are engineering placeholders."""

    schema_version: str = "b.demo-risk-model-config.v2"
    model_version: str = "demo_unvalidated_rule_baseline.v2"
    calibration_status: str = "demo_unvalidated"
    interval_minutes: int = 60
    temporal_policy_version: str = "visible_supports_hourly_v2"
    formula_version: str = "deterministic_environment_components_v2"
    risk_level_policy: str = "c_equal_width_floor_v1"
    hard_mask_policy: str = "land_sea_mask_threshold_v2"
    unknown_policy: str = "nan_confidence_zero_v1"
    components: tuple[RiskComponentConfig, ...] = field(default_factory=_default_components)
    quality_confidence: QualityConfidenceConfig = field(
        default_factory=QualityConfidenceConfig
    )
    temporal_method_confidence: TemporalMethodConfidenceConfig = field(
        default_factory=TemporalMethodConfidenceConfig
    )
    land_sea_mask_land_threshold: float = 0.5
    speed_risk_coefficient: float = 0.55
    minimum_speed_factor: float = 0.35

    def __post_init__(self) -> None:
        expected_policies = {
            "schema_version": "b.demo-risk-model-config.v2",
            "model_version": "demo_unvalidated_rule_baseline.v2",
            "temporal_policy_version": "visible_supports_hourly_v2",
            "formula_version": "deterministic_environment_components_v2",
            "risk_level_policy": "c_equal_width_floor_v1",
            "unknown_policy": "nan_confidence_zero_v1",
        }
        if any(getattr(self, name) != value for name, value in expected_policies.items()):
            raise RiskPipelineError("unsupported demo risk model policy")
        if self.hard_mask_policy not in {
            "land_sea_mask_threshold_v2",
            "land_sea_mask_plus_unknown_v1",
        }:
            raise RiskPipelineError("unsupported hard_mask_policy")
        if self.calibration_status != "demo_unvalidated":
            raise RiskPipelineError("demo baseline cannot claim calibration")
        if isinstance(self.interval_minutes, bool) or self.interval_minutes != 60:
            raise RiskPipelineError("formal MVP interval_minutes must be 60")
        if not isinstance(self.components, tuple):
            raise RiskPipelineError("risk components must be an immutable tuple")
        component_ids = tuple(component.component_id for component in self.components)
        expected_ids = tuple(component_id for component_id, _ in _COMPONENT_POLICIES)
        if component_ids != expected_ids:
            raise RiskPipelineError(
                "risk components must contain all 11 component IDs in canonical order"
            )
        if not math.isclose(
            math.fsum(component.weight for component in self.components),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RiskPipelineError("risk component weights must sum to 1")
        if not isinstance(self.quality_confidence, QualityConfidenceConfig):
            raise RiskPipelineError("quality_confidence has invalid type")
        if not isinstance(
            self.temporal_method_confidence,
            TemporalMethodConfidenceConfig,
        ):
            raise RiskPipelineError("temporal_method_confidence has invalid type")
        land_threshold = _unit_interval(
            self.land_sea_mask_land_threshold,
            field="land_sea_mask_land_threshold",
        )
        speed_coefficient = _unit_interval(
            self.speed_risk_coefficient,
            field="speed_risk_coefficient",
            positive=True,
        )
        minimum_speed_factor = _unit_interval(
            self.minimum_speed_factor,
            field="minimum_speed_factor",
            positive=True,
        )
        object.__setattr__(self, "land_sea_mask_land_threshold", land_threshold)
        object.__setattr__(self, "speed_risk_coefficient", speed_coefficient)
        object.__setattr__(self, "minimum_speed_factor", minimum_speed_factor)


@dataclass(frozen=True, slots=True)
class RiskBuildConfiguration:
    """One strictly parsed, versioned B grid and model policy document."""

    schema_version: str
    grid_config: TargetGridConfig
    model_config: DemoRiskModelConfig

    def __post_init__(self) -> None:
        if self.schema_version != "b.risk-build-configuration.v2":
            raise RiskPipelineError("unsupported B risk build configuration version")

    @property
    def model_config_digest(self) -> str:
        return model_config_digest(grid=self.grid_config, model=self.model_config)


def load_risk_build_configuration(path: str | Path) -> RiskBuildConfiguration:
    """Load a strict JSON policy; unknown or missing fields fail closed."""

    location = Path(path)
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RiskPipelineError(f"cannot load B risk configuration: {location}") from exc
    if not isinstance(document, dict):
        raise RiskPipelineError("B risk configuration must be a JSON object")
    _require_exact_keys(
        document,
        {"schema_version", "grid_config", "model_config"},
        field="risk build configuration",
    )
    grid_document = _require_object(document["grid_config"], field="grid_config")
    model_document = _require_object(document["model_config"], field="model_config")
    _require_exact_keys(
        grid_document,
        set(TargetGridConfig.__dataclass_fields__),
        field="grid_config",
    )
    _require_exact_keys(
        model_document,
        set(DemoRiskModelConfig.__dataclass_fields__),
        field="model_config",
    )
    components_document = model_document["components"]
    if not isinstance(components_document, list):
        raise RiskPipelineError("model_config.components must be a JSON array")
    components: list[RiskComponentConfig] = []
    for index, value in enumerate(components_document):
        component_document = _require_object(
            value,
            field=f"model_config.components[{index}]",
        )
        _require_exact_keys(
            component_document,
            set(RiskComponentConfig.__dataclass_fields__),
            field=f"model_config.components[{index}]",
        )
        try:
            components.append(RiskComponentConfig(**component_document))
        except TypeError as exc:
            raise RiskPipelineError(
                f"model_config.components[{index}] field types are invalid"
            ) from exc
    quality_document = _require_object(
        model_document["quality_confidence"],
        field="model_config.quality_confidence",
    )
    _require_exact_keys(
        quality_document,
        set(QualityConfidenceConfig.__dataclass_fields__),
        field="model_config.quality_confidence",
    )
    temporal_confidence_document = _require_object(
        model_document["temporal_method_confidence"],
        field="model_config.temporal_method_confidence",
    )
    _require_exact_keys(
        temporal_confidence_document,
        set(TemporalMethodConfidenceConfig.__dataclass_fields__),
        field="model_config.temporal_method_confidence",
    )
    parsed_model_document = dict(model_document)
    parsed_model_document["components"] = tuple(components)
    try:
        grid = TargetGridConfig(**grid_document)
        parsed_model_document["quality_confidence"] = QualityConfidenceConfig(
            **quality_document
        )
        parsed_model_document["temporal_method_confidence"] = (
            TemporalMethodConfidenceConfig(**temporal_confidence_document)
        )
        model = DemoRiskModelConfig(**parsed_model_document)
    except TypeError as exc:
        raise RiskPipelineError("B risk configuration field types are invalid") from exc
    return RiskBuildConfiguration(
        schema_version=document["schema_version"],
        grid_config=grid,
        model_config=model,
    )


def model_config_digest(
    *, grid: TargetGridConfig, model: DemoRiskModelConfig
) -> str:
    """Hash policies, never a corridor-specific realized bbox or coordinates."""

    return canonical_sha256(
        {
            "schema_version": "b.model-config.v2",
            "grid_policy": grid,
            "model_policy": model,
        }
    )


def _coordinate_digest(latitude: np.ndarray, longitude: np.ndarray) -> str:
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(np.asarray(latitude, dtype="<f8").tobytes(order="C"))
    hasher.update(np.asarray(longitude, dtype="<f8").tobytes(order="C"))
    return hasher.hexdigest()


def _require_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RiskPipelineError(f"{field} must be a JSON object")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise RiskPipelineError(
            f"{field} fields differ: missing={missing}, extra={extra}"
        )
