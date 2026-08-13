"""Versioned B-only temporal, target-grid, and demo risk policy."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from arctic_route_contracts import canonical_sha256

from arctic_route_risk.errors import GridCompatibilityError, RiskPipelineError


@dataclass(frozen=True, slots=True)
class TargetGridConfig:
    schema_version: str = "b.target-grid-policy.v1"
    crs: str = "EPSG:4326"
    latitude_step_degrees: float = 1.0
    longitude_step_degrees: float = 1.0
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
class DemoRiskModelConfig:
    """Uncalibrated deterministic baseline; values are engineering placeholders."""

    schema_version: str = "b.demo-risk-model-config.v1"
    model_version: str = "demo_unvalidated_rule_baseline.v1"
    calibration_status: str = "demo_unvalidated"
    interval_minutes: int = 60
    temporal_policy_version: str = "visible_supports_hourly_v1"
    formula_version: str = "deterministic_environment_components_v1"
    risk_level_policy: str = "c_equal_width_floor_v1"
    hard_mask_policy: str = "land_sea_mask_land_or_coast_only_v1"
    unknown_policy: str = "nan_confidence_zero_v1"
    minimum_speed_factor: float = 0.35

    def __post_init__(self) -> None:
        expected_policies = {
            "schema_version": "b.demo-risk-model-config.v1",
            "model_version": "demo_unvalidated_rule_baseline.v1",
            "temporal_policy_version": "visible_supports_hourly_v1",
            "formula_version": "deterministic_environment_components_v1",
            "risk_level_policy": "c_equal_width_floor_v1",
            "hard_mask_policy": "land_sea_mask_land_or_coast_only_v1",
            "unknown_policy": "nan_confidence_zero_v1",
        }
        if any(getattr(self, name) != value for name, value in expected_policies.items()):
            raise RiskPipelineError("unsupported demo risk model policy")
        if self.calibration_status != "demo_unvalidated":
            raise RiskPipelineError("demo baseline cannot claim calibration")
        if self.interval_minutes != 60:
            raise RiskPipelineError("formal MVP interval_minutes must be 60")
        if not 0 < self.minimum_speed_factor <= 1:
            raise RiskPipelineError("minimum_speed_factor must be within (0, 1]")


@dataclass(frozen=True, slots=True)
class RiskBuildConfiguration:
    """One strictly parsed, versioned B grid and model policy document."""

    schema_version: str
    grid_config: TargetGridConfig
    model_config: DemoRiskModelConfig

    def __post_init__(self) -> None:
        if self.schema_version != "b.risk-build-configuration.v1":
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
    try:
        grid = TargetGridConfig(**grid_document)
        model = DemoRiskModelConfig(**model_document)
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
            "schema_version": "b.model-config.v1",
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
