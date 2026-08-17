"""Backend-neutral contracts for optional B model adapters.

The formal B service deliberately does not import this module. These types are
an opt-in compatibility surface for model intake and short single-step tests;
they do not change the ``RiskFrame`` production path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.errors import RiskPipelineError

MODEL_RUNTIME_UNAVAILABLE = "model_runtime_unavailable"
MODEL_ARTIFACT_INVALID = "model_artifact_invalid"
MODEL_INPUT_INCOMPATIBLE = "model_input_incompatible"
MODEL_OUTPUT_INVALID = "model_output_invalid"


def model_error(prefix: str, message: str) -> RiskPipelineError:
    return RiskPipelineError(f"{prefix}: {message}")


def _copy_array(value: Any, *, field: str, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise model_error(MODEL_INPUT_INCOMPATIBLE, f"{field} is not an array") from exc
    array.setflags(write=False)
    return array


def _copy_mapping(values: Mapping[str, Any], *, field: str) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or not values:
        raise model_error(MODEL_INPUT_INCOMPATIBLE, f"{field} must be a non-empty mapping")
    copied: dict[str, np.ndarray] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise model_error(MODEL_INPUT_INCOMPATIBLE, f"{field} has an invalid variable name")
        copied[key] = _copy_array(value, field=f"{field}.{key}")
    return MappingProxyType(copied)


def _validate_coordinate(values: np.ndarray, *, field: str) -> None:
    if values.ndim != 1 or values.size < 2 or not np.issubdtype(values.dtype, np.number):
        raise model_error(MODEL_INPUT_INCOMPATIBLE, f"{field} must be a numeric 1-D coordinate")
    if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0):
        raise model_error(
            MODEL_INPUT_INCOMPATIBLE,
            f"{field} must be finite and strictly increasing",
        )


@dataclass(frozen=True, slots=True)
class ModelArtifactPolicy:
    """Strict, versioned intake policy for the delivered legacy checkpoint."""

    policy_id: str = "legacy_cnn_one_step_v1"
    policy_version: str = "b.model-artifact-policy.v1"
    source_zip_sha256: str = "a2ee74fd70cb0735695d9cf25ae8907a7c6d7aab866b7549d8142e412190ad79"
    source_checkpoint_sha256: str = (
        "0390fac17de1f082652fc5851b6979fd771d98ff803880c6562c54921e081666"
    )
    member_path: str = "22_深度学习综合风险预测模型/downloads/model/comprehensive_risk_cnn.pth"
    native_grid_step_degrees: float = 0.05
    expected_outer_keys: tuple[str, ...] = (
        "hidden_channels",
        "in_channels",
        "model_state_dict",
        "target",
    )
    expected_state_keys: tuple[str, ...] = (
        "net.0.bias",
        "net.0.weight",
        "net.2.bias",
        "net.2.weight",
        "net.4.bias",
        "net.4.weight",
        "net.6.bias",
        "net.6.weight",
    )
    expected_state_shapes: Mapping[str, tuple[int, ...]] = field(
        default_factory=lambda: MappingProxyType(
            {
                "net.0.weight": (32, 1, 3, 3),
                "net.0.bias": (32,),
                "net.2.weight": (32, 32, 3, 3),
                "net.2.bias": (32,),
                "net.4.weight": (32, 32, 3, 3),
                "net.4.bias": (32,),
                "net.6.weight": (1, 32, 1, 1),
                "net.6.bias": (1,),
            }
        )
    )
    expected_target: str = "next_time_comprehensive_risk"

    def __post_init__(self) -> None:
        import math

        for field_name in ("source_zip_sha256", "source_checkpoint_sha256"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise model_error(MODEL_ARTIFACT_INVALID, f"{field_name} must be lowercase SHA-256")
        if self.policy_version != "b.model-artifact-policy.v1":
            raise model_error(MODEL_ARTIFACT_INVALID, "unsupported artifact policy version")
        if not self.member_path or "\\" in self.member_path or ".." in self.member_path.split("/"):
            raise model_error(MODEL_ARTIFACT_INVALID, "member_path is not a safe normalized path")
        if (
            not isinstance(self.native_grid_step_degrees, int | float)
            or isinstance(self.native_grid_step_degrees, bool)
            or not math.isclose(float(self.native_grid_step_degrees), 0.05, abs_tol=1e-12)
        ):
            raise model_error(MODEL_ARTIFACT_INVALID, "native grid policy must be 0.05 degrees")
        if tuple(sorted(self.expected_outer_keys)) != self.expected_outer_keys:
            raise model_error(MODEL_ARTIFACT_INVALID, "expected outer keys must be canonical")
        if tuple(sorted(self.expected_state_keys)) != self.expected_state_keys:
            raise model_error(MODEL_ARTIFACT_INVALID, "expected state keys must be canonical")
        if set(self.expected_state_shapes) != set(self.expected_state_keys):
            raise model_error(MODEL_ARTIFACT_INVALID, "state key/shape policy is incomplete")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ModelArtifactPolicy:
        if not isinstance(document, Mapping):
            raise model_error(MODEL_ARTIFACT_INVALID, "policy must be a JSON object")
        allowed = {
            "policy_id",
            "policy_version",
            "source_zip_sha256",
            "source_checkpoint_sha256",
            "member_path",
            "native_grid_step_degrees",
            "expected_outer_keys",
            "expected_state_keys",
            "expected_state_shapes",
            "expected_target",
        }
        if set(document) != allowed:
            missing = sorted(allowed - set(document))
            extra = sorted(set(document) - allowed)
            raise model_error(
                MODEL_ARTIFACT_INVALID,
                f"policy fields differ: missing={missing}, extra={extra}",
            )
        unknown = set(document) - allowed
        if unknown:
            raise model_error(MODEL_ARTIFACT_INVALID, f"unknown policy fields: {sorted(unknown)}")
        values = dict(document)
        if "expected_state_shapes" in values:
            values["expected_state_shapes"] = MappingProxyType(
                {str(key): tuple(value) for key, value in values["expected_state_shapes"].items()}
            )
        for key in ("expected_outer_keys", "expected_state_keys"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_document(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "source_zip_sha256": self.source_zip_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "member_path": self.member_path,
            "native_grid_step_degrees": self.native_grid_step_degrees,
            "expected_outer_keys": list(self.expected_outer_keys),
            "expected_state_keys": list(self.expected_state_keys),
            "expected_state_shapes": {
                key: list(value) for key, value in self.expected_state_shapes.items()
            },
            "expected_target": self.expected_target,
        }


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Auditable descriptor for the converted, pickle-free model asset."""

    manifest_version: str
    artifact_id: str
    policy_id: str
    backend_id: str
    model_version: str
    source_zip_sha256: str
    source_checkpoint_sha256: str
    source_member: str
    safetensors_sha256: str
    tensor_keys: tuple[str, ...]
    tensor_shapes: Mapping[str, tuple[int, ...]]
    dtype: str
    parameter_count: int
    native_grid_step_degrees: float
    predicted_valid_time: None
    time_step_status: str
    calibration_status: str
    runtime: str
    upstream_license: str
    distribution_authorized_on: str
    conversion: str = "torch.weights_only_cpu_to_safetensors_v1"

    @property
    def artifact_sha256(self) -> str:
        return self.safetensors_sha256

    def to_document(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "artifact_id": self.artifact_id,
            "policy_id": self.policy_id,
            "backend_id": self.backend_id,
            "model_version": self.model_version,
            "source_zip_sha256": self.source_zip_sha256,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_member": self.source_member,
            "safetensors_sha256": self.safetensors_sha256,
            "tensor_keys": list(self.tensor_keys),
            "tensor_shapes": {key: list(value) for key, value in self.tensor_shapes.items()},
            "dtype": self.dtype,
            "parameter_count": self.parameter_count,
            "native_grid_step_degrees": self.native_grid_step_degrees,
            "predicted_valid_time": self.predicted_valid_time,
            "time_step_status": self.time_step_status,
            "calibration_status": self.calibration_status,
            "runtime": self.runtime,
            "upstream_license": self.upstream_license,
            "distribution_authorized_on": self.distribution_authorized_on,
            "conversion": self.conversion,
        }

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> ModelArtifactManifest:
        if not isinstance(document, Mapping):
            raise model_error(MODEL_ARTIFACT_INVALID, "manifest must be a JSON object")
        required = {
            "manifest_version",
            "artifact_id",
            "policy_id",
            "backend_id",
            "model_version",
            "source_zip_sha256",
            "source_checkpoint_sha256",
            "source_member",
            "safetensors_sha256",
            "tensor_keys",
            "tensor_shapes",
            "dtype",
            "parameter_count",
            "native_grid_step_degrees",
            "predicted_valid_time",
            "time_step_status",
            "calibration_status",
            "runtime",
            "upstream_license",
            "distribution_authorized_on",
            "conversion",
        }
        if set(document) != required:
            raise model_error(MODEL_ARTIFACT_INVALID, "manifest fields differ from the v1 contract")
        if document["predicted_valid_time"] is not None:
            raise model_error(MODEL_ARTIFACT_INVALID, "legacy artifact must not have a valid time")
        return cls(
            manifest_version=str(document["manifest_version"]),
            artifact_id=str(document["artifact_id"]),
            policy_id=str(document["policy_id"]),
            backend_id=str(document["backend_id"]),
            model_version=str(document["model_version"]),
            source_zip_sha256=str(document["source_zip_sha256"]),
            source_checkpoint_sha256=str(document["source_checkpoint_sha256"]),
            source_member=str(document["source_member"]),
            safetensors_sha256=str(document["safetensors_sha256"]),
            tensor_keys=tuple(document["tensor_keys"]),
            tensor_shapes={key: tuple(value) for key, value in document["tensor_shapes"].items()},
            dtype=str(document["dtype"]),
            parameter_count=int(document["parameter_count"]),
            native_grid_step_degrees=float(document["native_grid_step_degrees"]),
            predicted_valid_time=None,
            time_step_status=str(document["time_step_status"]),
            calibration_status=str(document["calibration_status"]),
            runtime=str(document["runtime"]),
            upstream_license=str(document["upstream_license"]),
            distribution_authorized_on=str(document["distribution_authorized_on"]),
            conversion=str(document["conversion"]),
        )


@dataclass(frozen=True, slots=True)
class RiskModelInput:
    """Immutable copy of a single model input grid."""

    variables: Mapping[str, Any]
    latitude: Any
    longitude: Any
    input_time: datetime | None = None
    requested_valid_time: datetime | None = None
    time_step_status: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        variables = _copy_mapping(self.variables, field="variables")
        latitude = _copy_array(self.latitude, field="latitude", dtype=np.float64)
        longitude = _copy_array(self.longitude, field="longitude", dtype=np.float64)
        _validate_coordinate(latitude, field="latitude")
        _validate_coordinate(longitude, field="longitude")
        if self.input_time is not None and not isinstance(self.input_time, datetime):
            raise model_error(MODEL_INPUT_INCOMPATIBLE, "input_time must be datetime or None")
        if self.requested_valid_time is not None and not isinstance(
            self.requested_valid_time, datetime
        ):
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, "requested_valid_time must be datetime or None"
            )
        if not isinstance(self.time_step_status, str) or not self.time_step_status:
            raise model_error(MODEL_INPUT_INCOMPATIBLE, "time_step_status must be non-empty")
        if not isinstance(self.metadata, Mapping):
            raise model_error(MODEL_INPUT_INCOMPATIBLE, "metadata must be a mapping")
        object.__setattr__(self, "variables", variables)
        object.__setattr__(self, "latitude", latitude)
        object.__setattr__(self, "longitude", longitude)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_comprehensive_risk(
        cls,
        comprehensive_risk: Any,
        latitude: Any,
        longitude: Any,
        *,
        input_time: datetime | None = None,
    ) -> RiskModelInput:
        return cls(
            variables={"comprehensive_risk": comprehensive_risk},
            latitude=latitude,
            longitude=longitude,
            input_time=input_time,
        )


@dataclass(frozen=True, slots=True)
class RiskModelOutput:
    """Output of one backend invocation; cadence is explicit and never inferred."""

    risk_score: Any
    predicted_valid_time: datetime | None
    time_step_status: str
    backend_id: str
    model_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            risk = np.array(self.risk_score, copy=True)
        except (TypeError, ValueError) as exc:
            raise model_error(MODEL_OUTPUT_INVALID, "risk_score is not numeric") from exc
        if risk.ndim != 2 or not np.issubdtype(risk.dtype, np.number):
            raise model_error(MODEL_OUTPUT_INVALID, "risk_score must be two-dimensional")
        risk.setflags(write=False)
        if not isinstance(self.time_step_status, str) or not self.time_step_status:
            raise model_error(MODEL_OUTPUT_INVALID, "time_step_status must be non-empty")
        if not isinstance(self.backend_id, str) or not self.backend_id:
            raise model_error(MODEL_OUTPUT_INVALID, "backend_id must be non-empty")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise model_error(MODEL_OUTPUT_INVALID, "model_version must be non-empty")
        if not isinstance(self.metadata, Mapping):
            raise model_error(MODEL_OUTPUT_INVALID, "metadata must be a mapping")
        object.__setattr__(self, "risk_score", risk)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class RiskModelBackend(Protocol):
    backend_id: str
    model_version: str
    required_variables: tuple[str, ...]

    def infer(self, model_input: RiskModelInput) -> RiskModelOutput:
        """Run one model step without creating a time window or publishing frames."""


class RuleBaselineBackend:
    """Opt-in parity adapter around B's existing deterministic rule function."""

    backend_id = "rule_baseline"
    required_variables = (
        "ice_concentration",
        "ice_thickness",
        "ice_type",
        "ice_edge",
        "ice_drift_u",
        "ice_drift_v",
        "significant_wave_height",
        "ocean_current_u",
        "ocean_current_v",
        "wind_u10",
        "wind_v10",
        "air_temperature_2m",
        "visibility",
        "sea_surface_height",
        "land_sea_mask",
    )

    def __init__(self, model_config: DemoRiskModelConfig | None = None) -> None:
        self.model_config = model_config or DemoRiskModelConfig()
        self.model_version = self.model_config.model_version

    def infer(self, model_input: RiskModelInput) -> RiskModelOutput:
        from arctic_route_risk.service import _demo_unvalidated_risk

        missing = set(self.required_variables) - set(model_input.variables)
        if missing:
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, f"rule baseline variables missing: {sorted(missing)}"
            )
        values = {
            key: np.asarray(model_input.variables[key], dtype=np.float64)
            for key in self.required_variables
        }
        shapes = {value.shape for value in values.values()}
        expected = (model_input.latitude.size, model_input.longitude.size)
        if len(shapes) != 1 or next(iter(shapes)) != expected:
            raise model_error(
                MODEL_INPUT_INCOMPATIBLE, "rule baseline variables do not share the input grid"
            )
        risk, _, _, _, _ = _demo_unvalidated_risk(
            values,
            source_confidence=1.0,
            model_config=self.model_config,
        )
        return RiskModelOutput(
            risk_score=risk,
            predicted_valid_time=model_input.requested_valid_time,
            time_step_status="hourly_compatible",
            backend_id=self.backend_id,
            model_version=self.model_version,
            metadata={"formal_path": False, "parity_reference": "_demo_unvalidated_risk"},
        )

    def assert_risk_score_equivalent(
        self,
        model_input: RiskModelInput,
        expected: Any,
        *,
        equal_nan: bool = True,
    ) -> None:
        actual = self.infer(model_input).risk_score
        np.testing.assert_allclose(
            actual, np.asarray(expected), rtol=0.0, atol=0.0, equal_nan=equal_nan
        )


__all__ = [
    "MODEL_ARTIFACT_INVALID",
    "MODEL_INPUT_INCOMPATIBLE",
    "MODEL_OUTPUT_INVALID",
    "MODEL_RUNTIME_UNAVAILABLE",
    "ModelArtifactManifest",
    "ModelArtifactPolicy",
    "RiskModelBackend",
    "RiskModelInput",
    "RiskModelOutput",
    "RuleBaselineBackend",
]
