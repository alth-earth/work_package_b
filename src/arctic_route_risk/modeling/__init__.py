"""Optional model intake and single-step backends for Work Package B."""

from arctic_route_risk.modeling.artifacts import (
    intake_legacy_cnn_zip,
    resolve_model_artifact_policy,
)
from arctic_route_risk.modeling.contracts import (
    MODEL_ARTIFACT_INVALID,
    MODEL_INPUT_INCOMPATIBLE,
    MODEL_OUTPUT_INVALID,
    MODEL_RUNTIME_UNAVAILABLE,
    ModelArtifactManifest,
    ModelArtifactPolicy,
    RiskModelBackend,
    RiskModelInput,
    RiskModelOutput,
    RuleBaselineBackend,
)
from arctic_route_risk.modeling.legacy_cnn import LegacyCnnOneStepBackend

__all__ = [
    "MODEL_ARTIFACT_INVALID",
    "MODEL_INPUT_INCOMPATIBLE",
    "MODEL_OUTPUT_INVALID",
    "MODEL_RUNTIME_UNAVAILABLE",
    "LegacyCnnOneStepBackend",
    "ModelArtifactManifest",
    "ModelArtifactPolicy",
    "RiskModelBackend",
    "RiskModelInput",
    "RiskModelOutput",
    "RuleBaselineBackend",
    "intake_legacy_cnn_zip",
    "resolve_model_artifact_policy",
]
