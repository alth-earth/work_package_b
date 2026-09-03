"""Public Work Package B API."""

from arctic_route_risk.config import (
    DemoRiskModelConfig,
    QualityConfidenceConfig,
    RiskBuildConfiguration,
    RiskComponentConfig,
    TargetGridConfig,
    TemporalMethodConfidenceConfig,
    load_risk_build_configuration,
    model_config_digest,
)
from arctic_route_risk.context import REQUIRED_FORMAL_DATA_TYPES, BInputEnvelope
from arctic_route_risk.errors import (
    CoverageError,
    GridCompatibilityError,
    InputIdentityError,
    PublicationConflictError,
    RiskPipelineError,
    StaleGenerationError,
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
from arctic_route_risk.publishing import PersistentRiskStore, RiskExplanationArtifactStore
from arctic_route_risk.risk_explanation import (
    RiskBuildTraceResult,
    RiskExplanationResearchExporter,
)
from arctic_route_risk.service import RiskBuildRequest, RiskBuildService
from arctic_route_risk.time_horizons import mentor_required_offsets, stage_for_offset

_OPTIONAL_EXPORTS = {
    "LegacyCnnOneStepBackend": (
        "arctic_route_risk.modeling.legacy_cnn",
        "LegacyCnnOneStepBackend",
    ),
    "intake_legacy_cnn_zip": (
        "arctic_route_risk.modeling.artifacts",
        "intake_legacy_cnn_zip",
    ),
    "render_binary_risk_map": ("arctic_route_risk.plotting", "render_binary_risk_map"),
    "render_color_risk_map": ("arctic_route_risk.plotting", "render_color_risk_map"),
    "render_risk_maps": ("arctic_route_risk.plotting", "render_risk_maps"),
}


def __getattr__(name: str):
    """Load legacy model intake and presentation helpers only on explicit use."""

    try:
        module_name, attribute = _OPTIONAL_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    from importlib import import_module

    return getattr(import_module(module_name), attribute)

__all__ = [
    "MODEL_ARTIFACT_INVALID",
    "MODEL_INPUT_INCOMPATIBLE",
    "MODEL_OUTPUT_INVALID",
    "MODEL_RUNTIME_UNAVAILABLE",
    "REQUIRED_FORMAL_DATA_TYPES",
    "BInputEnvelope",
    "CoverageError",
    "DemoRiskModelConfig",
    "GridCompatibilityError",
    "InputIdentityError",
    "LegacyCnnOneStepBackend",
    "ModelArtifactManifest",
    "ModelArtifactPolicy",
    "PersistentRiskStore",
    "PublicationConflictError",
    "QualityConfidenceConfig",
    "RiskBuildConfiguration",
    "RiskBuildRequest",
    "RiskBuildService",
    "RiskBuildTraceResult",
    "RiskComponentConfig",
    "RiskExplanationArtifactStore",
    "RiskExplanationResearchExporter",
    "RiskModelBackend",
    "RiskModelInput",
    "RiskModelOutput",
    "RiskPipelineError",
    "RuleBaselineBackend",
    "StaleGenerationError",
    "TargetGridConfig",
    "TemporalMethodConfidenceConfig",
    "intake_legacy_cnn_zip",
    "load_risk_build_configuration",
    "mentor_required_offsets",
    "model_config_digest",
    "render_binary_risk_map",
    "render_color_risk_map",
    "render_risk_maps",
    "stage_for_offset",
]

__version__ = "0.2.0"
