"""Formal model contracts plus explicitly requested legacy helpers.

Importing ``arctic_route_risk.modeling.contracts`` first executes this package
module.  Keep the optional artifact intake and legacy CNN out of that normal
production path so a trimmed/frozen release can exclude them safely.
"""

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

_OPTIONAL_EXPORTS = {
    "LegacyCnnOneStepBackend": (
        "arctic_route_risk.modeling.legacy_cnn",
        "LegacyCnnOneStepBackend",
    ),
    "intake_legacy_cnn_zip": (
        "arctic_route_risk.modeling.artifacts",
        "intake_legacy_cnn_zip",
    ),
    "resolve_model_artifact_policy": (
        "arctic_route_risk.modeling.artifacts",
        "resolve_model_artifact_policy",
    ),
}


def __getattr__(name: str):
    """Load optional legacy support only when a caller requests it."""

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
