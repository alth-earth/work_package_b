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
from arctic_route_risk.modeling.training import (
    ONE_HOUR_MINUTES,
    REQUIRED_TRAINING_VARIABLES,
    RiskTrainingEvaluation,
    RiskTrainingReadiness,
    RiskTrainingSample,
    RiskTrainingSplit,
    build_one_hour_training_samples,
    evaluate_persistence_baseline,
    iter_training_sample_documents,
    readiness_as_dict,
    summarize_training_readiness,
    temporal_holdout_split,
)

__all__ = [
    "MODEL_ARTIFACT_INVALID",
    "MODEL_INPUT_INCOMPATIBLE",
    "MODEL_OUTPUT_INVALID",
    "MODEL_RUNTIME_UNAVAILABLE",
    "ONE_HOUR_MINUTES",
    "REQUIRED_TRAINING_VARIABLES",
    "LegacyCnnOneStepBackend",
    "ModelArtifactManifest",
    "ModelArtifactPolicy",
    "RiskModelBackend",
    "RiskModelInput",
    "RiskModelOutput",
    "RiskTrainingEvaluation",
    "RiskTrainingReadiness",
    "RiskTrainingSample",
    "RiskTrainingSplit",
    "RuleBaselineBackend",
    "build_one_hour_training_samples",
    "evaluate_persistence_baseline",
    "intake_legacy_cnn_zip",
    "iter_training_sample_documents",
    "readiness_as_dict",
    "resolve_model_artifact_policy",
    "summarize_training_readiness",
    "temporal_holdout_split",
]
