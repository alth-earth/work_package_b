"""Auditable training sample preparation for formal Work Package B risk frames.

This module deliberately trains no opaque model by itself.  Its job is to turn
formal ``bc.risk-frame.v2`` outputs into time-aware, reproducible samples that
can be used by a later model trainer without weakening the B -> C contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from itertools import pairwise
from math import ceil
from typing import Any

import numpy as np

REQUIRED_TRAINING_VARIABLES = (
    "risk_score",
    "risk_level",
    "hard_mask",
    "confidence",
    "environment_speed_factor",
)

ONE_HOUR_MINUTES = 60


@dataclass(frozen=True)
class RiskTrainingSample:
    """One explicit one-hour-ahead supervised sample.

    ``input_*`` represents the current formal risk frame. ``target_*`` represents
    the next formal risk frame on the same grid and identity.  Missing cells stay
    as NaN and must be masked by the trainer.
    """

    sample_id: str
    route_id: str
    run_id: str
    scenario_id: str
    corridor_id: str
    vessel_profile_id: str
    generation_id: int
    model_config_digest: str
    input_valid_time: datetime
    target_valid_time: datetime
    forecast_horizon_minutes: int
    latitude: np.ndarray
    longitude: np.ndarray
    input_risk_score: np.ndarray
    target_risk_score: np.ndarray
    input_risk_level: np.ndarray
    target_risk_level: np.ndarray
    hard_mask: np.ndarray
    confidence: np.ndarray
    environment_speed_factor: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for field_name in (
            "latitude",
            "longitude",
            "input_risk_score",
            "target_risk_score",
            "input_risk_level",
            "target_risk_level",
            "hard_mask",
            "confidence",
            "environment_speed_factor",
        ):
            value = np.asarray(getattr(self, field_name)).copy()
            value.setflags(write=False)
            object.__setattr__(self, field_name, value)

    @property
    def valid_training_mask(self) -> np.ndarray:
        """Cells that have finite input, target and confidence and are not hard masked."""

        mask = (
            np.isfinite(self.input_risk_score)
            & np.isfinite(self.target_risk_score)
            & np.isfinite(self.confidence)
            & (self.confidence > 0)
            & ~self.hard_mask.astype(bool)
        )
        mask.setflags(write=False)
        return mask


@dataclass(frozen=True)
class RiskTrainingSplit:
    """Deterministic temporal split used for model training and validation."""

    train: tuple[RiskTrainingSample, ...]
    validation: tuple[RiskTrainingSample, ...]
    split_method: str


@dataclass(frozen=True)
class RiskTrainingEvaluation:
    """Baseline evaluation for a candidate one-hour-ahead risk predictor."""

    sample_count: int
    valid_cell_count: int
    mse: float
    mae: float
    max_abs_error: float
    baseline_name: str = "persistence_one_hour"


@dataclass(frozen=True)
class RiskTrainingReadiness:
    """Plain readiness report for promoting model training beyond an experiment."""

    status: str
    ready_for_formal_training: bool
    total_samples: int
    train_samples: int
    validation_samples: int
    forecast_horizon_minutes: int | None
    valid_time_start: datetime | None
    valid_time_end: datetime | None
    route_ids: tuple[str, ...]
    blockers: tuple[str, ...]
    notes: tuple[str, ...]


def build_one_hour_training_samples(frames: Sequence[Any]) -> tuple[RiskTrainingSample, ...]:
    """Build adjacent-frame samples with explicit one-hour target semantics.

    The input must be formal B frames ordered or orderable by ``valid_time``.
    Consecutive frames are paired only when their identities and grids match and
    their valid times are exactly one hour apart.
    """

    if len(frames) < 2:
        return ()

    ordered = tuple(sorted(frames, key=lambda frame: _require_datetime(frame, "valid_time")))
    samples: list[RiskTrainingSample] = []
    for current, target in pairwise(ordered):
        current_time = _require_datetime(current, "valid_time")
        target_time = _require_datetime(target, "valid_time")
        horizon_minutes = int((target_time - current_time).total_seconds() // 60)
        if horizon_minutes != ONE_HOUR_MINUTES:
            raise ValueError(
                "training frames must be adjacent one-hour frames; "
                f"got {horizon_minutes} minutes between {current_time} and {target_time}"
            )

        current_identity = _frame_identity(current)
        target_identity = _frame_identity(target)
        if current_identity != target_identity:
            raise ValueError(
                "training frames must share run, scenario, corridor, "
                "vessel and config identity"
            )

        latitude = _coord_array(current, "latitude")
        longitude = _coord_array(current, "longitude")
        np.testing.assert_array_equal(latitude, _coord_array(target, "latitude"))
        np.testing.assert_array_equal(longitude, _coord_array(target, "longitude"))

        _assert_payload_variables(current)
        _assert_payload_variables(target)

        input_risk = _payload_array(current, "risk_score", np.float32)
        target_risk = _payload_array(target, "risk_score", np.float32)
        if input_risk.shape != target_risk.shape:
            raise ValueError("training frames must use the same risk grid shape")

        hard_mask = _payload_array(current, "hard_mask", np.bool_)
        confidence = _payload_array(current, "confidence", np.float32)
        speed_factor = _payload_array(current, "environment_speed_factor", np.float32)
        for name, values in (
            ("hard_mask", hard_mask),
            ("confidence", confidence),
            ("environment_speed_factor", speed_factor),
        ):
            if values.shape != input_risk.shape:
                raise ValueError(f"{name} shape does not match risk_score")

        metadata = {
            "schema_version": "risk-training-sample.v1",
            "source_contract": "bc.risk-frame.v2",
            "target_semantics": "one_hour_ahead",
            "input_risk_id": getattr(current, "risk_id", None),
            "target_risk_id": getattr(target, "risk_id", None),
            "input_as_of_time": _optional_iso(getattr(current, "as_of_time", None)),
            "target_as_of_time": _optional_iso(getattr(target, "as_of_time", None)),
            "calibration_status": str(current.payload.attrs.get("calibration_status", "")),
        }

        samples.append(
            RiskTrainingSample(
                sample_id=_sample_id(
                    current_identity,
                    current_time,
                    target_time,
                    input_risk,
                    target_risk,
                ),
                route_id=current_identity["route_id"],
                run_id=current_identity["run_id"],
                scenario_id=current_identity["scenario_id"],
                corridor_id=current_identity["corridor_id"],
                vessel_profile_id=current_identity["vessel_profile_id"],
                generation_id=current_identity["generation_id"],
                model_config_digest=current_identity["model_config_digest"],
                input_valid_time=current_time,
                target_valid_time=target_time,
                forecast_horizon_minutes=horizon_minutes,
                latitude=latitude,
                longitude=longitude,
                input_risk_score=input_risk,
                target_risk_score=target_risk,
                input_risk_level=_payload_array(current, "risk_level", np.uint8),
                target_risk_level=_payload_array(target, "risk_level", np.uint8),
                hard_mask=hard_mask,
                confidence=confidence,
                environment_speed_factor=speed_factor,
                metadata=metadata,
            )
        )
    return tuple(samples)


def temporal_holdout_split(
    samples: Sequence[RiskTrainingSample],
    *,
    validation_fraction: float = 0.2,
    minimum_validation_samples: int = 1,
) -> RiskTrainingSplit:
    """Split samples by time so validation always occurs after training."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = tuple(sorted(samples, key=lambda sample: sample.input_valid_time))
    if len(ordered) < 2:
        return RiskTrainingSplit(train=ordered, validation=(), split_method="temporal_tail")
    validation_count = max(minimum_validation_samples, ceil(len(ordered) * validation_fraction))
    validation_count = min(validation_count, len(ordered) - 1)
    return RiskTrainingSplit(
        train=ordered[:-validation_count],
        validation=ordered[-validation_count:],
        split_method="temporal_tail",
    )


def evaluate_persistence_baseline(
    samples: Sequence[RiskTrainingSample],
) -> RiskTrainingEvaluation:
    """Evaluate the current-risk-equals-next-risk baseline on finite sea cells."""

    errors: list[np.ndarray] = []
    valid_cell_count = 0
    for sample in samples:
        mask = sample.valid_training_mask
        if not bool(mask.any()):
            continue
        delta = sample.input_risk_score[mask] - sample.target_risk_score[mask]
        errors.append(delta.astype(np.float64, copy=False))
        valid_cell_count += int(mask.sum())
    if not errors:
        return RiskTrainingEvaluation(
            sample_count=len(samples),
            valid_cell_count=0,
            mse=float("nan"),
            mae=float("nan"),
            max_abs_error=float("nan"),
        )
    combined = np.concatenate(errors)
    return RiskTrainingEvaluation(
        sample_count=len(samples),
        valid_cell_count=valid_cell_count,
        mse=float(np.mean(combined**2)),
        mae=float(np.mean(np.abs(combined))),
        max_abs_error=float(np.max(np.abs(combined))),
    )


def summarize_training_readiness(
    samples: Sequence[RiskTrainingSample],
    split: RiskTrainingSplit | None = None,
    *,
    minimum_total_samples: int = 168,
    minimum_validation_samples: int = 24,
) -> RiskTrainingReadiness:
    """Return a conservative readiness summary for formal model training."""

    ordered = tuple(sorted(samples, key=lambda sample: sample.input_valid_time))
    if split is None:
        split = (
            temporal_holdout_split(ordered)
            if ordered
            else RiskTrainingSplit((), (), "temporal_tail")
        )

    blockers: list[str] = []
    notes: list[str] = [
        "Formal training samples are derived from B RiskFrame v2 outputs, "
        "not from the legacy CNN ZIP.",
        "The forecast target is exactly one hour ahead; multi-hour products "
        "must be trained or validated separately.",
        "The legacy CNN remains experimental_unverified and is not promoted by this report.",
    ]
    horizon_values = {sample.forecast_horizon_minutes for sample in ordered}
    if len(horizon_values) > 1:
        blockers.append("mixed forecast horizons detected")
    if ordered and horizon_values != {ONE_HOUR_MINUTES}:
        blockers.append("forecast horizon is not exactly one hour")
    if len(ordered) < minimum_total_samples:
        blockers.append(
            "not enough total samples for formal training: "
            f"{len(ordered)} < {minimum_total_samples}"
        )
    if len(split.validation) < minimum_validation_samples:
        blockers.append(
            "not enough temporal holdout samples for validation: "
            f"{len(split.validation)} < {minimum_validation_samples}"
        )
    if ordered:
        route_counts: dict[str, int] = {}
        for sample in ordered:
            route_counts[sample.route_id] = route_counts.get(sample.route_id, 0) + 1
        if len(route_counts) < 2:
            notes.append(
                "Only one route is present; corridor transfer ability still needs "
                "separate evidence."
            )

    return RiskTrainingReadiness(
        status="ready" if not blockers else "blocked",
        ready_for_formal_training=not blockers,
        total_samples=len(ordered),
        train_samples=len(split.train),
        validation_samples=len(split.validation),
        forecast_horizon_minutes=next(iter(horizon_values)) if len(horizon_values) == 1 else None,
        valid_time_start=ordered[0].input_valid_time if ordered else None,
        valid_time_end=ordered[-1].target_valid_time if ordered else None,
        route_ids=tuple(sorted({sample.route_id for sample in ordered})),
        blockers=tuple(blockers),
        notes=tuple(notes),
    )


def readiness_as_dict(readiness: RiskTrainingReadiness) -> dict[str, Any]:
    """Serialize a readiness report without losing UTC timestamps."""

    return {
        "status": readiness.status,
        "ready_for_formal_training": readiness.ready_for_formal_training,
        "total_samples": readiness.total_samples,
        "train_samples": readiness.train_samples,
        "validation_samples": readiness.validation_samples,
        "forecast_horizon_minutes": readiness.forecast_horizon_minutes,
        "valid_time_start": _optional_iso(readiness.valid_time_start),
        "valid_time_end": _optional_iso(readiness.valid_time_end),
        "route_ids": list(readiness.route_ids),
        "blockers": list(readiness.blockers),
        "notes": list(readiness.notes),
    }


def _assert_payload_variables(frame: Any) -> None:
    missing = [name for name in REQUIRED_TRAINING_VARIABLES if name not in frame.payload]
    if missing:
        raise ValueError(f"risk frame payload is missing training variables: {missing}")


def _coord_array(frame: Any, name: str) -> np.ndarray:
    if name not in frame.payload.coords:
        raise ValueError(f"risk frame payload is missing {name} coordinate")
    return np.asarray(frame.payload.coords[name].values, dtype=np.float64)


def _payload_array(frame: Any, name: str, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    if name not in frame.payload:
        raise ValueError(f"risk frame payload is missing {name}")
    return np.asarray(frame.payload[name].values, dtype=dtype)


def _frame_identity(frame: Any) -> dict[str, Any]:
    return {
        "route_id": str(frame.route_id),
        "run_id": str(frame.run_id),
        "scenario_id": str(frame.scenario_id),
        "corridor_id": str(frame.corridor_id),
        "vessel_profile_id": str(frame.vessel_profile_id),
        "generation_id": int(frame.generation_id),
        "model_config_digest": str(frame.model_config_digest),
    }


def _require_datetime(frame: Any, field_name: str) -> datetime:
    value = getattr(frame, field_name)
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _sample_id(
    identity: Mapping[str, Any],
    input_valid_time: datetime,
    target_valid_time: datetime,
    input_risk: np.ndarray,
    target_risk: np.ndarray,
) -> str:
    digest = sha256()
    for key in sorted(identity):
        digest.update(str(identity[key]).encode("utf-8"))
        digest.update(b"\0")
    digest.update(input_valid_time.isoformat().encode("utf-8"))
    digest.update(target_valid_time.isoformat().encode("utf-8"))
    digest.update(np.nan_to_num(input_risk, nan=-1.0).astype(np.float32).tobytes())
    digest.update(np.nan_to_num(target_risk, nan=-1.0).astype(np.float32).tobytes())
    return digest.hexdigest()


def _optional_iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return str(value)


def iter_training_sample_documents(
    samples: Iterable[RiskTrainingSample],
) -> tuple[dict[str, Any], ...]:
    """Return lightweight metadata records suitable for an audit manifest."""

    return tuple(
        {
            "sample_id": sample.sample_id,
            "route_id": sample.route_id,
            "run_id": sample.run_id,
            "scenario_id": sample.scenario_id,
            "corridor_id": sample.corridor_id,
            "vessel_profile_id": sample.vessel_profile_id,
            "generation_id": sample.generation_id,
            "model_config_digest": sample.model_config_digest,
            "input_valid_time": sample.input_valid_time.isoformat(),
            "target_valid_time": sample.target_valid_time.isoformat(),
            "forecast_horizon_minutes": sample.forecast_horizon_minutes,
            "shape": list(sample.input_risk_score.shape),
            "finite_training_cells": int(sample.valid_training_mask.sum()),
            "metadata": dict(sample.metadata),
        }
        for sample in samples
    )
