from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from arctic_route_risk.modeling.training import (
    build_one_hour_training_samples,
    evaluate_persistence_baseline,
    iter_training_sample_documents,
    readiness_as_dict,
    summarize_training_readiness,
    temporal_holdout_split,
)

pytestmark = pytest.mark.model


def _frame(
    valid_time: datetime,
    *,
    risk: float,
    route_id: str = "offshore_murmansk_to_offshore_dikson",
) -> SimpleNamespace:
    latitude = np.array([70.0, 70.05], dtype=np.float64)
    longitude = np.array([30.0, 30.05, 30.10], dtype=np.float64)
    risk_score = np.full((2, 3), risk, dtype=np.float32)
    payload = xr.Dataset(
        {
            "risk_score": (("latitude", "longitude"), risk_score),
            "risk_level": (
                ("latitude", "longitude"),
                np.clip(np.floor(risk_score * 5) + 1, 1, 5).astype(np.uint8),
            ),
            "hard_mask": (("latitude", "longitude"), np.zeros((2, 3), dtype=bool)),
            "confidence": (("latitude", "longitude"), np.ones((2, 3), dtype=np.float32)),
            "environment_speed_factor": (
                ("latitude", "longitude"),
                np.full((2, 3), 0.85, dtype=np.float32),
            ),
        },
        coords={"latitude": latitude, "longitude": longitude},
        attrs={"calibration_status": "demo_unvalidated"},
    )
    return SimpleNamespace(
        risk_id=f"{route_id}-{valid_time:%Y%m%d%H}",
        route_id=route_id,
        run_id="run-00000000-0000-4000-8000-000000000001",
        scenario_id="scenario",
        corridor_id=route_id,
        vessel_profile_id="vessel",
        generation_id=7,
        model_config_digest="a" * 64,
        valid_time=valid_time,
        as_of_time=valid_time,
        payload=payload,
    )


def test_builds_explicit_one_hour_training_samples() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    frames = [
        _frame(start + timedelta(hours=2), risk=0.5),
        _frame(start, risk=0.1),
        _frame(start + timedelta(hours=1), risk=0.3),
    ]

    samples = build_one_hour_training_samples(frames)

    assert len(samples) == 2
    assert samples[0].input_valid_time == start
    assert samples[0].target_valid_time == start + timedelta(hours=1)
    assert samples[0].forecast_horizon_minutes == 60
    assert samples[0].input_risk_score.flags.writeable is False
    assert samples[0].valid_training_mask.shape == (2, 3)


def test_rejects_unknown_or_non_hourly_time_semantics() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    frames = [_frame(start, risk=0.1), _frame(start + timedelta(hours=6), risk=0.2)]

    with pytest.raises(ValueError, match="one-hour"):
        build_one_hour_training_samples(frames)


def test_rejects_mixed_route_identity() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    frames = [
        _frame(start, risk=0.1),
        _frame(start + timedelta(hours=1), risk=0.2, route_id="tromso_to_svalbard"),
    ]

    with pytest.raises(ValueError, match="share run"):
        build_one_hour_training_samples(frames)


def test_temporal_split_readiness_and_baseline_are_auditable() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    frames = [_frame(start + timedelta(hours=hour), risk=hour / 10) for hour in range(5)]
    samples = build_one_hour_training_samples(frames)

    split = temporal_holdout_split(samples, validation_fraction=0.25)
    evaluation = evaluate_persistence_baseline(split.validation)
    readiness = summarize_training_readiness(
        samples,
        split,
        minimum_total_samples=4,
        minimum_validation_samples=1,
    )

    assert len(split.train) == 3
    assert len(split.validation) == 1
    assert evaluation.sample_count == 1
    assert evaluation.valid_cell_count == 6
    assert evaluation.mse > 0
    assert readiness.ready_for_formal_training
    assert readiness.forecast_horizon_minutes == 60
    assert readiness_as_dict(readiness)["status"] == "ready"
    assert iter_training_sample_documents(samples)[0]["forecast_horizon_minutes"] == 60


def test_readiness_blocks_small_unvalidated_training_sets() -> None:
    start = datetime(2026, 8, 1, tzinfo=UTC)
    samples = build_one_hour_training_samples(
        [_frame(start, risk=0.1), _frame(start + timedelta(hours=1), risk=0.2)]
    )

    readiness = summarize_training_readiness(samples)

    assert not readiness.ready_for_formal_training
    assert readiness.status == "blocked"
    assert any("not enough total samples" in blocker for blocker in readiness.blockers)
