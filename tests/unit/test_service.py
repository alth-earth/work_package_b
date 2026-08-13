from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
from arctic_route_planning.contracts import validate_canonical_risk_id

from arctic_route_risk import (
    BInputEnvelope,
    RiskBuildRequest,
    RiskBuildService,
    TargetGridConfig,
    load_risk_build_configuration,
    model_config_digest,
)
from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.service import _regrid_variable

GENERATED = datetime(2026, 8, 2, tzinfo=UTC)
PROJECT_ROOT = Path(__file__).parents[2]


def _frames(formal_fixture):
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    request = RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)
    return RiskBuildService(utc_now=lambda: GENERATED).build_window(request)


def test_builds_complete_hourly_c_contract_window(formal_fixture) -> None:
    frames = _frames(formal_fixture)

    assert len(frames) == 3
    assert frames[0].valid_time == formal_fixture.context.simulation_start
    assert frames[-1].valid_time == formal_fixture.context.simulation_end
    assert all(
        (right.valid_time - left.valid_time).total_seconds() == 3600
        for left, right in pairwise(frames)
    )
    for frame in frames:
        validate_canonical_risk_id(frame)
        assert frame.generated_at == GENERATED
        assert frame.payload.attrs["calibration_status"] == "demo_unvalidated"
        assert frame.payload["hard_mask"].dtype == np.bool_
        assert float(frame.payload["environment_speed_factor"].min()) > 0
        risk = frame.payload["risk_score"].values
        level = frame.payload["risk_level"].values
        finite = np.isfinite(risk)
        np.testing.assert_array_equal(
            level[finite],
            np.clip(np.floor(risk[finite] * 5) + 1, 1, 5).astype(np.uint8),
        )
        assert all(source.issue_time <= frame.as_of_time for source in frame.source_summary)


def test_land_sea_mask_uses_nearest_spatial_alignment(formal_fixture) -> None:
    source = formal_fixture.prepared.frames["land_sea_mask"][0]
    latitude = np.linspace(70.0, 72.0, 9)
    longitude = np.linspace(10.0, 12.0, 9)

    values = _regrid_variable(
        source,
        "land_sea_mask",
        latitude,
        longitude,
        categorical=True,
    )

    assert set(np.unique(values)) == {0.0, 1.0}


def test_model_digest_is_policy_only_not_realized_corridor_grid() -> None:
    grid = TargetGridConfig(latitude_step_degrees=0.5, longitude_step_degrees=0.5)
    model = DemoRiskModelConfig()
    first = model_config_digest(grid=grid, model=model)
    grid.realize((10.0, 70.0, 12.0, 72.0))
    second = model_config_digest(grid=grid, model=model)
    grid.realize((30.0, 69.0, 80.0, 75.0))

    assert first == second == model_config_digest(grid=grid, model=model)


def test_versioned_default_configuration_is_the_runtime_policy() -> None:
    configuration = load_risk_build_configuration(
        PROJECT_ROOT / "configs/models/demo_unvalidated_v1.json"
    )

    assert configuration.grid_config == TargetGridConfig()
    assert configuration.model_config == DemoRiskModelConfig()
    assert configuration.model_config_digest == model_config_digest(
        grid=TargetGridConfig(),
        model=DemoRiskModelConfig(),
    )
