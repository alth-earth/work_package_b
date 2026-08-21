from __future__ import annotations

from datetime import UTC, datetime

from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.context import BInputEnvelope
from arctic_route_risk.formal_grid_experiments import build_formal_grid_profile
from arctic_route_risk.grid_experiments import GridExperimentProfile


def test_formal_grid_profile_runs_real_builder_and_summarizes(formal_fixture) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    profile = GridExperimentProfile(
        name="baseline",
        latitude_step_degrees=1.0,
        longitude_step_degrees=1.0,
    )
    result = build_formal_grid_profile(
        envelope=envelope,
        target_bbox=formal_fixture.bbox,
        model_config=DemoRiskModelConfig(),
        profile=profile,
        utc_now=lambda: datetime(2026, 8, 1, tzinfo=UTC),
    )

    assert len(result.frames) == 3
    assert result.summary["schema_versions"] == ["bc.risk-frame.v2"]
    assert result.summary["source_provenance"] == ["formal"]
    assert result.summary["rows"] == 3
    assert result.summary["cols"] == 3
    assert result.summary["cells_per_frame"] == 9
    assert sum(result.summary["risk_levels"].values()) == 27
    assert result.summary["risk_frame_json_bytes"] > 0
    assert len(result.summary["frame_distributions"]) == 3
