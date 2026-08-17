"""Conservative hard-mask policy: unknown risk inputs make a node unavailable."""

from __future__ import annotations

import numpy as np

from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.service import _demo_unvalidated_risk


def _base_values() -> dict[str, np.ndarray]:
    grid = np.ones((2, 2), dtype=np.float64)
    return {
        "land_sea_mask": grid.copy(),
        "ice_concentration": grid.copy(),
        "ice_thickness": grid.copy(),
        "ice_type": grid.copy(),
        "ice_edge": grid.copy(),
        "ice_drift_u": grid.copy(),
        "ice_drift_v": grid.copy(),
        "significant_wave_height": grid.copy(),
        "ocean_current_u": grid.copy(),
        "ocean_current_v": grid.copy(),
        "wind_u10": grid.copy(),
        "wind_v10": grid.copy(),
        "air_temperature_2m": grid.copy() + 273.0,
        "visibility": grid.copy() * 1000.0,
        "sea_surface_height": grid.copy(),
    }


def test_default_policy_keeps_unknown_navigable() -> None:
    values = _base_values()
    values["ocean_current_u"][0, 0] = np.nan
    risk, hard, _, _, reason = _demo_unvalidated_risk(
        values=values,
        source_confidence=1.0,
        model_config=DemoRiskModelConfig(),
    )
    assert np.isnan(risk[0, 0])
    assert not hard[0, 0]
    assert reason[0, 0] == "NONE"


def test_plus_unknown_policy_marks_unknown_node_hard() -> None:
    values = _base_values()
    values["ocean_current_u"][0, 0] = np.nan
    model = DemoRiskModelConfig(hard_mask_policy="land_sea_mask_plus_unknown_v1")
    risk, hard, confidence, _, reason = _demo_unvalidated_risk(
        values=values,
        source_confidence=1.0,
        model_config=model,
    )
    assert np.isnan(risk[0, 0])
    assert hard[0, 0]
    assert confidence[0, 0] == 0.0
    assert reason[0, 0] == "DATA_UNAVAILABLE"
    # finite neighbor stays navigable
    assert not hard[0, 1]
    assert reason[0, 1] == "NONE"


def test_plus_unknown_policy_keeps_land_hard() -> None:
    values = _base_values()
    values["land_sea_mask"][1, 1] = 0.0  # land
    model = DemoRiskModelConfig(hard_mask_policy="land_sea_mask_plus_unknown_v1")
    _, hard, _, _, reason = _demo_unvalidated_risk(
        values=values,
        source_confidence=1.0,
        model_config=model,
    )
    assert hard[1, 1]
    assert reason[1, 1] == "LAND"


def test_unsupported_policy_rejected() -> None:
    import pytest

    from arctic_route_risk.errors import RiskPipelineError

    with pytest.raises(RiskPipelineError):
        DemoRiskModelConfig(hard_mask_policy="invented_policy_v9")
