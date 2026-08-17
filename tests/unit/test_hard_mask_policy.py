"""Conservative hard-mask policy: unknown risk inputs make a node unavailable."""

from __future__ import annotations

import numpy as np

from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.service import (
    _apply_ice_free_neutral_fill,
    _demo_unvalidated_risk,
)


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


def test_ice_free_neutral_fill_only_fills_ice_type_and_edge() -> None:
    values = {
        "ice_concentration": np.array([[0.0, 0.3], [np.nan, 0.0]]),
        "ice_type": np.array([[np.nan, np.nan], [np.nan, 2.0]]),
        "ice_edge": np.array([[np.nan, np.nan], [np.nan, 1.0]]),
    }
    filled, counts = _apply_ice_free_neutral_fill(values)

    # ice-free cells (0.0) get neutral 0.0
    assert filled["ice_type"][0, 0] == 0.0
    assert filled["ice_edge"][0, 0] == 0.0
    # above threshold stays missing
    assert np.isnan(filled["ice_type"][0, 1])
    # concentration unknown stays missing
    assert np.isnan(filled["ice_type"][1, 0])
    # existing finite value is preserved
    assert filled["ice_type"][1, 1] == 2.0
    assert filled["ice_edge"][1, 1] == 1.0
    assert counts == {"ice_type": 1, "ice_edge": 1}


def test_ice_free_predicate_boundary_and_fail_closed() -> None:
    concentration = np.array([[0.149, 0.150, 0.151], [np.nan, -0.1, 0.0]])
    values = {
        "ice_concentration": concentration,
        "ice_type": np.full(concentration.shape, np.nan),
        "ice_edge": np.full(concentration.shape, np.nan),
    }
    filled, counts = _apply_ice_free_neutral_fill(values)

    # strictly below threshold and non-negative -> neutralised
    assert filled["ice_type"][0, 0] == 0.0
    assert filled["ice_edge"][0, 0] == 0.0
    assert filled["ice_type"][1, 2] == 0.0
    # exactly at threshold is ice, not open water -> NOT neutralised
    assert np.isnan(filled["ice_type"][0, 1])
    assert np.isnan(filled["ice_edge"][0, 1])
    # above threshold -> NOT neutralised
    assert np.isnan(filled["ice_type"][0, 2])
    # concentration NaN cannot prove open water -> fail-closed
    assert np.isnan(filled["ice_type"][1, 0])
    # negative concentration is invalid evidence -> fail-closed
    assert np.isnan(filled["ice_type"][1, 1])
    assert counts == {"ice_type": 2, "ice_edge": 2}


def test_ice_free_policy_marks_true_unknown_hard_but_ice_free_navigable() -> None:
    shape = (2, 2)
    values = {
        "land_sea_mask": np.ones(shape),
        "ice_concentration": np.array([[0.0, np.nan], [0.0, 0.0]]),
        "ice_thickness": np.ones(shape),
        "ice_type": np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        "ice_edge": np.array([[np.nan, np.nan], [np.nan, np.nan]]),
        "ice_drift_u": np.zeros(shape),
        "ice_drift_v": np.zeros(shape),
        "significant_wave_height": np.zeros(shape),
        "ocean_current_u": np.zeros(shape),
        "ocean_current_v": np.zeros(shape),
        "wind_u10": np.zeros(shape),
        "wind_v10": np.zeros(shape),
        "air_temperature_2m": np.full(shape, 280.0),
        "visibility": np.full(shape, 20_000.0),
        "sea_surface_height": np.zeros(shape),
    }
    model = DemoRiskModelConfig(
        hard_mask_policy="land_sea_mask_plus_unknown_ice_free_v1"
    )
    filled, _ = _apply_ice_free_neutral_fill(values)
    risk, hard, confidence, _, reason = _demo_unvalidated_risk(
        values=filled,
        source_confidence=1.0,
        model_config=model,
    )
    # ice-free cells are navigable and NONE
    assert not hard[0, 0]
    assert reason[0, 0] == "NONE"
    assert np.isfinite(risk[0, 0])
    # unknown concentration remains DATA_UNAVAILABLE
    assert hard[0, 1]
    assert reason[0, 1] == "DATA_UNAVAILABLE"
    assert confidence[0, 1] == 0.0
