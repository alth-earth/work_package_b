from __future__ import annotations

from pathlib import Path

from arctic_route_risk.config import TargetGridConfig
from arctic_route_risk.grid_experiments import (
    load_grid_experiment_suite,
    run_grid_kernel_benchmark,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/tromso_grid_profiles_v1.json"


def test_grid_experiment_profiles_do_not_change_production_default() -> None:
    suite = load_grid_experiment_suite(CONFIG)

    assert TargetGridConfig().latitude_step_degrees == 0.75
    assert TargetGridConfig().longitude_step_degrees == 2.2
    assert tuple(profile.name for profile in suite.profiles) == (
        "baseline",
        "medium",
        "fine",
    )
    shapes = tuple(
        tuple(axis.size for axis in profile.target_grid().realize(suite.bbox)[:2])
        for profile in suite.profiles
    )
    assert shapes == ((16, 7), (31, 11), (60, 21))


def test_grid_kernel_benchmark_is_deterministic() -> None:
    suite = load_grid_experiment_suite(CONFIG)
    first = run_grid_kernel_benchmark(suite)
    second = run_grid_kernel_benchmark(suite)

    assert first["status"] == "EXPERIMENTAL"
    assert first["formal_risk_build"] is False
    assert [item["cells"] for item in first["profiles"]] == [112, 341, 1260]
    assert [item["output_digest"] for item in first["profiles"]] == [
        item["output_digest"] for item in second["profiles"]
    ]
