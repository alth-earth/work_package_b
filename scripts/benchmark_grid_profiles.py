#!/usr/bin/env python3
"""Run the isolated B fixed-grid regridding experiment suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arctic_route_risk.grid_experiments import (
    load_grid_experiment_suite,
    run_grid_kernel_benchmark,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/tromso_grid_profiles_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_grid_kernel_benchmark(load_grid_experiment_suite(args.config))
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
