"""Isolated fixed-grid experiment helpers; never used by formal B production."""

from __future__ import annotations

import hashlib
import json
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import xarray as xr

try:  # pragma: no cover - Windows does not provide resource.
    import resource
except ModuleNotFoundError:  # pragma: no cover - platform fallback.
    resource = None

from arctic_route_risk.config import TargetGridConfig
from arctic_route_risk.errors import RiskPipelineError


@dataclass(frozen=True, slots=True)
class GridExperimentProfile:
    """One named fixed-grid policy used only by the benchmark harness."""

    name: str
    latitude_step_degrees: float
    longitude_step_degrees: float

    def __post_init__(self) -> None:
        if self.name not in {"baseline", "medium", "fine"}:
            raise RiskPipelineError(f"unsupported grid experiment profile: {self.name}")
        for field_name in ("latitude_step_degrees", "longitude_step_degrees"):
            value = getattr(self, field_name)
            if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise RiskPipelineError(f"{field_name} must be positive and finite")

    def target_grid(self) -> TargetGridConfig:
        return TargetGridConfig(
            latitude_step_degrees=self.latitude_step_degrees,
            longitude_step_degrees=self.longitude_step_degrees,
        )


@dataclass(frozen=True, slots=True)
class GridExperimentSuite:
    """Strict experiment definition kept separate from formal model configuration."""

    schema_version: str
    bbox: tuple[float, float, float, float]
    source_latitude_step_degrees: float
    source_longitude_step_degrees: float
    iterations: int
    profiles: tuple[GridExperimentProfile, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "b.grid-experiment-suite.v1":
            raise RiskPipelineError("unsupported grid experiment suite version")
        if len(self.bbox) != 4:
            raise RiskPipelineError("grid experiment bbox must contain four values")
        TargetGridConfig().realize(self.bbox)
        for field_name in (
            "source_latitude_step_degrees",
            "source_longitude_step_degrees",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value <= 0:
                raise RiskPipelineError(f"{field_name} must be positive and finite")
        if isinstance(self.iterations, bool) or self.iterations <= 0:
            raise RiskPipelineError("grid experiment iterations must be a positive integer")
        if tuple(profile.name for profile in self.profiles) != (
            "baseline",
            "medium",
            "fine",
        ):
            raise RiskPipelineError(
                "grid experiment profiles must be baseline, medium, fine in order"
            )


def load_grid_experiment_suite(path: str | Path) -> GridExperimentSuite:
    """Load a strict experiment suite without touching production configuration."""

    location = Path(path)
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RiskPipelineError(f"cannot load grid experiment suite: {location}") from exc
    if not isinstance(document, dict):
        raise RiskPipelineError("grid experiment suite must be a JSON object")
    expected = {
        "schema_version",
        "bbox",
        "source_latitude_step_degrees",
        "source_longitude_step_degrees",
        "iterations",
        "profiles",
    }
    _require_exact_keys(document, expected, field="grid experiment suite")
    raw_profiles = document["profiles"]
    if not isinstance(raw_profiles, list):
        raise RiskPipelineError("grid experiment profiles must be an array")
    profile_fields = {
        "name",
        "latitude_step_degrees",
        "longitude_step_degrees",
    }
    profiles: list[GridExperimentProfile] = []
    for index, raw_profile in enumerate(raw_profiles):
        if not isinstance(raw_profile, dict):
            raise RiskPipelineError(f"grid experiment profile {index} must be an object")
        _require_exact_keys(raw_profile, profile_fields, field=f"profile {index}")
        profiles.append(GridExperimentProfile(**raw_profile))
    try:
        return GridExperimentSuite(
            schema_version=document["schema_version"],
            bbox=tuple(float(value) for value in document["bbox"]),
            source_latitude_step_degrees=float(
                document["source_latitude_step_degrees"]
            ),
            source_longitude_step_degrees=float(
                document["source_longitude_step_degrees"]
            ),
            iterations=document["iterations"],
            profiles=tuple(profiles),
        )
    except (TypeError, ValueError) as exc:
        raise RiskPipelineError("grid experiment suite field types are invalid") from exc


def run_grid_kernel_benchmark(
    suite: GridExperimentSuite,
) -> dict[str, Any]:
    """Benchmark deterministic xarray regridding, not a formal B risk build."""

    continuous, categorical = _synthetic_source_datasets(suite)
    results = [
        _benchmark_profile(suite, profile, continuous, categorical)
        for profile in suite.profiles
    ]
    return {
        "schema_version": "b.grid-kernel-benchmark.v1",
        "status": "EXPERIMENTAL",
        "scope": "synthetic_spatial_regridding_kernel_only",
        "formal_risk_build": False,
        "bbox": list(suite.bbox),
        "iterations": suite.iterations,
        "profiles": results,
    }


def _synthetic_source_datasets(
    suite: GridExperimentSuite,
) -> tuple[xr.Dataset, xr.Dataset]:
    west, south, east, north = suite.bbox
    source_latitude = _axis(south, north, suite.source_latitude_step_degrees)
    source_longitude = _axis(west, east, suite.source_longitude_step_degrees)
    lat_mesh, lon_mesh = np.meshgrid(source_latitude, source_longitude, indexing="ij")
    continuous = xr.Dataset(
        {
            f"continuous_{index}": (
                ("latitude", "longitude"),
                np.sin(np.radians(lat_mesh * (index + 1)))
                + np.cos(np.radians(lon_mesh * (index + 2))),
            )
            for index in range(8)
        },
        coords={"latitude": source_latitude, "longitude": source_longitude},
    )
    categorical = xr.Dataset(
        {
            f"categorical_{index}": (
                ("latitude", "longitude"),
                ((lat_mesh + lon_mesh + index) % (index + 2)).astype(np.int16),
            )
            for index in range(3)
        },
        coords={"latitude": source_latitude, "longitude": source_longitude},
    )
    return continuous, categorical


def _axis(start: float, end: float, maximum_step: float) -> np.ndarray:
    size = math.ceil((end - start) / maximum_step) + 1
    return np.linspace(start, end, size, dtype=np.float64)


def _benchmark_profile(
    suite: GridExperimentSuite,
    profile: GridExperimentProfile,
    continuous: xr.Dataset,
    categorical: xr.Dataset,
) -> dict[str, Any]:
    latitude, longitude, grid_id = profile.target_grid().realize(suite.bbox)
    durations: list[float] = []
    digest = ""
    output_bytes = 0
    # Exclude xarray/scipy one-time import and index construction from timing.
    continuous.interp(latitude=latitude, longitude=longitude, method="linear")
    categorical.interp(latitude=latitude, longitude=longitude, method="nearest")
    tracemalloc.start()
    for _ in range(suite.iterations):
        started = time.perf_counter()
        continuous_output = continuous.interp(
            latitude=latitude,
            longitude=longitude,
            method="linear",
        )
        categorical_output = categorical.interp(
            latitude=latitude,
            longitude=longitude,
            method="nearest",
        )
        arrays = tuple(
            np.asarray(array.values)
            for dataset in (continuous_output, categorical_output)
            for array in dataset.data_vars.values()
        )
        durations.append((time.perf_counter() - started) * 1000.0)
        output_bytes = sum(array.nbytes for array in arrays)
        digest = _arrays_digest(arrays)
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "name": profile.name,
        "latitude_step_degrees": profile.latitude_step_degrees,
        "longitude_step_degrees": profile.longitude_step_degrees,
        "rows": int(latitude.size),
        "cols": int(longitude.size),
        "cells": int(latitude.size * longitude.size),
        "grid_id": grid_id,
        "mean_runtime_ms": round(fmean(durations), 6),
        "minimum_runtime_ms": round(min(durations), 6),
        "maximum_runtime_ms": round(max(durations), 6),
        "kernel_output_bytes": output_bytes,
        "peak_python_bytes": peak_python_bytes,
        "process_peak_rss_kib": _process_peak_rss_kib(),
        "output_digest": digest,
    }


def _process_peak_rss_kib() -> int | None:
    if resource is None:
        return None
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _arrays_digest(arrays: tuple[np.ndarray, ...]) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        hasher.update(str(values.dtype).encode("ascii"))
        hasher.update(np.asarray(values.shape, dtype="<i8").tobytes())
        hasher.update(values.tobytes(order="C"))
    return hasher.hexdigest()


def _require_exact_keys(
    document: dict[str, Any], expected: set[str], *, field: str
) -> None:
    if set(document) != expected:
        missing = sorted(expected - set(document))
        extra = sorted(set(document) - expected)
        raise RiskPipelineError(
            f"{field} fields differ: missing={missing}, extra={extra}"
        )
