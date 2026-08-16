"""Realtime single-factor risk layers for operational situation awareness.

Single-factor products are intentionally lightweight: each factor uses the
latest visible valid time available for that factor, then exports a processed
0..1 risk layer.  They are display and monitoring products, not the formal
multi-horizon planning contract consumed by C.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from arctic_route_risk.config import RiskComponentConfig
from arctic_route_risk.errors import CoverageError, RiskPipelineError
from arctic_route_risk.service import (
    _COMPONENT_INPUTS,
    _VARIABLES,
    RiskBuildRequest,
    _resolve_field,
    _risk_component,
)


@dataclass(frozen=True, slots=True)
class SingleFactorRiskLayer:
    """One processed realtime risk layer for a single environmental factor."""

    factor_id: str
    valid_time: datetime
    issue_time: datetime
    collect_time: datetime
    dataset: xr.Dataset
    source_data_types: tuple[str, ...]
    source_data_ids: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        risk = np.asarray(self.dataset["risk_score"].values, dtype=np.float64)
        sea_mask = np.isfinite(risk)
        high_mask = sea_mask & (risk >= 0.6)
        very_high_mask = sea_mask & (risk >= 0.8)
        return {
            "schema_version": "b.single-factor-risk-summary.v1",
            "factor_id": self.factor_id,
            "issue_time": _iso_z(self.issue_time),
            "valid_time": _iso_z(self.valid_time),
            "collect_time": _iso_z(self.collect_time),
            "source_data_types": list(self.source_data_types),
            "source_data_ids": list(self.source_data_ids),
            "cell_count": int(risk.size),
            "valid_cell_count": int(sea_mask.sum()),
            "high_risk_cell_count": int(high_mask.sum()),
            "very_high_risk_cell_count": int(very_high_mask.sum()),
            "mean_risk": _finite_or_none(np.nanmean(risk)) if sea_mask.any() else None,
            "max_risk": _finite_or_none(np.nanmax(risk)) if sea_mask.any() else None,
            "high_risk_fraction": (
                float(high_mask.sum() / sea_mask.sum()) if sea_mask.any() else None
            ),
            "very_high_risk_fraction": (
                float(very_high_mask.sum() / sea_mask.sum()) if sea_mask.any() else None
            ),
        }


@dataclass(frozen=True, slots=True)
class SingleFactorOutputPaths:
    """Files written for one realtime single-factor output batch."""

    summary_json: Path
    factor_json: tuple[Path, ...]
    factor_netcdf: tuple[Path, ...]
    factor_png: tuple[Path, ...]


def build_realtime_single_factor_layers(
    request: RiskBuildRequest,
    *,
    utc_now: Callable[[], datetime] | None = None,
) -> tuple[SingleFactorRiskLayer, ...]:
    """Build latest-valid-time single-factor risk layers.

    Each factor is derived from official A data already admitted into the B
    input envelope.  For a factor, ``T0`` is the latest valid time at or before
    the envelope knowledge cutoff for that factor's source data.  The raw values
    are regridded, converted through the same configured component transform
    used by the rule baseline, clipped to ``[0, 1]``, and masked over land.
    """

    request = _snapshot_request(request)
    collect_time = _ensure_utc((utc_now or (lambda: datetime.now(UTC)))())
    latitude, longitude, grid_id = request.grid_config.realize(request.target_bbox)
    layers: list[SingleFactorRiskLayer] = []
    for component in request.model_config.components:
        data_types = _data_types_for_component(component)
        valid_time = _latest_common_valid_time(request, data_types)
        resolved = {
            data_type: _resolve_field(
                data_type=data_type,
                frames=request.envelope.frames[data_type],
                target_time=valid_time,
                knowledge_as_of=request.envelope.knowledge_as_of,
                latitude=latitude,
                longitude=longitude,
                target_bbox=request.target_bbox,
                model_config=request.model_config,
            )
            for data_type in data_types
        }
        land = _resolve_field(
            data_type="land_sea_mask",
            frames=request.envelope.frames["land_sea_mask"],
            target_time=valid_time,
            knowledge_as_of=request.envelope.knowledge_as_of,
            latitude=latitude,
            longitude=longitude,
            target_bbox=request.target_bbox,
            model_config=request.model_config,
        )
        arrays = {
            variable: values
            for field in (*resolved.values(), land)
            for variable, values in field.variables.items()
        }
        risk = np.clip(_risk_component(arrays, component), 0.0, 1.0)
        land_sea = np.asarray(arrays["land_sea_mask"], dtype=np.float64)
        hard = ~np.isfinite(land_sea) | (
            land_sea < request.model_config.land_sea_mask_land_threshold
        )
        source_confidence = min(field.confidence for field in (*resolved.values(), land))
        confidence = np.where(np.isfinite(risk) & ~hard, source_confidence, 0.0)
        risk = np.where(hard, np.nan, risk).astype(np.float32)
        level = np.zeros(risk.shape, dtype=np.uint8)
        finite = np.isfinite(risk)
        level[finite] = np.clip(np.floor(risk[finite] * 5) + 1, 1, 5).astype(np.uint8)
        support_frames = tuple(
            frame
            for field in (*resolved.values(), land)
            for frame in field.support_frames
        )
        issue_time = max(frame.record.issue_time for frame in support_frames)
        source_data_ids = tuple(sorted({frame.record.data_id for frame in support_frames}))
        dataset = xr.Dataset(
            data_vars={
                "risk_score": (("latitude", "longitude"), risk),
                "risk_level": (("latitude", "longitude"), level),
                "hard_mask": (("latitude", "longitude"), hard.astype(np.bool_)),
                "confidence": (("latitude", "longitude"), confidence.astype(np.float32)),
            },
            coords={"latitude": latitude, "longitude": longitude},
            attrs={
                "schema_version": "b.single-factor-risk-layer.v1",
                "factor_id": component.component_id,
                "issue_time": _iso_z(issue_time),
                "valid_time": _iso_z(valid_time),
                "collect_time": _iso_z(collect_time),
                "route_id": request.envelope.run_context.corridor_id,
                "run_id": request.envelope.run_context.run_id,
                "grid_id": grid_id,
                "crs": "EPSG:4326",
                "risk_role": "realtime_situation_awareness",
                "planning_contract": "non_authoritative_display_layer",
                "risk_transform": component.transform,
                "risk_lower": component.lower,
                "risk_upper": component.upper,
                "source_data_types": ",".join(data_types),
                "source_data_ids": ",".join(source_data_ids),
            },
        )
        layers.append(
            SingleFactorRiskLayer(
                factor_id=component.component_id,
                valid_time=valid_time,
                issue_time=issue_time,
                collect_time=collect_time,
                dataset=dataset,
                source_data_types=data_types,
                source_data_ids=source_data_ids,
            )
        )
    return tuple(layers)


def write_realtime_single_factor_outputs(
    request: RiskBuildRequest,
    output_dir: str | Path,
    *,
    utc_now: Callable[[], datetime] | None = None,
    write_netcdf: bool = False,
    write_png: bool = False,
) -> SingleFactorOutputPaths:
    """Write realtime single-factor data products and a batch summary.

    JSON is always written.  NetCDF and PNG are optional so the formal package
    does not require output or plotting extras during server-side risk builds.
    """

    layers = build_realtime_single_factor_layers(request, utc_now=utc_now)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    factor_json: list[Path] = []
    factor_netcdf: list[Path] = []
    factor_png: list[Path] = []
    for layer in layers:
        stem = f"{layer.factor_id}_{_stamp(layer.valid_time)}"
        json_path = root / f"{stem}.json"
        json_path.write_text(
            json.dumps(layer.summary(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        factor_json.append(json_path)
        if write_netcdf:
            netcdf_path = root / f"{stem}.nc"
            try:
                layer.dataset.to_netcdf(netcdf_path, engine="h5netcdf")
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
                raise RuntimeError(
                    "h5netcdf is required when write_netcdf=True; "
                    "install the single-factor-output extra"
                ) from exc
            factor_netcdf.append(netcdf_path)
        if write_png:
            png_path = root / f"{stem}.png"
            _write_png(layer, png_path)
            factor_png.append(png_path)
    summary_path = root / "single_factor_realtime_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": "b.single-factor-risk-batch.v1",
                "layer_count": len(layers),
                "generated_at": _iso_z(_ensure_utc((utc_now or (lambda: datetime.now(UTC)))())),
                "layers": [layer.summary() for layer in layers],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return SingleFactorOutputPaths(
        summary_json=summary_path,
        factor_json=tuple(factor_json),
        factor_netcdf=tuple(factor_netcdf),
        factor_png=tuple(factor_png),
    )


def _snapshot_request(request: RiskBuildRequest) -> RiskBuildRequest:
    return RiskBuildRequest(
        envelope=request.envelope.verified_build_snapshot(),
        target_bbox=request.target_bbox,
        grid_config=request.grid_config,
        model_config=request.model_config,
    )


def _data_types_for_component(component: RiskComponentConfig) -> tuple[str, ...]:
    variable_to_data_type = {
        variable: data_type
        for data_type, variables in _VARIABLES.items()
        for variable in variables
    }
    data_types = []
    for variable in _COMPONENT_INPUTS[component.component_id]:
        data_type = variable_to_data_type.get(variable)
        if data_type is None:
            raise RiskPipelineError(f"single-factor source variable is unknown: {variable}")
        data_types.append(data_type)
    return tuple(dict.fromkeys(data_types))


def _latest_common_valid_time(
    request: RiskBuildRequest,
    data_types: Iterable[str],
) -> datetime:
    latest: list[datetime] = []
    for data_type in data_types:
        frames = tuple(
            frame
            for frame in request.envelope.frames[data_type]
            if frame.record.issue_time <= request.envelope.knowledge_as_of
            and frame.record.valid_time <= request.envelope.knowledge_as_of
        )
        if not frames:
            raise CoverageError(
                f"forecast_coverage_insufficient: no realtime {data_type} support at T0"
            )
        latest.append(max(frame.record.valid_time for frame in frames))
    return min(latest)


def _write_png(layer: SingleFactorRiskLayer, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "matplotlib is required when write_png=True; install the visualization extra"
        ) from exc
    risk = layer.dataset["risk_score"]
    figure, axes = plt.subplots(figsize=(8, 5), constrained_layout=True)
    image = axes.imshow(
        risk.values,
        origin="lower",
        extent=[
            float(risk.longitude.min()),
            float(risk.longitude.max()),
            float(risk.latitude.min()),
            float(risk.latitude.max()),
        ],
        vmin=0.0,
        vmax=1.0,
        cmap="YlOrRd",
        interpolation="nearest",
    )
    axes.set_title(f"{layer.factor_id} realtime risk | {layer.valid_time:%Y%m%dT%HZ}")
    axes.set_xlabel("Longitude")
    axes.set_ylabel("Latitude")
    figure.colorbar(image, ax=axes, label="Single-factor risk (0-1)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _ensure_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise RiskPipelineError("time must be timezone-aware UTC")
    return value.astimezone(UTC)


def _iso_z(value: datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _stamp(value: datetime) -> str:
    return _ensure_utc(value).strftime("%Y%m%dT%HZ")


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None
