"""Plot helpers for Work Package B risk products.

The C contract consumes RiskFrame data, not images. These helpers are a
presentation layer for reviews, reports and operator-facing checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def render_color_risk_map(
    frame: Any,
    output_path: str | Path,
    *,
    title: str | None = None,
) -> Path:
    """Render a continuous color risk map from one RiskFrame."""

    plt = _pyplot()
    lat, lon, risk, hard = _extract_frame_arrays(frame)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    ax.set_facecolor("#f7fbff")
    ocean = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_where(hard | ~np.isfinite(risk), risk),
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        shading="nearest",
        alpha=0.95,
    )
    _draw_land(ax, lon, lat, hard)
    _format_axes(ax, title or _default_title(frame, "Comprehensive Risk Map"))
    colorbar = fig.colorbar(ocean, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Comprehensive risk (0-1)")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def render_binary_risk_map(
    frame: Any,
    output_path: str | Path,
    *,
    threshold: float = 0.6,
    title: str | None = None,
) -> Path:
    """Render a black-white obstacle-style risk map from one RiskFrame."""

    plt = _pyplot()
    lat, lon, risk, hard = _extract_frame_arrays(frame)
    blocked = hard | (np.nan_to_num(risk, nan=1.0) >= threshold)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    ax.imshow(
        blocked.astype(float),
        extent=(float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())),
        origin="lower",
        cmap="gray_r",
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )
    _format_axes(ax, title or _default_title(frame, "Comprehensive Risk Binary Map"))
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output


def render_risk_maps(
    frame: Any,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
    binary_threshold: float = 0.6,
) -> dict[str, Path]:
    """Render both color and black-white maps for one RiskFrame."""

    directory = Path(output_dir)
    name = prefix or _safe_name(_default_title(frame, "risk_frame"))
    return {
        "color": render_color_risk_map(frame, directory / f"{name}_color.png"),
        "binary": render_binary_risk_map(
            frame,
            directory / f"{name}_binary.png",
            threshold=binary_threshold,
        ),
    }


def _extract_frame_arrays(frame: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    payload = frame.payload
    lat = np.asarray(payload.coords["latitude"].values, dtype=np.float64)
    lon = np.asarray(payload.coords["longitude"].values, dtype=np.float64)
    risk = np.asarray(payload["risk_score"].values, dtype=np.float64)
    hard = np.asarray(payload["hard_mask"].values, dtype=bool)
    return lat, lon, risk, hard


def _draw_land(ax: Any, lon: np.ndarray, lat: np.ndarray, hard: np.ndarray) -> None:
    ax.contourf(
        lon,
        lat,
        hard.astype(float),
        levels=[0.5, 1.5],
        colors=["#e7dfcf"],
        alpha=1.0,
    )
    ax.contour(
        lon,
        lat,
        hard.astype(float),
        levels=[0.5],
        colors=["#8c8579"],
        linewidths=0.8,
    )


def _format_axes(ax: Any, title: str) -> None:
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="#ffffff", linewidth=0.35, alpha=0.28)


def _default_title(frame: Any, label: str) -> str:
    valid_time = frame.valid_time.isoformat().replace("+00:00", "Z")
    return f"{frame.corridor_id} | {label} | {valid_time}"


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)[:160]


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for B plotting helpers; install project dependencies"
        ) from exc
    return plt
