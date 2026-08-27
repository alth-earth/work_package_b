"""Bounded measurements around the real formal RiskBuildService.

This module changes neither production defaults nor risk semantics.  It exists
only to make fixed-grid research runs repeatable and explicitly labelled.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from arctic_route_planning.contracts import RiskFrame

from arctic_route_risk.bc_codec import risk_frame_to_document
from arctic_route_risk.config import DemoRiskModelConfig
from arctic_route_risk.context import BInputEnvelope
from arctic_route_risk.grid_experiments import GridExperimentProfile
from arctic_route_risk.service import RiskBuildRequest, RiskBuildService


@dataclass(frozen=True, slots=True)
class FormalGridProfileResult:
    """One measured formal B build plus its immutable output frames."""

    summary: dict[str, Any]
    frames: tuple[RiskFrame, ...]


class PeakRssSampler:
    """Sample current process RSS without starting another worker process."""

    def __init__(self, *, interval_seconds: float = 0.02) -> None:
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._peak_kib = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> PeakRssSampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    @property
    def peak_kib(self) -> int:
        return self._peak_kib

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        status = Path("/proc/self/status")
        try:
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    self._peak_kib = max(self._peak_kib, int(line.split()[1]))
                    return
        except (FileNotFoundError, OSError, ValueError):
            return


def _hard_reason_counter(frame: RiskFrame) -> Counter[str]:
    if "hard_reason" in frame.payload:
        reasons = np.asarray(frame.payload["hard_reason"].values, dtype=np.str_)
        return Counter(str(value) for value in reasons.ravel() if str(value))
    attr_counts = frame.payload.attrs.get("hard_reason_counts", {})
    if isinstance(attr_counts, dict):
        return Counter({str(key): int(value) for key, value in attr_counts.items()})
    return Counter()


def build_formal_grid_profile(
    *,
    envelope: BInputEnvelope,
    target_bbox: tuple[float, float, float, float],
    model_config: DemoRiskModelConfig,
    profile: GridExperimentProfile,
    utc_now: Callable[[], datetime],
) -> FormalGridProfileResult:
    """Run the real formal B builder for one experimental fixed grid."""

    request = RiskBuildRequest(
        envelope=envelope,
        target_bbox=target_bbox,
        grid_config=profile.target_grid(),
        model_config=model_config,
    )
    started = time.perf_counter()
    with PeakRssSampler() as memory:
        frames = RiskBuildService(utc_now=utc_now).build_window(request)
    elapsed = time.perf_counter() - started
    return FormalGridProfileResult(
        summary=summarize_formal_frames(
            frames,
            profile=profile,
            elapsed_seconds=elapsed,
            peak_rss_kib=memory.peak_kib,
        ),
        frames=frames,
    )


def summarize_formal_frames(
    frames: Sequence[RiskFrame],
    *,
    profile: GridExperimentProfile,
    elapsed_seconds: float,
    peak_rss_kib: int,
) -> dict[str, Any]:
    """Summarize shape, transport size and risk/hard distributions."""

    if not frames:
        raise ValueError("formal grid experiment requires at least one RiskFrame")
    rows = int(frames[0].payload.sizes["latitude"])
    cols = int(frames[0].payload.sizes["longitude"])
    level_counts: Counter[int] = Counter()
    hard_reason_counts: Counter[str] = Counter()
    finite_scores: list[np.ndarray] = []
    frame_distributions: list[dict[str, Any]] = []
    encoded_bytes = 0
    for frame in frames:
        levels = np.asarray(frame.payload["risk_level"].values, dtype=np.uint8)
        scores = np.asarray(frame.payload["risk_score"].values, dtype=np.float64)
        per_frame_levels = Counter(int(value) for value in levels.ravel())
        per_frame_reasons = _hard_reason_counter(frame)
        level_counts.update(per_frame_levels)
        hard_reason_counts.update(per_frame_reasons)
        finite = scores[np.isfinite(scores)]
        if finite.size:
            finite_scores.append(finite)
        document = risk_frame_to_document(frame)
        encoded_bytes += len(
            json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        frame_distributions.append(
            {
                "valid_time": frame.valid_time.isoformat().replace("+00:00", "Z"),
                "risk_levels": {
                    str(level): per_frame_levels.get(level, 0) for level in range(1, 6)
                },
                "hard_reasons": dict(sorted(per_frame_reasons.items())),
                "finite_risk_score_min": float(np.min(finite)) if finite.size else None,
                "finite_risk_score_max": float(np.max(finite)) if finite.size else None,
                "finite_risk_score_mean": float(np.mean(finite)) if finite.size else None,
            }
        )
    combined = np.concatenate(finite_scores) if finite_scores else np.array([], dtype=float)
    return {
        "name": profile.name,
        "latitude_step_degrees": profile.latitude_step_degrees,
        "longitude_step_degrees": profile.longitude_step_degrees,
        "rows": rows,
        "cols": cols,
        "cells_per_frame": rows * cols,
        "frame_count": len(frames),
        "total_generation_seconds": round(elapsed_seconds, 6),
        "peak_sampled_rss_kib": peak_rss_kib,
        "risk_frame_json_bytes": encoded_bytes,
        "risk_levels": {str(level): level_counts.get(level, 0) for level in range(1, 6)},
        "hard_reasons": dict(sorted(hard_reason_counts.items())),
        "finite_risk_score_min": float(np.min(combined)) if combined.size else None,
        "finite_risk_score_max": float(np.max(combined)) if combined.size else None,
        "finite_risk_score_mean": float(np.mean(combined)) if combined.size else None,
        "frame_distributions": frame_distributions,
        "source_provenance": sorted({frame.provenance.value for frame in frames}),
        "schema_versions": sorted({frame.schema_version for frame in frames}),
        "model_config_digests": sorted({frame.model_config_digest for frame in frames}),
    }


def write_frame_documents(frames: Sequence[RiskFrame], output: Path) -> None:
    """Write experimental transport documents without publishing a BC window."""

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "b.formal-grid-experiment-frames.v1",
        "status": "EXPERIMENTAL",
        "published": False,
        "frames": [risk_frame_to_document(frame) for frame in frames],
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
