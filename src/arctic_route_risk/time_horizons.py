"""Mentor-aligned time horizon labels for B risk products."""

from __future__ import annotations

import math


def stage_for_offset(
    offset_hours: float,
    *,
    total_route_hours: float | None = None,
) -> str:
    """Return the project stage label for a forecast offset in hours."""

    if not math.isfinite(float(offset_hours)):
        raise ValueError("offset_hours must be finite")
    offset = float(offset_hours)
    if offset < 0:
        return "outside_supported_window"
    if offset <= 2:
        return "execution_high_confidence_0_2h"
    if offset <= 4:
        return "execution_recommended_2_4h"
    if offset <= 6:
        return "execution_predicted_4_6h"
    if offset <= 24:
        return "rolling_dynamic_0_24h"
    if offset <= 72:
        return "main_corridor_24_72h"
    if total_route_hours is not None and offset >= float(total_route_hours):
        return "full_route_reference"
    return "full_route_reference"


def mentor_required_offsets(total_route_hours: float) -> tuple[int, ...]:
    """Return representative offsets for the mentor-required time sequence."""

    if not math.isfinite(float(total_route_hours)) or total_route_hours < 0:
        raise ValueError("total_route_hours must be a non-negative finite number")
    route_hours = round(float(total_route_hours))
    offsets: list[int] = list(range(0, min(24, route_hours) + 1))
    if route_hours > 24:
        offsets.extend(range(30, min(72, route_hours) + 1, 6))
    if route_hours > 72:
        offsets.append(route_hours)
    return tuple(dict.fromkeys(offsets))
