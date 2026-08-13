"""Single adapter around C-owned canonical BC transport and content identity."""

from __future__ import annotations

from arctic_route_planning.contracts import (
    RiskFrame,
    canonical_risk_frame_bytes,
    canonical_risk_id,
    risk_frame_content_digest,
    risk_frame_from_document,
    risk_frame_to_document,
    validate_canonical_risk_id,
)


def with_canonical_risk_id(frame: RiskFrame) -> str:
    """Return the C-owned full content ID for a validated draft frame."""

    return canonical_risk_id(frame)


__all__ = [
    "canonical_risk_frame_bytes",
    "risk_frame_content_digest",
    "risk_frame_from_document",
    "risk_frame_to_document",
    "validate_canonical_risk_id",
    "with_canonical_risk_id",
]
