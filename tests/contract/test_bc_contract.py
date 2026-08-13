from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from arctic_route_planning.contracts import (
    canonical_risk_frame_bytes,
    risk_frame_from_document,
    risk_frame_to_document,
)
from jsonschema import Draft202012Validator, FormatChecker

from arctic_route_risk import BInputEnvelope, RiskBuildRequest, RiskBuildService

SCHEMA = Path(__file__).parents[3] / "work_package_c/schemas/risk-frame-v2.schema.json"


def test_b_output_passes_c_schema_and_canonical_roundtrip(formal_fixture) -> None:
    envelope = BInputEnvelope.from_prepared_window(
        run_context=formal_fixture.context,
        prepared_window=formal_fixture.prepared,
        generation_id=formal_fixture.prepared.generation_id,
        knowledge_as_of=formal_fixture.prepared.as_of_time,
    )
    frame = RiskBuildService(
        utc_now=lambda: datetime(2026, 8, 2, tzinfo=UTC)
    ).build_window(
        RiskBuildRequest(envelope=envelope, target_bbox=formal_fixture.bbox)
    )[0]
    document = risk_frame_to_document(frame)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    validator.validate(document)
    decoded = risk_frame_from_document(document)

    assert decoded.risk_id == frame.risk_id
    assert canonical_risk_frame_bytes(decoded) == canonical_risk_frame_bytes(frame)
