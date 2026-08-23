#!/usr/bin/env python3
"""Build a research-only risk calibration comparison sidecar."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arctic_route_risk.calibration_shadow import write_shadow_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-index", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Approved root below which the no-clobber sidecar may be created",
    )
    args = parser.parse_args()
    document = write_shadow_comparison(
        frame_index_path=args.frame_index,
        config_path=args.config,
        output_path=args.output,
        approved_output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "artifact_id": document["artifact_id"],
                "content_digest": document["content_digest"],
                "output": str(args.output),
                "publication_status": document["publication_status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
