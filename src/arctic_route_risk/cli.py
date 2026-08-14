"""Small operational CLI; orchestration stays outside B."""

from __future__ import annotations

import argparse
import json
import sys

from arctic_route_risk import __version__
from arctic_route_risk.errors import RiskPipelineError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arctic-route-risk",
        description=(
            "工作包 B：消费经验证的 A PreparedWindow，并发布逐小时 bc.risk-frame.v2。"
            "当前规则为 demo_unvalidated，不可用于真实导航。"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")
    intake = subparsers.add_parser(
        "model-intake",
        help="受限接收并转换交付的 legacy CNN 权重（不执行 ZIP 内容）",
    )
    intake.add_argument("--zip", required=True, dest="zip_path", help="原始模型 ZIP 路径")
    intake.add_argument(
        "--policy",
        default="legacy_cnn_one_step_v1",
        help="策略 JSON 路径或内置策略 ID",
    )
    intake.add_argument(
        "--output",
        default="models/legacy_cnn_one_step_v1",
        help="转换资产目录（默认在 B 仓库内）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "model-intake":
        from arctic_route_risk.modeling import intake_legacy_cnn_zip

        try:
            manifest = intake_legacy_cnn_zip(
                arguments.zip_path,
                policy=arguments.policy,
                output_dir=arguments.output,
            )
        except RiskPipelineError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(manifest.to_document(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
