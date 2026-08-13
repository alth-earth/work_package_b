"""Small operational CLI; orchestration stays outside B."""

from __future__ import annotations

import argparse

from arctic_route_risk import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arctic-route-risk",
        description=(
            "工作包 B：消费经验证的 A PreparedWindow，并发布逐小时 bc.risk-frame.v2。"
            "当前规则为 demo_unvalidated，不可用于真实导航。"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
