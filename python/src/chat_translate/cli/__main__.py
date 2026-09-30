"""Entry point: ``chat-translate <command>`` or ``python -m chat_translate.cli``."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import catalog, plurals, review
from ._common import load_env_file


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="chat-translate",
        description="Translate, review and plural-expand natural-key UI message catalogs.",
    )
    ap.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="dotenv file to load before building providers (default: ./.env if present)",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    catalog.add_parser(sub)
    review.add_parser(sub)
    plurals.add_parser(sub)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.env_file.is_file():
        load_env_file(args.env_file)
    return int(args.run(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
