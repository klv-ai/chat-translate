"""``chat-translate review``: flag translations with the wrong meaning.

Uses the completion provider (``COMPLETION_*`` env). Entries whose value equals
the key are skipped. Advisory only; the catalog is not modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..review import Verdict, review_catalog
from ._common import completion_provider_from_env, read_catalog


def add_parser(sub: Any) -> None:
    ap: argparse.ArgumentParser = sub.add_parser(
        "review",
        help="review a translated catalog for meaning",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("catalog", type=Path, help="translated catalog JSON")
    ap.add_argument("target_lang", help="locale of that catalog, e.g. es-ES")
    ap.add_argument(
        "--context", type=Path, help="text file describing the product, given to the reviewer"
    )
    ap.add_argument("--limit", type=int, help="review only the first N entries")
    ap.add_argument("--out", type=Path, help="write flagged entries as JSON")
    ap.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    catalog = read_catalog(args.catalog)
    context = args.context.read_text(encoding="utf-8").strip() if args.context else ""
    pairs = [(k, v) for k, v in catalog.items() if v and v != k]
    if args.limit:
        pairs = pairs[: args.limit]

    llm = completion_provider_from_env()
    total = len(pairs)
    print(f"Reviewing {total} {args.target_lang} entries with {llm.name}…")

    def progress(i: int, v: Verdict) -> None:
        if v.flagged:
            print(f"  ! [{i + 1}/{total}] {v.key[:44]!r} -> {v.translation[:34]!r}")
            print(f"      {v.reason[:100]}")
        elif (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{total}]…", flush=True)

    result = review_catalog(pairs, args.target_lang, llm, context=context, on_progress=progress)

    rate = total / result.elapsed_seconds if result.elapsed_seconds else 0.0
    print(
        f"\n{len(result.flagged)}/{total} flagged for review "
        f"({result.elapsed_seconds:.0f}s, {rate:.2f} entries/s)."
    )
    if result.unreviewed:
        print(
            f"WARNING: {len(result.unreviewed)}/{total} were NOT reviewed — "
            f"{result.unreviewed[0].reason}"
        )
    if args.out:
        args.out.write_text(
            json.dumps(
                {v.key: {"translation": v.translation, "reason": v.reason} for v in result.flagged},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Wrote flagged entries to {args.out}")
    return 0
