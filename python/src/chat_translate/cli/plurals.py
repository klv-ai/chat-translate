"""``chat-translate plurals``: add plural categories the target language needs.

Uses the completion provider (``COMPLETION_*`` env). Edits the catalog in place
unless ``--dry-run`` is given. Only languages in ``PLURAL_EXAMPLES`` are
supported. Generated forms should be reviewed by a speaker of the language.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ..icu import missing_plural_categories
from ..plurals import PLURAL_EXAMPLES, Expansion, expand_message
from ._common import completion_provider_from_env, read_catalog, write_catalog


def add_parser(sub: Any) -> None:
    ap: argparse.ArgumentParser = sub.add_parser(
        "plurals",
        help="add plural categories a translated catalog is missing",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("catalog", type=Path, help="translated catalog JSON (edited in place)")
    ap.add_argument("target_lang", help="locale of that catalog, e.g. ru-RU")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    ap.set_defaults(run=run)


def run(args: argparse.Namespace) -> int:
    base = args.target_lang.split("-")[0].lower()
    if base not in PLURAL_EXAMPLES:
        print(f"No plural examples for {args.target_lang}; nothing to do.")
        return 0

    catalog = read_catalog(args.catalog)
    todo = [
        (k, v)
        for k, v in catalog.items()
        if ", plural," in k and missing_plural_categories(k, args.target_lang)
    ]
    if not todo:
        print("No message needs additional plural categories.")
        return 0

    llm = completion_provider_from_env()
    print(f"Expanding {len(todo)} message(s) for {args.target_lang} with {llm.name}…")
    started = time.monotonic()
    results: list[Expansion] = []
    for i, (k, v) in enumerate(todo):
        r = expand_message(k, v, args.target_lang, llm)
        results.append(r)
        flag = "  " if r.ok else "! "
        print(f"{flag}[{i + 1}/{len(todo)}] {k[:52]!r}")
        if r.ok:
            print(f"      + {', '.join(r.added)}: {r.after[:96]}")
        else:
            print(f"      {r.error}")

    ok = [r for r in results if r.ok]
    if not args.dry_run and ok:
        for r in ok:
            catalog[r.key] = r.after
        write_catalog(args.catalog, catalog)
    print(
        f"\n{len(ok)}/{len(todo)} expanded in {time.monotonic() - started:.0f}s"
        + ("" if args.dry_run else f" — written to {args.catalog}")
    )
    print("Generated forms should be reviewed by a speaker of the language.")
    return 0
