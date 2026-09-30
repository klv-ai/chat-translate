"""``chat-translate catalog``: translate a natural-key catalog into one locale.

Resumes by default: keys already present in the output file are skipped.
Only verified translations are written; failed keys are listed and left out,
so a rerun retries them. Progress is checkpointed to the output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ..catalog import EntryResult, translate_catalog
from ._common import read_catalog, translation_provider_from_env, write_catalog


def add_parser(sub: Any) -> None:
    ap: argparse.ArgumentParser = sub.add_parser(
        "catalog",
        help="translate a natural-key UI message catalog",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("source", type=Path, help="source catalog JSON (keys are source strings)")
    ap.add_argument("target_lang", help="target locale, e.g. es-ES")
    ap.add_argument(
        "--out", type=Path, help="output JSON (default: <target_lang>.json beside source)"
    )
    ap.add_argument("--source-lang", default="en", help="language of the keys (default: en)")
    ap.add_argument("--limit", type=int, help="translate only the first N pending keys")
    ap.add_argument(
        "--attempts", type=int, default=3, help="tries per fragment on a retryable error"
    )
    ap.add_argument(
        "--protect",
        type=Path,
        help='JSON file of terms to keep verbatim: an array, or {"terms": [...]}',
    )
    ap.add_argument(
        "--redo",
        action="store_true",
        help="retranslate keys already present in the output instead of resuming",
    )
    ap.set_defaults(run=run)


def load_protected_terms(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    terms = raw.get("terms", []) if isinstance(raw, dict) else raw
    if not isinstance(terms, list):
        raise SystemExit(f'{path}: expected an array or {{"terms": [...]}}')
    return [str(t) for t in terms]


def run(args: argparse.Namespace) -> int:
    source = read_catalog(args.source)
    out_path: Path = args.out or args.source.with_name(f"{args.target_lang}.json")
    existing = read_catalog(out_path) if out_path.exists() else {}

    pending = [k for k in source if args.redo or k not in existing]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"Nothing to do — {out_path.name} already covers all {len(source)} keys.")
        return 0

    protected = load_protected_terms(args.protect) if args.protect else []
    if protected:
        print(f"Protecting {len(protected)} term(s) from translation.")

    provider = translation_provider_from_env()
    if not provider.health_check():
        print(f"Translation provider {provider.name!r} is not available.", file=sys.stderr)
        return 2

    total = len(pending)
    print(f"Translating {total} key(s) to {args.target_lang} with {provider.name}…")

    def progress(i: int, entry: EntryResult) -> None:
        flag = "!" if not entry.ok else " "
        print(f"  {flag} [{i + 1}/{total}] {entry.key[:60]!r} -> {entry.value[:60]!r}", flush=True)

    def merged(done: dict[str, str]) -> dict[str, str]:
        combined = {**existing, **done}
        return {k: combined[k] for k in source if k in combined}

    result = translate_catalog(
        pending,
        provider,
        args.target_lang,
        source_lang=args.source_lang,
        protected=protected,
        attempts=args.attempts,
        on_progress=progress,
        on_checkpoint=lambda done: write_catalog(out_path, merged(done)),
    )

    written = merged(result.translated)
    rate = total / result.elapsed_seconds if result.elapsed_seconds else 0.0
    print(
        f"\nWrote {len(written)}/{len(source)} keys to {out_path} "
        f"({result.elapsed_seconds:.0f}s, {rate:.2f} keys/s)."
    )
    if result.failures:
        print(f"\n{len(result.failures)} left untranslated — rerun to retry them:")
        for e in result.failures[:20]:
            print(f"  {e.key[:70]!r}: {e.error}")
    if result.flagged:
        print(f"\n{len(result.flagged)} need a plural review for {args.target_lang}:")
        for e in result.flagged[:20]:
            print(f"  {e.key[:70]!r}: {e.notes[0]}")
    return 0
