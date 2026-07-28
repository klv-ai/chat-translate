"""Fill in plural categories an English source cannot express.

English distinguishes two forms, `one` and `other`, so an English-authored ICU
message only ever defines those two. For Spanish or German that is enough. For
Russian it is not:

    one  → 1, 21, 31        "1 файл"
    few  → 2, 3, 4, 22      "2 файла"
    many → 0, 5–20, 25      "5 файлов"

`few` and `many` are reached by ordinary counts, so a message translated from
English renders the wrong case for most numbers — "5 файл" instead of
"5 файлов", which a Russian speaker notices immediately.

The missing forms cannot be *translated*, because there is no source text to
translate: they have to be **inflected** from the forms we already have. That is
a grammar task, not a translation one, so this asks a general instruct model
rather than the translation model — the same reasoning as `review.py`.

Advisory output. Every generated branch is structurally verified, but a human
who reads the language should still look at the result.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .icu import (
    PLURAL_CATEGORIES,
    find_arguments,
    hash_slots_of,
    missing_plural_categories,
    parse_branches,
)
from .provider.base import language_name

#: Representative counts per category, per language. The model needs concrete
#: numbers — naming the CLDR category alone ("give me the `few` form") is not
#: something a general model reliably knows. Only languages whose extra
#: categories are reached by everyday counts are listed; for the rest the
#: English two-form message is already correct.
PLURAL_EXAMPLES: dict[str, dict[str, str]] = {
    "ru": {
        "one": "1, 21, 31, 101",
        "few": "2, 3, 4, 22, 23",
        "many": "0, 5, 6, 11, 12, 20, 25",
        "other": "fractions such as 1.5",
    },
    "uk": {
        "one": "1, 21, 31",
        "few": "2, 3, 4, 22",
        "many": "0, 5–20, 25",
        "other": "fractions",
    },
    "pl": {
        "one": "1",
        "few": "2, 3, 4, 22, 23, 24",
        "many": "0, 5–21, 25",
        "other": "fractions",
    },
    "cs": {
        "one": "1",
        "few": "2, 3, 4",
        "many": "fractions",
        "other": "0, 5, 6, 11, 100",
    },
    "ar": {
        "zero": "0",
        "one": "1",
        "two": "2",
        "few": "3–10, 103–110",
        "many": "11–26, 111–126",
        "other": "100, 101, 102, 1000",
    },
}


@dataclass(slots=True)
class Expansion:
    key: str
    before: str
    after: str
    added: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.added)


def build_expand_prompt(
    source_key: str, branches: list[tuple[str, str]], missing: list[str], target_lang: str
) -> str:
    lang = language_name(target_lang)
    base = target_lang.split("-")[0].lower()
    examples = PLURAL_EXAMPLES.get(base, {})
    rules = "\n".join(
        f"  {cat:5} is used for {examples[cat]}" for cat in examples if cat in examples
    )
    have = "\n".join(f"  {sel}: {msg}" for sel, msg in branches)
    want = ", ".join(missing)
    return (
        f"You are localizing a software user interface into {lang}.\n\n"
        f"{lang} chooses a plural form by the number:\n{rules}\n\n"
        f"This UI message already has these forms:\n{have}\n\n"
        f"(The original English was: {source_key})\n\n"
        f"Give the missing form(s): {want}.\n"
        f"Use the SAME noun as the forms above, put it in the grammatical case "
        f"that form requires, and keep the # exactly where it is — it is the "
        f"number placeholder, not a character to translate.\n\n"
        f"Answer with one line per form, formatted exactly as:\n"
        f"category: text"
    )


def _ask(prompt: str, host: str, model: str, client: httpx.Client) -> str:
    res = client.post(
        f"{host}/api/chat",
        json={
            "model": model,
            "stream": False,
            "keep_alive": "60m",
            # gemma4 buries its answer behind chain-of-thought otherwise; see
            # review.py for the same trap.
            "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "num_predict": 200},
        },
    )
    return str((res.json().get("message") or {}).get("content") or "").strip()


def parse_forms(reply: str, wanted: list[str]) -> dict[str, str]:
    """Pull `category: text` lines out of the reply, keeping only what we asked for."""
    out: dict[str, str] = {}
    for line in reply.splitlines():
        m = re.match(r"\s*\*{0,2}(\w+)\*{0,2}\s*:\s*(.+?)\s*$", line)
        if not m:
            continue
        cat, text = m.group(1).lower(), m.group(2).strip().strip("`\"'")
        if cat in wanted and text:
            out[cat] = text
    return out


def expand_message(
    key: str,
    translated: str,
    target_lang: str,
    *,
    host: str = "http://localhost:11434",
    model: str = "gemma4:12b",
    client: httpx.Client | None = None,
) -> Expansion:
    """Add the categories `target_lang` needs to one translated ICU message."""
    gaps = missing_plural_categories(key, target_lang)
    if not gaps:
        return Expansion(key=key, before=translated, after=translated)

    http = client or httpx.Client(timeout=120.0)
    out = translated
    added: list[str] = []

    for arg in find_arguments(translated):
        if not arg.has_submessages or arg.name not in gaps:
            continue
        branches = parse_branches(arg.raw)
        missing = [c for c in gaps[arg.name] if c not in {s for s, _ in branches}]
        if not missing:
            continue

        prompt = build_expand_prompt(key, branches, missing, target_lang)
        try:
            forms = parse_forms(_ask(prompt, host, model, http), missing)
        except (httpx.HTTPError, ValueError) as exc:
            return Expansion(
                key=key, before=translated, after=translated, error=f"reviewer unreachable: {exc}"
            )
        if not forms:
            return Expansion(
                key=key, before=translated, after=translated, error="model returned no usable forms"
            )

        # Rebuild with the new branches inserted before `other`, which CLDR
        # requires to come last.
        rebuilt: list[tuple[str, str]] = []
        for sel, msg in branches:
            if sel == "other":
                rebuilt.extend((c, forms[c]) for c in missing if c in forms)
            rebuilt.append((sel, msg))
        if not any(sel == "other" for sel, _ in branches):
            rebuilt.extend((c, forms[c]) for c in missing if c in forms)

        body = " ".join(f"{sel} {{{msg}}}" for sel, msg in rebuilt)
        out = out.replace(arg.raw, f"{{{arg.name}, {arg.arg_type}, {body}}}")
        added.extend(forms)

    if not added:
        return Expansion(
            key=key, before=translated, after=translated, error="no categories were added"
        )

    # The whole point is the number slot, so a branch that lost it is unusable.
    src_slots = hash_slots_of(key)
    for name, counts in hash_slots_of(out).items():
        expected = src_slots.get(name, [])
        if expected and any(c != expected[0] for c in counts):
            return Expansion(
                key=key,
                before=translated,
                after=translated,
                error=f"a generated branch lost its # slot: {counts}",
            )

    return Expansion(key=key, before=translated, after=out, added=added)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m chat_translate.plurals",
        description="Add plural categories an English source cannot express.",
    )
    ap.add_argument("catalog", type=Path, help="translated catalog JSON (edited in place)")
    ap.add_argument("target_lang", help="locale of that catalog, e.g. ru-RU")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument(
        "--model",
        default="gemma4:12b",
        help="a GENERAL instruct model — this is inflection, not translation",
    )
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args(argv)

    base = args.target_lang.split("-")[0].lower()
    if base not in PLURAL_EXAMPLES:
        needed = PLURAL_CATEGORIES.get(base, frozenset())
        print(
            f"{args.target_lang} needs {sorted(needed)} and has no example table here.\n"
            f"For languages whose extra categories only fire at very large numbers "
            f"(es, fr, pt) the English two-form message is already correct — "
            f"nothing to do."
        )
        return 0

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    todo = [
        (k, v)
        for k, v in catalog.items()
        if ", plural," in k and missing_plural_categories(k, args.target_lang)
    ]
    if not todo:
        print("No message needs additional plural categories.")
        return 0

    print(f"Expanding {len(todo)} message(s) for {args.target_lang} with {args.model}…")
    started = time.monotonic()
    http = httpx.Client(timeout=120.0)
    results = []
    for i, (k, v) in enumerate(todo):
        r = expand_message(k, v, args.target_lang, host=args.host, model=args.model, client=http)
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
        args.catalog.write_text(
            json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"\n{len(ok)}/{len(todo)} expanded in {time.monotonic() - started:.0f}s"
        + ("" if args.dry_run else f" — written to {args.catalog}")
    )
    print("These are generated grammar, not translation. Have a speaker review them.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
