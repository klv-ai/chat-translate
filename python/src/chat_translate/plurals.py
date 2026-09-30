"""Add plural categories a translated ICU message is missing for its target language.

A message authored with only ``one``/``other`` lacks categories such as ``few``
and ``many`` that some languages (e.g. Russian, Polish, Arabic) need. This asks
a :class:`CompletionProvider` to inflect the missing forms from the existing
branches, using the example counts in :data:`PLURAL_EXAMPLES`, and inserts them
before ``other``. Generated branches are checked for their ``#`` slot; the text
itself is unverified and should be reviewed by a speaker of the language.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .icu import (
    find_arguments,
    hash_slots_of,
    missing_plural_categories,
    parse_branches,
)
from .provider.base import CompletionError, CompletionProvider, language_name

#: Example counts per plural category, per language, included in the prompt.
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

#: Token budget for one reply (a few ``category: text`` lines).
_EXPAND_MAX_TOKENS = 200


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
    rules = "\n".join(f"  {cat:5} is used for {counts}" for cat, counts in examples.items())
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
    llm: CompletionProvider,
) -> Expansion:
    """Add the plural categories *target_lang* needs to one translated message.

    *key* is the source message, used to find the gaps. Returns the message
    unchanged, with ``error`` set, if the provider fails, yields no usable
    forms, or a generated branch has a different ``#`` count from the source.
    """
    gaps = missing_plural_categories(key, target_lang)
    if not gaps:
        return Expansion(key=key, before=translated, after=translated)

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
            forms = parse_forms(llm.complete(prompt, max_tokens=_EXPAND_MAX_TOKENS), missing)
        except CompletionError as exc:
            return Expansion(
                key=key, before=translated, after=translated, error=f"model unreachable: {exc}"
            )
        if not forms:
            return Expansion(
                key=key, before=translated, after=translated, error="model returned no usable forms"
            )

        # CLDR requires `other` last, so new branches go before it.
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
