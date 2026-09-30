"""Meaning review of translated UI strings by a general instruct model.

Each (source, translation) pair is sent to a :class:`CompletionProvider` with an
optional description of the product, and the reply is parsed into a
:class:`Verdict`. Output is advisory: a reviewer that fails or does not answer
yields ``unavailable=True``, never a flag.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .provider.base import CompletionError, CompletionProvider, language_name

#: Token budget for a verdict: one word plus a short sentence.
_REVIEW_MAX_TOKENS = 96


@dataclass(slots=True)
class Verdict:
    key: str
    translation: str
    ok: bool
    reason: str = ""
    #: True when the reviewer did not return a verdict (error or unparseable reply).
    unavailable: bool = False

    @property
    def flagged(self) -> bool:
        return not self.ok and not self.unavailable


@dataclass(slots=True)
class ReviewResult:
    verdicts: list[Verdict]
    elapsed_seconds: float
    flagged: list[Verdict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.flagged = [v for v in self.verdicts if v.flagged]

    @property
    def unreviewed(self) -> list[Verdict]:
        return [v for v in self.verdicts if v.unavailable]


_VERDICT_RE = re.compile(r"\b(OK|WRONG)\b", re.IGNORECASE)


def build_review_prompt(source: str, translation: str, target_lang: str, context: str = "") -> str:
    """Prompt asking for an ``OK``/``WRONG`` verdict first, then a reason.

    *context*, when non-empty, is placed before the instructions to describe
    the product the strings belong to.
    """
    lang = language_name(target_lang)
    preamble = f"{context.strip()}\n\n" if context.strip() else ""
    return (
        f"{preamble}"
        f"You are reviewing the {lang} localization of this product's user "
        f"interface. Below is one English UI string and its proposed {lang} "
        f"translation.\n\n"
        f"English:     {source}\n"
        f"{lang}: {translation}\n\n"
        f"Flag it ONLY if a user would be misled. That means:\n"
        f'  - the wrong SENSE of an ambiguous word (e.g. English "Resume" '
        f'translated as a CV rather than as "continue")\n'
        f"  - the wrong part of speech (a verb where the UI needs a noun)\n"
        f"  - a product or brand name that was translated instead of kept\n"
        f"  - meaning that is plainly wrong or reversed\n\n"
        f"Do NOT flag style. A translation that is merely longer, more formal, "
        f"or not your preferred wording is OK. If it conveys the right meaning, "
        f"it is OK. When unsure, answer OK.\n\n"
        f"Answer with exactly one word first — OK or WRONG — then, only if "
        f"WRONG, one short sentence naming the wrong sense and the right one."
    )


def review_entry(
    source: str,
    translation: str,
    target_lang: str,
    llm: CompletionProvider,
    *,
    context: str = "",
) -> Verdict:
    """Review one entry.

    Returns ``ok=True`` for an ``OK`` reply, ``ok=False`` with the reviewer's
    reason for ``WRONG``, and ``ok=True, unavailable=True`` when the provider
    raises :class:`CompletionError` or the reply contains neither word.
    """
    prompt = build_review_prompt(source, translation, target_lang, context)
    try:
        body = llm.complete(prompt, max_tokens=_REVIEW_MAX_TOKENS)
    except CompletionError as exc:
        return Verdict(
            key=source,
            translation=translation,
            ok=True,
            unavailable=True,
            reason=f"reviewer unreachable: {exc}",
        )

    match = _VERDICT_RE.search(body)
    if not match:
        return Verdict(
            key=source,
            translation=translation,
            ok=True,
            unavailable=True,
            reason=f"no verdict in reply: {body[:60]!r}" if body else "empty reply",
        )
    if match.group(1).upper() == "OK":
        return Verdict(key=source, translation=translation, ok=True)
    reason = body[match.end() :].strip(" .:-\n") or "flagged without a reason"
    return Verdict(key=source, translation=translation, ok=False, reason=reason)


def review_catalog(
    entries: Iterable[tuple[str, str]],
    target_lang: str,
    llm: CompletionProvider,
    *,
    context: str = "",
    on_progress: Callable[[int, Verdict], None] | None = None,
) -> ReviewResult:
    """Review each (source, translation) pair in order with :func:`review_entry`."""
    started = time.monotonic()
    verdicts: list[Verdict] = []
    for i, (source, translation) in enumerate(entries):
        verdict = review_entry(source, translation, target_lang, llm, context=context)
        verdicts.append(verdict)
        if on_progress:
            on_progress(i, verdict)
    return ReviewResult(verdicts=verdicts, elapsed_seconds=time.monotonic() - started)
