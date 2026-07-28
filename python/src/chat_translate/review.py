"""Second-pass review of a translated catalog by a general instruct model.

`catalog.verify` checks STRUCTURE — placeholders, plural skeletons, brand
names, dictionary-style blowups. All of it is shape, and shape is exactly what
a translation model gets right while still being wrong:

    "Resume {expertName}"  ->  "Currículum vitae {expertName}"

Structurally perfect. Semantically the wrong word entirely — the CV sense of
"resume" rather than "continue". Nothing mechanical catches that, because the
only thing wrong is the meaning.

So this asks a *different* model, a general instruct one, to read source and
translation side by side and say whether the translation works **as a control in
this product**. The product context is the point: told that Experts is an app
with an Ask surface and a Chatterbox surface, a reviewer knows "Resume" sits on
a paused run and that "Ask" may be a place as well as a verb.

Advisory, never a gate. It produces a review list for a human — an LLM judging
an LLM is a strong signal and a poor arbiter.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .provider.base import language_name

#: What the reviewer needs to know about the product to judge a label in situ.
#: Override with --context; this is the fallback so the tool is useful bare.
DEFAULT_CONTEXT = """\
The product is "Experts", a private AI workspace. Its named surfaces are:
Ask (chat with an AI expert), Chatterbox (team chat), CopyWrite (document
editor), Inbox (email), Research, Projects, Opportunities, and Collections
(document knowledge bases). Users are a mix of administrators configuring the
install and ordinary staff using it day to day."""


@dataclass(slots=True)
class Verdict:
    key: str
    translation: str
    ok: bool
    reason: str = ""
    #: True when the reviewer did not actually answer — down, or not emitting a
    #: verdict. Kept distinct from `ok` so "nothing flagged" cannot mean
    #: "nothing was reviewed".
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


def build_review_prompt(source: str, translation: str, target_lang: str, context: str) -> str:
    """Ask for a verdict token first, so a chatty model is still parseable."""
    lang = language_name(target_lang)
    return (
        f"{context}\n\n"
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
    *,
    context: str = DEFAULT_CONTEXT,
    host: str = "http://localhost:11434",
    model: str = "gemma4:12b",
    client: httpx.Client | None = None,
    keep_alive: str = "60m",
) -> Verdict:
    """Judge one entry. A reviewer that fails to answer is treated as OK —
    this is advisory, and an unavailable reviewer must not manufacture work."""
    http = client or httpx.Client(timeout=120.0)
    prompt = build_review_prompt(source, translation, target_lang, context)
    try:
        res = http.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "stream": False,
                "keep_alive": keep_alive,
                # Reasoning models (gemma4) put chain-of-thought in
                # `message.thinking` and the answer in `message.content`. Left
                # on, the token budget goes entirely to thinking and `content`
                # comes back EMPTY — a reviewer that silently approves
                # everything. We want the verdict, not the deliberation.
                "think": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_predict": 96},
            },
        )
        message = res.json().get("message") or {}
        body = str(message.get("content") or "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        return Verdict(
            key=source,
            translation=translation,
            ok=True,
            unavailable=True,
            reason=f"reviewer unreachable: {exc}",
        )

    match = _VERDICT_RE.search(body)
    if not match:
        # Not a refusal to flag — the reviewer did not answer the question.
        # Reported separately so a misconfigured model cannot masquerade as a
        # clean review.
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
    *,
    context: str = DEFAULT_CONTEXT,
    host: str = "http://localhost:11434",
    model: str = "gemma4:12b",
    on_progress: Callable[[int, Verdict], None] | None = None,
) -> ReviewResult:
    started = time.monotonic()
    http = httpx.Client(timeout=120.0)
    verdicts: list[Verdict] = []
    for i, (source, translation) in enumerate(entries):
        verdict = review_entry(
            source,
            translation,
            target_lang,
            context=context,
            host=host,
            model=model,
            client=http,
        )
        verdicts.append(verdict)
        if on_progress:
            on_progress(i, verdict)
    return ReviewResult(verdicts=verdicts, elapsed_seconds=time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m chat_translate.review",
        description="Review a translated catalog for meaning, not structure.",
    )
    ap.add_argument("catalog", type=Path, help="translated catalog JSON")
    ap.add_argument("target_lang", help="locale of that catalog, e.g. es-ES")
    ap.add_argument("--context", type=Path, help="file describing the product")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument(
        "--model",
        default="gemma4:12b",
        help="a GENERAL instruct model — a translation model cannot judge itself",
    )
    ap.add_argument("--limit", type=int, help="review only the first N entries")
    ap.add_argument("--out", type=Path, help="write flagged entries as JSON")
    args = ap.parse_args(argv)

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    context = (
        args.context.read_text(encoding="utf-8").strip()
        if args.context and args.context.exists()
        else DEFAULT_CONTEXT
    )
    # An entry translated to itself was never translated; nothing to review.
    pairs = [(k, v) for k, v in catalog.items() if v and v != k]
    if args.limit:
        pairs = pairs[: args.limit]

    total = len(pairs)
    print(f"Reviewing {total} {args.target_lang} entries with {args.model}…")

    def progress(i: int, v: Verdict) -> None:
        if v.flagged:
            print(f"  ! [{i + 1}/{total}] {v.key[:44]!r} -> {v.translation[:34]!r}")
            print(f"      {v.reason[:100]}")
        elif (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{total}]…", flush=True)

    result = review_catalog(
        pairs,
        args.target_lang,
        context=context,
        host=args.host,
        model=args.model,
        on_progress=progress,
    )

    rate = total / result.elapsed_seconds if result.elapsed_seconds else 0.0
    print(
        f"\n{len(result.flagged)}/{total} flagged for review "
        f"({result.elapsed_seconds:.0f}s, {rate:.2f} entries/s)."
    )
    if result.unreviewed:
        # Say so loudly. "0 flagged" because the reviewer never answered looks
        # exactly like "0 flagged" because everything was fine.
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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
