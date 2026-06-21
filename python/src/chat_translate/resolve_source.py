r"""Ingest-time source-language resolution.

Runs once, when a message is posted, and the result is the source of truth for
every later translation. Precedence (highest signal first):

  1. user-confirmed override  — a replay-widget correction is ground truth.
  2. confident detection      — the author wrote something other than their UI
                                language, and the detector is sure.
  3. UI-language prior         — the deterministic, cold-start-free fallback.

It NEVER returns None (the local GGUF path can't auto-detect, and a persisted
row needs a concrete fact), and it's deterministic, so cache keys never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import regex as re

LanguageCode = str

SourceProvenance = Literal["user_confirmed", "detected", "ui_fallback"]

# What to do in the short-text loanword danger zone when a reliable detection
# DISAGREES with ui_lang. "prefer_prior" keeps ui_lang (safe against "ok"/"ciao"
# false positives); "trust_detector" takes the detection (better for short
# code-switching). Either branch marks the row confident=False.
ShortTextPolicy = Literal["prefer_prior", "trust_detector"]

DEFAULT_SHORT_TEXT_LETTERS = 10
DEFAULT_SHORT_TEXT_POLICY: ShortTextPolicy = "prefer_prior"

_LETTER_RE = re.compile(r"\p{L}")


def letter_count(s: str) -> int:
    """Unicode letters only — masked input has had mentions/URLs/emoji turned to spaces."""
    return len(_LETTER_RE.findall(s))


@dataclass(slots=True)
class DetectionResult:
    #: Best guess, or None when the detector has no opinion at all.
    lang: LanguageCode | None
    #: The detector's own reliability verdict (e.g. ELD's is_reliable()).
    reliable: bool
    #: Optional top score in [0,1], when the detector exposes one.
    score: float | None = None


class ConfidenceDetector(Protocol):
    def detect(self, text: str) -> DetectionResult: ...


class _EldResult(Protocol):
    language: str | None

    def scores(self) -> dict[str, float]: ...

    def is_reliable(self) -> bool: ...


class EldDetectorLike(Protocol):
    """The slice of ``eld.LanguageDetector`` this adapter relies on."""

    def detect(self, text: str) -> _EldResult: ...

    def dynamic_lang_subset(self, languages: list[str]) -> object: ...


def eld_detector(
    eld: EldDetectorLike, subset: list[LanguageCode] | None = None
) -> ConfidenceDetector:
    """Adapt an ``eld.LanguageDetector`` to ``ConfidenceDetector``. Constrain to
    the room's plausible languages once, at construction, for accuracy + latency.
    ELD emits ISO 639-1 codes, so no code mapping is needed."""
    if subset:
        eld.dynamic_lang_subset(subset)

    class _Adapter:
        def detect(self, text: str) -> DetectionResult:
            r = eld.detect(text)
            lang = r.language or None
            score = r.scores().get(lang) if lang else None
            return DetectionResult(lang=lang, reliable=r.is_reliable(), score=score)

    return _Adapter()


@dataclass(slots=True)
class ResolvedSource:
    #: The source language to store, translate FROM, and key the cache on. Never None.
    lang: LanguageCode
    provenance: SourceProvenance
    #: True = settled; False = fell back/guarded, a correction or stronger model
    #: could improve it later — persist so such rows stay revisitable.
    confident: bool
    #: What the detector actually said, even when unused — telemetry / re-detect.
    detected: LanguageCode | None = None


@dataclass(slots=True)
class ResolveSourceOptions:
    #: Author's chosen UI language — the fallback prior. Present from message #1.
    ui_lang: LanguageCode
    #: User-confirmed correction (replay widget). Strongest possible signal.
    override: LanguageCode | None = None
    #: Below this many letters we're in the loanword danger zone. Default 10.
    short_text_letters: int | None = None
    #: Behaviour in the short-text zone on disagreement. Default "prefer_prior".
    short_text_policy: ShortTextPolicy | None = None
    #: Optional floor on the detector's top score, layered on its reliable flag.
    min_score: float | None = None


def _same_lang(a: LanguageCode, b: LanguageCode) -> bool:
    """Compare base subtags: "pt-BR" and "pt" count as the same language."""
    return a.split("-")[0].lower() == b.split("-")[0].lower()


def resolve_source(
    detect_input: str, detector: ConfidenceDetector, opts: ResolveSourceOptions
) -> ResolvedSource:
    """Resolve one freshly-posted message's source language at ingest.

    ``detect_input`` is the MASKED text with sentinels replaced by spaces.
    """
    # 1. A user correction is ground truth. Don't even detect — just record it.
    if opts.override:
        return ResolvedSource(lang=opts.override, provenance="user_confirmed", confident=True)

    # 2. Detect.
    det = detector.detect(detect_input)
    detected = det.lang
    passes_score = opts.min_score is None or (det.score or 0.0) >= opts.min_score

    # 3. Nothing usable -> UI prior, flagged revisitable.
    if det.lang is None or not det.reliable or not passes_score:
        return ResolvedSource(
            lang=opts.ui_lang, provenance="ui_fallback", confident=False, detected=detected
        )
    lang = det.lang

    # 4. Enough letters that we trust the detector outright (incl. when it
    #    overrides the prior — the German-UI user writing English).
    threshold = (
        opts.short_text_letters
        if opts.short_text_letters is not None
        else DEFAULT_SHORT_TEXT_LETTERS
    )
    if letter_count(detect_input) >= threshold:
        return ResolvedSource(lang=lang, provenance="detected", confident=True, detected=detected)

    # 5. Short input, loanword danger zone. A detection that corroborates the
    #    prior is trusted; on disagreement the policy decides (confident=False).
    if _same_lang(lang, opts.ui_lang):
        return ResolvedSource(lang=lang, provenance="detected", confident=True, detected=detected)
    policy = opts.short_text_policy or DEFAULT_SHORT_TEXT_POLICY
    if policy == "trust_detector":
        return ResolvedSource(lang=lang, provenance="detected", confident=False, detected=detected)
    return ResolvedSource(
        lang=opts.ui_lang, provenance="ui_fallback", confident=False, detected=detected
    )
