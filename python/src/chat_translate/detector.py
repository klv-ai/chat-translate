"""Adapters bridging a confidence-bearing detector down to the bare-code
``LanguageDetector`` the chat / room layers consume.

``resolve_source`` needs a ``ConfidenceDetector`` (lang + reliability + score),
but the chat/room translators only want a ``LanguageDetector`` (a concrete
code). These helpers narrow it, supplying a concrete ``fallback`` for the case
where the detector has no opinion (ELD can return None) — the chat layer must
never receive None.
"""

from __future__ import annotations

from .chat import LanguageDetector
from .provider import LanguageCode
from .resolve_source import ConfidenceDetector, EldDetectorLike, eld_detector


def to_language_detector(detector: ConfidenceDetector, fallback: LanguageCode) -> LanguageDetector:
    """Narrow a ConfidenceDetector to a bare-code LanguageDetector."""

    class _Adapter:
        def detect(self, text: str) -> LanguageCode:
            r = detector.detect(text)
            return r.lang if r.lang is not None else fallback

    return _Adapter()


def eld_language_detector(
    eld: EldDetectorLike,
    *,
    subset: list[LanguageCode] | None = None,
    fallback: LanguageCode,
) -> LanguageDetector:
    """Build a bare-code LanguageDetector straight from an ``eld.LanguageDetector``."""
    return to_language_detector(eld_detector(eld, subset), fallback)
