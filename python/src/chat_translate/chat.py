"""Provider-agnostic chat layer above the TranslationProvider boundary.

Masks do-not-translate spans, resolves the source language (detecting ourselves
when the backend can't), and applies cheap short-circuits — so both providers
behave identically from here up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .masking import DEFAULT_RULES, MaskedMessage, MaskRule, mask_non_translatable
from .provider import (
    Formality,
    LanguageCode,
    ProviderCapabilities,
    TranslateOptions,
    TranslationError,
    TranslationProvider,
)


class LanguageDetector(Protocol):
    """Bare-code detector (returns a concrete language). Injected so the layer
    stays testable and backend-agnostic."""

    def detect(self, text: str) -> LanguageCode: ...


@dataclass(slots=True)
class ViewerTranslateOptions:
    source_lang: LanguageCode | None = None
    context: str | None = None
    formality: Formality | None = None
    glossary_id: str | None = None


@dataclass(slots=True)
class ViewerTranslation:
    text: str
    detected_source_lang: LanguageCode
    #: False = short-circuited (nothing to translate, or same language).
    translated: bool
    #: Non-empty means the engine ate placeholders — inspect / fall back.
    unrestored_tokens: list[int] = field(default_factory=list)


def _same_lang(a: LanguageCode, b: LanguageCode) -> bool:
    return a.split("-")[0].lower() == b.split("-")[0].lower()


class ChatTranslator:
    def __init__(
        self,
        provider: TranslationProvider,
        *,
        detector: LanguageDetector | None = None,
        default_formality: Formality = "prefer_less",
        rules: Sequence[MaskRule] = DEFAULT_RULES,
    ) -> None:
        self._provider = provider
        self._caps = provider.capabilities()
        self._detector = detector
        self._default_formality = default_formality
        self._rules = rules

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    def mask(self, raw: str) -> MaskedMessage:
        """Exposed so the fan-out layer can mask once and reuse across viewers."""
        return mask_non_translatable(raw, self._rules, self._provider.sentinels)

    def _resolve_source(
        self, detect_input: str, explicit: LanguageCode | None = None
    ) -> LanguageCode | None:
        if explicit:
            return explicit
        if self._caps.auto_detect_source:
            return None  # let the provider do it
        if self._detector is None:
            raise TranslationError(
                f"{self._provider.name} cannot auto-detect source and no detector is configured",
                self._provider.name,
            )
        return self._detector.detect(detect_input)

    def translate_for_viewer(
        self,
        raw: str,
        target_lang: LanguageCode,
        opts: ViewerTranslateOptions | None = None,
    ) -> ViewerTranslation:
        opts = opts or ViewerTranslateOptions()
        m = mask_non_translatable(raw, self._rules, self._provider.sentinels)

        # Short-circuit 1: nothing translatable (pure emoji / mention / code / url).
        if not m.has_translatable():
            return ViewerTranslation(
                text=raw,
                detected_source_lang=opts.source_lang or target_lang,
                translated=False,
            )

        detect_input = m.detection_text()
        source_lang = self._resolve_source(detect_input, opts.source_lang)

        # Short-circuit 2: the viewer already speaks the source language.
        if source_lang and _same_lang(source_lang, target_lang):
            return ViewerTranslation(text=raw, detected_source_lang=source_lang, translated=False)

        translate_opts = TranslateOptions(
            target_lang=target_lang,
            source_lang=source_lang,
            formality=(opts.formality or self._default_formality) if self._caps.formality else None,
            context=opts.context,
            glossary_id=opts.glossary_id,
        )
        result = self._provider.translate(m.masked, translate_opts)
        return ViewerTranslation(
            text=m.restore(result.text),
            detected_source_lang=result.detected_source_lang,
            translated=True,
            unrestored_tokens=m.find_unrestored(result.text),
        )


def create_chat_translator(
    provider: TranslationProvider,
    *,
    detector: LanguageDetector | None = None,
    default_formality: Formality = "prefer_less",
    rules: Sequence[MaskRule] = DEFAULT_RULES,
) -> ChatTranslator:
    return ChatTranslator(
        provider, detector=detector, default_formality=default_formality, rules=rules
    )
