"""Shared provider contracts.

The provider boundary sits as low as possible: it knows how to translate a
string and nothing about chat. Masking, caching, fan-out, and source resolution
all live above this interface, written once and reused regardless of backend.

`capabilities()` is the contract that lets the chat layer degrade gracefully
instead of assuming the backends are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

LanguageCode = str

# Register / politeness control, mapping onto DeepL's `formality` parameter.
# Prefer the ``prefer_*`` variants: plain ``more``/``less`` make DeepL error on
# languages that do not support formality, whereas ``prefer_*`` silently no-op.
Formality = Literal["default", "more", "less", "prefer_more", "prefer_less"]


@dataclass(slots=True)
class TranslateOptions:
    target_lang: LanguageCode
    #: Omit / None = let the provider auto-detect — IF capabilities allow it.
    source_lang: LanguageCode | None = None
    formality: Formality | None = None
    glossary_id: str | None = None
    #: Surrounding text used to disambiguate, WITHOUT being translated.
    context: str | None = None


@dataclass(slots=True)
class TranslateResult:
    text: str
    detected_source_lang: LanguageCode


@dataclass(slots=True)
class ProviderCapabilities:
    """A flag is True only if the chat layer can DEPEND on it.

    Best-effort approximations (e.g. coaxing a raw model toward a register via
    prompt) stay False so nothing builds on sand.
    """

    auto_detect_source: bool
    formality: bool
    glossaries: bool
    context_hint: bool
    native_batch: bool
    max_batch_size: int | None = None


class TranslationError(Exception):
    def __init__(
        self,
        message: str,
        provider: str,
        *,
        cause: object | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause
        self.retryable = retryable


class TranslationProvider(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def translate(self, text: str, options: TranslateOptions) -> TranslateResult: ...

    @abstractmethod
    def translate_batch(
        self, texts: list[str], options: TranslateOptions
    ) -> list[TranslateResult]: ...

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap liveness probe — matters most for the self-hosted path."""


class BaseTranslationProvider(TranslationProvider):
    """Shared scaffolding. The important default is `translate_batch`: providers
    without a native batch endpoint get a correct, if slower, sequential
    implementation for free. Sequential (not parallel) is deliberate for the
    self-hosted path: a single GPU does not benefit from being hammered.
    """

    def translate_batch(self, texts: list[str], options: TranslateOptions) -> list[TranslateResult]:
        return [self.translate(t, options) for t in texts]


def build_translation_prompt(text: str, options: TranslateOptions) -> str:
    """The TranslateGemma-style translation prompt shared by the self-hosted
    backends (local GGUF + Ollama). The register hint is best-effort only —
    which is exactly why those providers report ``formality=False``.
    """
    if options.formality in ("more", "prefer_more"):
        register = " Use a formal register."
    elif options.formality in ("less", "prefer_less"):
        register = " Use an informal register."
    else:
        register = ""
    ctx = (
        f"\nConversation context (for disambiguation only, do not translate):\n{options.context}\n"
        if options.context
        else ""
    )
    return (
        f"Translate the following text from {options.source_lang} to {options.target_lang}."
        f"{register} Output only the translation, with no preamble or quotes."
        f"{ctx}\n\n{text}"
    )
