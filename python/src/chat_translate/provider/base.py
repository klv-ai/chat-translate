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

from ..masking import DEFAULT_SENTINELS, Sentinels

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

    @property
    def sentinels(self) -> Sentinels:
        """Placeholder delimiter scheme this backend preserves through
        translation. Defaults to the Private-Use-Area scheme (safe for NMT);
        instruction-tuned LLM backends override it because their tokenizers eat
        PUA code points."""
        return DEFAULT_SENTINELS

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
    """A minimal, generic translation prompt for GGUFs whose chat template wraps
    a plain user message. The register hint is best-effort only — which is why
    those providers report ``formality=False``.

    Instruction-tuned TranslateGemma models should use
    :func:`build_instruction_tuned_prompt`, which follows the published model
    card; the rich GGUF template builds that prompt itself from structured
    content (see ``LocalInstructionTunedProvider``).
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


# ISO 639-1 → English language name. Used to render the instruction-tuned prompt
# (the published model card names the languages). Region subtags are stripped;
# unknown codes fall back to the code itself.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "pl": "Polish",
    "ru": "Russian",
    "uk": "Ukrainian",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "nb": "Norwegian",
    "fi": "Finnish",
    "cs": "Czech",
    "sk": "Slovak",
    "sl": "Slovenian",
    "el": "Greek",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ga": "Irish",
    "mt": "Maltese",
    "is": "Icelandic",
    "tr": "Turkish",
    "ca": "Catalan",
    "ja": "Japanese",
    "zh": "Chinese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    # Low-resource, but TranslateGemma handles it usably and the catalog
    # verifier keeps English for anything it mangles. Naming it matters:
    # the fallback prompt would say 'translate to tet', which the model
    # cannot be trusted to read as Tetum.
    "tet": "Tetum",
}


def language_name(code: str) -> str:
    """English name for an ISO 639-1 code (region subtag ignored); falls back to
    the code so an unmapped language still produces a usable prompt."""
    return LANGUAGE_NAMES.get(code.split("-")[0].lower(), code)


def build_ui_label_prompt(text: str, options: TranslateOptions) -> str:
    """Translation prompt for UI strings rather than prose.

    The model-card prompt asks for a faithful rendering of *text*, and for a
    bare label with no surrounding context TranslateGemma answers the way a
    dictionary would — every sense, on its own line, with a parenthetical
    caveat. "Flagged" comes back as "Marcado. / Señalizado. / Identificado. /
    (Dependiendo del contexto...)", which then renders as a paragraph inside a
    button.

    Naming the register — this is a control in a user interface, give exactly
    one short answer — collapses that to a single usable string. Measured on
    translategemma:12b against the labels that failed in production.
    """
    src = options.source_lang or ""
    s_name = language_name(src)
    t_name = language_name(options.target_lang)
    return (
        f"You are a professional {s_name} ({src}) to {t_name} "
        f"({options.target_lang}) translator localizing a software user "
        f"interface. The text is a button, menu item, or label. Give exactly "
        f"ONE translation — the single best fit for a UI control — with no "
        f"alternatives, no explanations, and no commentary. Keep it as short "
        f"as the original.\n"
        # Only mention placeholders when there ARE some. The instruction has to
        # show the delimiters to be understood, and a model given that example
        # alongside a string with no placeholders copies the EXAMPLE into its
        # answer — "Data from" came back as "Daten von ⟦…⟧". That cost ~60 keys
        # per locale, far more than the instruction saves.
        + (
            f"Any {chr(0x27E6)}…{chr(0x27E7)} token is a placeholder: copy each "
            f"one into your translation exactly as it appears, unchanged, and "
            f"never drop or renumber one.\n"
            if chr(0x27E6) in text
            else ""
        )
        + f"Please translate the following {s_name} text into {t_name}:"
        f"\n\n\n{text}"
    )


def build_instruction_tuned_prompt(text: str, options: TranslateOptions) -> str:
    """The instruction-tuned TranslateGemma translation prompt, verbatim from the
    published model card. Used by backends whose chat template does NOT build the
    translation instruction itself (e.g. Ollama's standard Gemma template). The
    rich llama.cpp GGUF template renders an equivalent prompt from structured
    content, so ``LocalInstructionTunedProvider`` does not use this.
    """
    src = options.source_lang or ""
    s_name = language_name(src)
    t_name = language_name(options.target_lang)
    return (
        f"You are a professional {s_name} ({src}) to {t_name} ({options.target_lang}) "
        f"translator. Your goal is to accurately convey the meaning and nuances of the "
        f"original {s_name} text while adhering to {t_name} grammar, vocabulary, and "
        f"cultural sensitivities.\n"
        f"Produce only the {t_name} translation, without any additional explanations or "
        f"commentary. Please translate the following {s_name} text into {t_name}:\n\n\n{text}"
    )
