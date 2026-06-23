"""Local GGUF adapter (self-hosted, via llama.cpp)."""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from .base import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    build_translation_prompt,
)


@runtime_checkable
class LlamaLike(Protocol):
    """The slice of the llama_cpp.Llama API this provider relies on."""

    def create_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


class LocalGGUFProvider(BaseTranslationProvider):
    """TranslateGemma-style backend backed by a local GGUF via llama.cpp.

    The llama runtime is INJECTABLE: pass a ready ``llama`` instance (the usual
    case — the calling app owns it), or a ``model_path`` to have the provider
    build one lazily via the optional ``[local]`` extra.
    """

    name = "local"

    def __init__(
        self,
        *,
        llama: LlamaLike | None = None,
        model_path: str | None = None,
        temperature: float = 0.0,
        **llama_kwargs: Any,
    ) -> None:
        self._temperature = temperature
        if llama is not None:
            self._llama: LlamaLike = llama
        elif model_path is not None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise TranslationError(
                    "llama-cpp-python is not installed; install the '[local]' extra "
                    "or inject a llama instance",
                    self.name,
                    cause=exc,
                ) from exc
            self._llama = cast(LlamaLike, Llama(model_path=model_path, **llama_kwargs))
        else:
            raise TranslationError(
                "LocalGGUFProvider requires either an injected `llama` or a `model_path`",
                self.name,
            )

    def capabilities(self) -> ProviderCapabilities:
        # The raw model cannot do these as features the layer can RELY on, so
        # they stay False. The chat layer reads this and compensates — since
        # auto_detect_source is False, it runs its own detector first.
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=True,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        if not options.source_lang:
            # Fail loudly rather than guess: capabilities() already told the
            # caller this backend can't auto-detect.
            raise TranslationError(
                "local GGUF provider requires an explicit source_lang (no auto-detect)",
                self.name,
            )
        prompt = build_translation_prompt(text, options)
        try:
            res = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            out = str(res["choices"][0]["message"]["content"] or "").strip()
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap any llama/runtime failure uniformly
            raise TranslationError(
                "llama.cpp inference failed", self.name, cause=exc, retryable=True
            ) from exc
        if not out:
            raise TranslationError("Empty translation from model", self.name)
        return TranslateResult(text=out, detected_source_lang=options.source_lang)

    def health_check(self) -> bool:
        return self._llama is not None
