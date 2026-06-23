"""Local GGUF adapters (self-hosted, via llama.cpp).

Two flavours, because the chat template baked into the GGUF dictates the input
shape:

- ``LocalInstructionTunedProvider`` — for instruction-tuned TranslateGemma GGUFs
  whose template expects STRUCTURED content (``source_lang_code`` /
  ``target_lang_code`` / ``text``) and builds the official translation
  instruction itself. This is the right default for the TranslateGemma models
  this SDK targets.
- ``LocalGGUFProvider`` — generic: sends a flat instruction prompt, for GGUFs
  whose template simply wraps a plain user message (standard Gemma, etc.).

A raw text completion that bypasses the chat template is deliberately NOT used:
an instruction-tuned model needs the template's turn tokens, and without them it
ignores the instruction and continues the text instead.
"""

from __future__ import annotations

from typing import Any, Protocol, cast, runtime_checkable

from ..masking import BRACKET_SENTINELS, Sentinels
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
    """The slice of the llama_cpp.Llama API these providers rely on. ``content``
    may be a plain string or the structured list the IT template expects."""

    def create_chat_completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...


class _LocalLlamaProvider(BaseTranslationProvider):
    """Shared llama.cpp wiring: inject a ready ``llama`` (the usual case — the
    calling app owns it) or pass a ``model_path`` to build one lazily via the
    optional ``[local]`` extra."""

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
                f"{type(self).__name__} requires either an injected `llama` or a `model_path`",
                self.name,
            )

    @property
    def sentinels(self) -> Sentinels:
        # llama.cpp runs instruction-tuned models whose tokenizers eat PUA code
        # points; the visible bracket scheme survives.
        return BRACKET_SENTINELS

    def health_check(self) -> bool:
        return self._llama is not None


class LocalInstructionTunedProvider(_LocalLlamaProvider):
    """Instruction-tuned TranslateGemma via llama.cpp, using the model's native
    structured translation template.

    The GGUF ships a chat template that maps language codes to names and emits
    the official instruction prompt itself, so we pass STRUCTURED content rather
    than a flat string (which the template rejects).
    """

    def capabilities(self) -> ProviderCapabilities:
        # The template owns source/target; it has no slot for surrounding
        # context, so context_hint is False (the chat layer won't pass one).
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=False,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        if not options.source_lang:
            raise TranslationError(
                "instruction-tuned local provider requires an explicit source_lang "
                "(no auto-detect)",
                self.name,
            )
        try:
            res = self._llama.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "source_lang_code": options.source_lang,
                                "target_lang_code": options.target_lang,
                                "text": text,
                            }
                        ],
                    }
                ],
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


class LocalGGUFProvider(_LocalLlamaProvider):
    """Generic GGUF backend: a flat instruction prompt via chat completion.

    Suitable for GGUFs whose chat template wraps a plain user message. For an
    instruction-tuned TranslateGemma GGUF (rich structured template) use
    ``LocalInstructionTunedProvider`` instead.
    """

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=True,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        if not options.source_lang:
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
