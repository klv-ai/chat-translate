"""Ollama adapters: TranslateGemma translation, and general instruct completion."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..masking import BRACKET_SENTINELS, Sentinels
from .base import (
    BaseTranslationProvider,
    CompletionError,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    build_instruction_tuned_prompt,
    build_ui_label_prompt,
)


@dataclass(slots=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "translategemma:4b"
    #: Per-request timeout. A read timeout surfaces as a retryable TranslationError.
    timeout_seconds: float = 60.0
    #: How long Ollama keeps the model loaded after a request (Ollama duration string).
    keep_alive: str = "5m"
    #: Cap on generated tokens; None leaves it to the server.
    max_output_tokens: int | None = None


class OllamaProvider(BaseTranslationProvider):
    """Runs a TranslateGemma-style model through a local Ollama daemon.

    Like the GGUF backend it can't auto-detect a source language, so a detector
    is load-bearing (the chat/room layer enforces this). The HTTP client is
    injectable, mirroring ``DeepLProvider``.
    """

    name = "ollama"

    def __init__(
        self, config: OllamaConfig | None = None, *, client: httpx.Client | None = None
    ) -> None:
        cfg = config or OllamaConfig()
        self._host = cfg.host
        self._model = cfg.model
        self._keep_alive = cfg.keep_alive
        self._max_output_tokens = cfg.max_output_tokens
        self._client = client or httpx.Client(timeout=cfg.timeout_seconds)

    @property
    def sentinels(self) -> Sentinels:
        # An LLM tokenizer drops PUA code points; the bracket scheme survives.
        return BRACKET_SENTINELS

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,  # best-effort prompt hint only; not dependable
            glossaries=False,
            context_hint=False,  # the instruction-tuned prompt has no context slot
            native_batch=False,  # Ollama is one-shot; the base class loops
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        if not options.source_lang:
            raise TranslationError(
                "ollama provider requires an explicit source_lang (no auto-detect)",
                self.name,
            )
        # Ollama's standard Gemma template just wraps the message, so we supply
        # the full instruction prompt as content.
        build = (
            build_ui_label_prompt
            if options.content_kind == "ui_label"
            else build_instruction_tuned_prompt
        )
        prompt = build(text, options)
        try:
            gen_options: dict[str, object] = {"temperature": 0}  # fidelity over flair
            if self._max_output_tokens is not None:
                gen_options["num_predict"] = self._max_output_tokens
            res = self._client.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": gen_options,
                },
            )
        except httpx.HTTPError as exc:
            raise TranslationError(
                "Ollama request failed", self.name, cause=exc, retryable=True
            ) from exc

        if not res.is_success:
            raise TranslationError(
                f"Ollama returned {res.status_code}",
                self.name,
                cause=res.text,
                retryable=res.status_code >= 500,
            )

        data = res.json()
        out = str((data.get("message") or {}).get("content") or "").strip()
        if not out:
            raise TranslationError("Empty translation from model", self.name)
        return TranslateResult(text=out, detected_source_lang=options.source_lang)

    def health_check(self) -> bool:
        return _has_model(self._client, self._host, self._model)


@dataclass(slots=True)
class OllamaCompletionConfig:
    model: str
    host: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    keep_alive: str = "5m"


class OllamaCompletionProvider:
    """:class:`~.base.CompletionProvider` backed by Ollama's chat endpoint.

    Deterministic (temperature 0) with reasoning disabled, so the reply is the
    answer itself.
    """

    name = "ollama"

    def __init__(
        self, config: OllamaCompletionConfig, *, client: httpx.Client | None = None
    ) -> None:
        self._host = config.host
        self._model = config.model
        self._keep_alive = config.keep_alive
        self._client = client or httpx.Client(timeout=config.timeout_seconds)

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        options: dict[str, object] = {"temperature": 0}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        try:
            res = self._client.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "keep_alive": self._keep_alive,
                    # Reasoning models otherwise spend the token budget in
                    # `message.thinking` and leave `content` empty.
                    "think": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": options,
                },
            )
            if not res.is_success:
                raise CompletionError(
                    f"Ollama returned {res.status_code}", self.name, cause=res.text
                )
            message = res.json().get("message") or {}
        except (httpx.HTTPError, ValueError) as exc:
            raise CompletionError("Ollama request failed", self.name, cause=exc) from exc
        return str(message.get("content") or "").strip()

    def health_check(self) -> bool:
        return _has_model(self._client, self._host, self._model)

    def close(self) -> None:
        self._client.close()


def _has_model(client: httpx.Client, host: str, model: str) -> bool:
    try:
        res = client.get(f"{host}/api/tags")
        if not res.is_success:
            return False
        models = res.json().get("models", [])
        return any(m.get("name") == model for m in models)
    except httpx.HTTPError:
        return False
