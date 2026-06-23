"""Ollama adapter (self-hosted TranslateGemma over Ollama's HTTP API)."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..masking import BRACKET_SENTINELS, Sentinels
from .base import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    build_instruction_tuned_prompt,
)


@dataclass(slots=True)
class OllamaConfig:
    host: str = "http://localhost:11434"
    model: str = "translategemma:4b"


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
        self._client = client or httpx.Client(timeout=60.0)

    @property
    def sentinels(self) -> Sentinels:
        # Same reasoning as the GGUF backend: an LLM tokenizer drops PUA code
        # points, so use the visible bracket scheme.
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
        # the full instruction-tuned prompt (the model card format) as content.
        prompt = build_instruction_tuned_prompt(text, options)
        try:
            res = self._client.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0},  # fidelity over flair
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
        try:
            res = self._client.get(f"{self._host}/api/tags")
            if not res.is_success:
                return False
            models = res.json().get("models", [])
            return any(m.get("name") == self._model for m in models)
        except httpx.HTTPError:
            return False
