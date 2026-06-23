"""DeepL adapter (managed pass-through)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .base import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
)


@dataclass(slots=True)
class DeepLConfig:
    api_key: str
    #: Override the host if needed; auto-selected from the key by default.
    api_url: str | None = None


class DeepLProvider(BaseTranslationProvider):
    name = "deepl"

    def __init__(self, config: DeepLConfig, *, client: httpx.Client | None = None) -> None:
        # Free keys end in ":fx" and must hit the free host.
        is_free = config.api_key.endswith(":fx")
        self._endpoint = config.api_url or (
            "https://api-free.deepl.com/v2" if is_free else "https://api.deepl.com/v2"
        )
        self._api_key = config.api_key
        self._client = client or httpx.Client(timeout=30.0)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=True,
            formality=True,
            glossaries=True,
            context_hint=True,
            native_batch=True,
            max_batch_size=50,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        return self.translate_batch([text], options)[0]

    def translate_batch(self, texts: list[str], options: TranslateOptions) -> list[TranslateResult]:
        body: dict[str, Any] = {"text": texts, "target_lang": options.target_lang}
        # A glossary requires an explicit source_lang (can't auto-detect).
        if options.source_lang:
            body["source_lang"] = options.source_lang
        if options.formality and options.formality != "default":
            body["formality"] = options.formality
        if options.glossary_id:
            body["glossary_id"] = options.glossary_id
        if options.context:
            body["context"] = options.context

        try:
            res = self._client.post(
                f"{self._endpoint}/translate",
                json=body,
                headers={"Authorization": f"DeepL-Auth-Key {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise TranslationError(
                "DeepL request failed", self.name, cause=exc, retryable=True
            ) from exc

        if not res.is_success:
            retryable = res.status_code == 429 or res.status_code >= 500
            raise TranslationError(
                f"DeepL returned {res.status_code}", self.name, cause=res.text, retryable=retryable
            )

        data = res.json()
        return [
            TranslateResult(
                text=t["text"], detected_source_lang=str(t["detected_source_language"]).lower()
            )
            for t in data["translations"]
        ]

    def health_check(self) -> bool:
        try:
            res = self._client.get(
                f"{self._endpoint}/usage",
                headers={"Authorization": f"DeepL-Auth-Key {self._api_key}"},
            )
            return res.is_success
        except httpx.HTTPError:
            return False
