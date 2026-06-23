from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from chat_translate import (
    DeepLConfig,
    DeepLProvider,
    LocalGGUFProvider,
    OllamaConfig,
    OllamaProvider,
    TranslateOptions,
    TranslationError,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ─────────────────────────── DeepL (mocked httpx) ───────────────────────────


def test_deepl_uses_free_host_and_builds_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"translations": [{"detected_source_language": "DE", "text": "hello"}]}
        )

    p = DeepLProvider(DeepLConfig(api_key="key-123:fx"), client=_client(handler))
    r = p.translate("hallo", TranslateOptions(target_lang="en", source_lang="de"))

    assert r.text == "hello"
    assert r.detected_source_lang == "de"  # detected lang lowercased
    assert seen["url"] == "https://api-free.deepl.com/v2/translate"
    assert seen["auth"] == "DeepL-Auth-Key key-123:fx"
    assert seen["body"]["text"] == ["hallo"]
    assert seen["body"]["target_lang"] == "en"
    assert seen["body"]["source_lang"] == "de"


def test_deepl_omits_source_lang_for_paid_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "source_lang" not in body
        assert str(request.url).startswith("https://api.deepl.com/")
        return httpx.Response(
            200, json={"translations": [{"detected_source_language": "FR", "text": "hello"}]}
        )

    p = DeepLProvider(DeepLConfig(api_key="paid-key"), client=_client(handler))
    r = p.translate("bonjour", TranslateOptions(target_lang="en"))
    assert r.detected_source_lang == "fr"


def test_deepl_429_is_retryable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    p = DeepLProvider(DeepLConfig(api_key="paid-key"), client=_client(handler))
    with pytest.raises(TranslationError) as ei:
        p.translate("x", TranslateOptions(target_lang="de"))
    assert ei.value.provider == "deepl"
    assert ei.value.retryable is True


# ─────────────────────── Local GGUF (injected fake llama) ───────────────────


class _FakeLlama:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def create_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        self.prompts.append(messages[0]["content"])
        return {"choices": [{"message": {"content": "translated text"}}]}


def test_local_gguf_requires_source_lang() -> None:
    p = LocalGGUFProvider(llama=_FakeLlama())
    with pytest.raises(TranslationError):
        p.translate("hello", TranslateOptions(target_lang="de"))


def test_local_gguf_builds_prompt_and_returns_text() -> None:
    fake = _FakeLlama()
    p = LocalGGUFProvider(llama=fake)
    r = p.translate("hello world", TranslateOptions(target_lang="de", source_lang="en"))

    assert r.text == "translated text"
    assert r.detected_source_lang == "en"
    assert "from en to de" in fake.prompts[0]


def test_local_gguf_requires_llama_or_model_path() -> None:
    with pytest.raises(TranslationError):
        LocalGGUFProvider()


# ───────────────────────────── Ollama (mocked httpx) ─────────────────────────


def test_ollama_requires_source_lang() -> None:
    p = OllamaProvider(client=_client(lambda _r: httpx.Response(200, json={})))
    with pytest.raises(TranslationError):
        p.translate("hello", TranslateOptions(target_lang="de"))


def test_ollama_posts_chat_and_parses_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "hallo welt"}})

    p = OllamaProvider(
        OllamaConfig(host="http://ollama:11434", model="translategemma:4b"),
        client=_client(handler),
    )
    r = p.translate("hello world", TranslateOptions(target_lang="de", source_lang="en"))

    assert r.text == "hallo welt"
    assert r.detected_source_lang == "en"
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["model"] == "translategemma:4b"
    assert seen["body"]["stream"] is False
    assert "from en to de" in seen["body"]["messages"][0]["content"]


def test_ollama_5xx_is_retryable() -> None:
    p = OllamaProvider(client=_client(lambda _r: httpx.Response(500, text="boom")))
    with pytest.raises(TranslationError) as ei:
        p.translate("x", TranslateOptions(target_lang="de", source_lang="en"))
    assert ei.value.provider == "ollama"
    assert ei.value.retryable is True


def test_ollama_health_check_matches_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/tags"
        return httpx.Response(
            200, json={"models": [{"name": "translategemma:4b"}, {"name": "llama3"}]}
        )

    assert (
        OllamaProvider(
            OllamaConfig(model="translategemma:4b"), client=_client(handler)
        ).health_check()
        is True
    )
    assert (
        OllamaProvider(OllamaConfig(model="not-pulled"), client=_client(handler)).health_check()
        is False
    )
