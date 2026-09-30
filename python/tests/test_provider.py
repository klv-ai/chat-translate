from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from chat_translate import (
    BRACKET_SENTINELS,
    DEFAULT_SENTINELS,
    CompletionError,
    CompletionProvider,
    DeepLConfig,
    DeepLProvider,
    LocalGGUFProvider,
    LocalInstructionTunedProvider,
    OllamaCompletionConfig,
    OllamaCompletionProvider,
    OllamaConfig,
    OllamaProvider,
    TranslateOptions,
    TranslationError,
    completion_config_from_env,
    config_from_env,
    create_completion_provider,
    create_provider,
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


# ─────────── Local instruction-tuned GGUF (structured chat content) ──────────


class _FakeStructLlama:
    """Captures the (structured) messages the IT provider sends."""

    def __init__(self) -> None:
        self.messages: list[Any] = []

    def create_chat_completion(self, *, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        self.messages.append(messages)
        return {"choices": [{"message": {"content": "hallo welt"}}]}


def test_local_it_requires_source_lang() -> None:
    p = LocalInstructionTunedProvider(llama=_FakeStructLlama())
    with pytest.raises(TranslationError):
        p.translate("hello", TranslateOptions(target_lang="de"))


def test_local_it_sends_structured_content() -> None:
    fake = _FakeStructLlama()
    p = LocalInstructionTunedProvider(llama=fake)
    r = p.translate("hello world", TranslateOptions(target_lang="de", source_lang="en"))

    assert r.text == "hallo welt"
    assert r.detected_source_lang == "en"
    # The IT template needs structured content, not a flat prompt string.
    item = fake.messages[0][0]["content"][0]
    assert item == {
        "type": "text",
        "source_lang_code": "en",
        "target_lang_code": "de",
        "text": "hello world",
    }


def test_local_it_requires_llama_or_model_path() -> None:
    with pytest.raises(TranslationError):
        LocalInstructionTunedProvider()


# ─────────────────────────── Per-provider sentinels ─────────────────────────


def test_provider_sentinel_schemes() -> None:
    # NMT default keeps the Private-Use-Area scheme; the LLM backends override to
    # the bracket scheme their tokenizers preserve.
    assert DeepLProvider(DeepLConfig(api_key="k")).sentinels is DEFAULT_SENTINELS
    assert LocalInstructionTunedProvider(llama=_FakeStructLlama()).sentinels is BRACKET_SENTINELS
    assert LocalGGUFProvider(llama=_FakeLlama()).sentinels is BRACKET_SENTINELS
    assert OllamaProvider().sentinels is BRACKET_SENTINELS


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
    # Ollama gets the full instruction-tuned (model card) prompt as content.
    content = seen["body"]["messages"][0]["content"]
    assert "English (en)" in content
    assert "German (de)" in content
    assert "Please translate the following English text into German" in content


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


def test_ollama_sends_keep_alive_and_output_cap() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "hola"}})

    provider = OllamaProvider(
        OllamaConfig(model="translategemma:12b", keep_alive="60m", max_output_tokens=256),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.translate("hi", TranslateOptions(target_lang="es", source_lang="en"))

    assert captured["keep_alive"] == "60m"
    assert captured["options"] == {"temperature": 0, "num_predict": 256}


def test_ui_label_prompt_tells_the_model_to_copy_placeholders() -> None:
    from chat_translate.provider.base import build_ui_label_prompt

    opts = TranslateOptions(target_lang="es", source_lang="en")
    prompt = build_ui_label_prompt("⟦PH0⟧/⟦PH1⟧ GB", opts)
    assert "placeholder" in prompt
    assert "exactly as it appears" in prompt
    assert "ONE translation" in prompt


def test_ui_label_prompt_omits_the_placeholder_rule_when_there_are_none() -> None:
    from chat_translate.provider.base import build_ui_label_prompt

    opts = TranslateOptions(target_lang="de", source_lang="en")
    assert "placeholder" not in build_ui_label_prompt("Data from", opts)
    assert "\u27e6" not in build_ui_label_prompt("Data from", opts)


def test_ollama_uses_the_ui_label_prompt_for_ui_label_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "Guardar"}})

    p = OllamaProvider(client=_client(handler))
    ui = TranslateOptions(target_lang="es", source_lang="en", content_kind="ui_label")
    p.translate("Save", ui)
    assert "ONE translation" in seen["body"]["messages"][0]["content"]

    p.translate("Save", TranslateOptions(target_lang="es", source_lang="en"))
    assert "ONE translation" not in seen["body"]["messages"][0]["content"]


# ───────────────────────── Ollama completion (mocked httpx) ─────────────────


def test_ollama_completion_disables_thinking_and_returns_content() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "  OK  "}})

    llm = OllamaCompletionProvider(
        OllamaCompletionConfig(model="instruct:1b", keep_alive="10m"), client=_client(handler)
    )
    assert llm.complete("hi", max_tokens=12) == "OK"
    assert seen["body"]["model"] == "instruct:1b"
    assert seen["body"]["think"] is False
    assert seen["body"]["keep_alive"] == "10m"
    assert seen["body"]["options"] == {"temperature": 0, "num_predict": 12}


def test_ollama_completion_errors_raise_completion_error() -> None:
    llm = OllamaCompletionProvider(
        OllamaCompletionConfig(model="m"),
        client=_client(lambda _r: httpx.Response(500, text="boom")),
    )
    with pytest.raises(CompletionError):
        llm.complete("hi")

    def down(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    llm = OllamaCompletionProvider(OllamaCompletionConfig(model="m"), client=_client(down))
    with pytest.raises(CompletionError):
        llm.complete("hi")


# ───────────────────────────── Env configuration ────────────────────────────


def test_config_from_env_reads_ollama_tuning() -> None:
    cfg = config_from_env(
        {
            "TRANSLATION_PROVIDER": "ollama",
            "OLLAMA_MODEL": "translategemma:12b",
            "OLLAMA_TIMEOUT_SECONDS": "180",
            "OLLAMA_KEEP_ALIVE": "60m",
            "OLLAMA_MAX_OUTPUT_TOKENS": "256",
        }
    )
    assert cfg.ollama is not None
    assert cfg.ollama.model == "translategemma:12b"
    assert cfg.ollama.timeout_seconds == 180.0
    assert cfg.ollama.keep_alive == "60m"
    assert cfg.ollama.max_output_tokens == 256
    assert isinstance(create_provider(cfg), OllamaProvider)


def test_config_from_env_rejects_a_non_numeric_timeout() -> None:
    with pytest.raises(ValueError, match="OLLAMA_TIMEOUT_SECONDS"):
        config_from_env({"TRANSLATION_PROVIDER": "ollama", "OLLAMA_TIMEOUT_SECONDS": "soon"})


def test_completion_config_from_env() -> None:
    cfg = completion_config_from_env(
        {"COMPLETION_MODEL": "instruct:1b", "OLLAMA_HOST": "http://ollama:11434"}
    )
    assert cfg.provider == "ollama"
    assert cfg.ollama is not None
    assert cfg.ollama.model == "instruct:1b"
    assert cfg.ollama.host == "http://ollama:11434"
    assert isinstance(create_completion_provider(cfg), CompletionProvider)

    override = completion_config_from_env(
        {"COMPLETION_MODEL": "m", "COMPLETION_HOST": "http://llm:1", "OLLAMA_HOST": "http://x"}
    )
    assert override.ollama is not None and override.ollama.host == "http://llm:1"


def test_completion_config_requires_a_model() -> None:
    with pytest.raises(ValueError, match="COMPLETION_MODEL"):
        completion_config_from_env({})
    with pytest.raises(ValueError, match="COMPLETION_PROVIDER"):
        completion_config_from_env({"COMPLETION_PROVIDER": "nope", "COMPLETION_MODEL": "m"})
