from __future__ import annotations

from chat_translate import (
    BaseTranslationProvider,
    InMemoryTranslationCache,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    create_room_translator,
)


class _EchoProvider(BaseTranslationProvider):
    name = "echo"

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=True,
            formality=True,
            glossaries=False,
            context_hint=True,
            native_batch=True,
            max_batch_size=50,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        self.calls += 1
        return TranslateResult(
            text=f"[{options.target_lang}] {text}",
            detected_source_lang=options.source_lang or "en",
        )

    def health_check(self) -> bool:
        return True


def test_room_dedupes_targets_and_keys_by_language() -> None:
    room = create_room_translator(_EchoProvider())
    out = room.translate_for_room("hey @alice ship it", ["de", "fr", "de"])

    assert sorted(out.by_language.keys()) == ["de", "fr"]
    assert out.stats.targets == 2  # distinct languages
    assert "@alice" in out.by_language["de"].text  # mention preserved through translation


def test_room_serves_repeat_from_cache() -> None:
    provider = _EchoProvider()
    room = create_room_translator(provider, cache=InMemoryTranslationCache())

    room.translate_for_room("hello world", ["de"])
    assert provider.calls == 1

    out = room.translate_for_room("hello world", ["de"])  # identical masked form → cache hit
    assert provider.calls == 1
    assert out.stats.cache_hits == 1
    assert "[de]" in out.by_language["de"].text
