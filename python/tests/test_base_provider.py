from __future__ import annotations

from chat_translate import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
)


class _StubProvider(BaseTranslationProvider):
    name = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=True,
            formality=False,
            glossaries=False,
            context_hint=False,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        self.calls.append(text)
        return TranslateResult(text=text.upper(), detected_source_lang="en")

    def health_check(self) -> bool:
        return True


def test_translate_batch_is_sequential_and_ordered() -> None:
    p = _StubProvider()
    res = p.translate_batch(["a", "b", "c"], TranslateOptions(target_lang="de"))

    assert [r.text for r in res] == ["A", "B", "C"]
    assert p.calls == ["a", "b", "c"]
