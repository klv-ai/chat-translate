from __future__ import annotations

from chat_translate.provider.base import CompletionError
from chat_translate.review import build_review_prompt, review_catalog, review_entry


class FakeLLM:
    """Scripted CompletionProvider that records the prompts it receives."""

    name = "fake"

    def __init__(self, reply: str = "OK", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.prompts: list[str] = []

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        self.prompts.append(prompt)
        if self.fail:
            raise CompletionError("no reviewer", self.name)
        return self.reply


def test_prompt_includes_the_supplied_context() -> None:
    prompt = build_review_prompt(
        "Resume", "Currículum vitae", "es-ES", "The product is Acme Notes."
    )
    assert prompt.startswith("The product is Acme Notes.")
    assert "UI string" in prompt
    assert "Do NOT flag style" in prompt
    assert "When unsure, answer OK" in prompt
    assert "OK or WRONG" in prompt


def test_prompt_without_context_has_no_preamble() -> None:
    prompt = build_review_prompt("Save", "Guardar", "es-ES")
    assert prompt.startswith("You are reviewing")


def test_ok_verdict() -> None:
    v = review_entry("Save", "Guardar", "es-ES", FakeLLM("OK"))
    assert v.ok and not v.flagged


def test_wrong_verdict_carries_the_reason() -> None:
    v = review_entry(
        "Resume",
        "Currículum vitae",
        "es-ES",
        FakeLLM("WRONG. This is the CV sense; it should be 'Reanudar'."),
    )
    assert v.flagged
    assert "Reanudar" in v.reason


def test_an_unparseable_answer_is_unreviewed_not_flagged() -> None:
    v = review_entry("Save", "Guardar", "es-ES", FakeLLM("I think it's fine?"))
    assert v.ok and v.unavailable and not v.flagged


def test_a_provider_error_is_unreviewed_not_flagged() -> None:
    v = review_entry("Save", "Guardar", "es-ES", FakeLLM(fail=True))
    assert v.ok and v.unavailable and not v.flagged


def test_an_empty_reply_is_unreviewed_not_approval() -> None:
    v = review_entry("Save", "Guardar", "es-ES", FakeLLM(""))
    assert v.unavailable
    assert not v.flagged


def test_review_catalog_collects_flagged_entries() -> None:
    llm = FakeLLM("WRONG: wrong sense")
    result = review_catalog(
        [("Save", "Guardar"), ("Resume", "Currículum vitae")],
        "es-ES",
        llm,
        context="ctx",
    )
    assert len(result.verdicts) == 2
    assert len(result.flagged) == 2
    assert all(p.startswith("ctx") for p in llm.prompts)
