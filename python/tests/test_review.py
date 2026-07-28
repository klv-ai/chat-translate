from __future__ import annotations

import httpx

from chat_translate.review import (
    DEFAULT_CONTEXT,
    build_review_prompt,
    review_catalog,
    review_entry,
)


def _client(reply: str) -> httpx.Client:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": reply}})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_prompt_carries_the_product_context() -> None:
    prompt = build_review_prompt("Resume", "Currículum vitae", "es-ES", DEFAULT_CONTEXT)
    # Without the product framing the reviewer cannot know which sense is meant.
    assert "Chatterbox" in prompt
    assert "UI string" in prompt
    # The bar is meaning, not taste. Prompted to judge whether a translation
    # "reads naturally", gemma4:e2b flagged 46% of a real catalog — including
    # correct entries like "(container)" -> "(contenedor)".
    assert "Do NOT flag style" in prompt
    assert "When unsure, answer OK" in prompt
    assert "OK or WRONG" in prompt


def test_ok_verdict() -> None:
    v = review_entry("Save", "Guardar", "es-ES", client=_client("OK"))
    assert v.ok and not v.flagged


def test_wrong_verdict_carries_the_reason() -> None:
    v = review_entry(
        "Resume",
        "Currículum vitae",
        "es-ES",
        client=_client("WRONG. This is the CV sense; it should be 'Reanudar'."),
    )
    assert v.flagged
    assert "Reanudar" in v.reason


def test_an_unparseable_answer_is_not_treated_as_a_problem() -> None:
    # Advisory tooling must not manufacture review work out of a bad response.
    v = review_entry("Save", "Guardar", "es-ES", client=_client("I think it's fine?"))
    assert v.ok


def test_a_reviewer_that_is_down_does_not_flag_everything() -> None:
    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no reviewer")

    v = review_entry(
        "Save", "Guardar", "es-ES", client=httpx.Client(transport=httpx.MockTransport(boom))
    )
    assert v.ok


def test_review_catalog_collects_flagged_entries() -> None:
    result = review_catalog(
        [("Save", "Guardar"), ("Resume", "Currículum vitae")],
        "es-ES",
        model="test",
    )
    # No Ollama in tests — every call fails and is treated as OK, so the suite
    # never depends on a running model.
    assert len(result.verdicts) == 2


def test_an_empty_reply_is_reported_as_unreviewed_not_as_approval() -> None:
    # gemma4 is a reasoning model: with thinking on, the token budget goes to
    # `message.thinking` and `content` comes back empty. Treating that as "OK"
    # made a completely broken reviewer look like a clean pass.
    v = review_entry("Save", "Guardar", "es-ES", client=_client(""))
    assert v.unavailable
    assert not v.flagged


def test_thinking_is_disabled_in_the_request() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.update(_json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "OK"}})

    review_entry(
        "Save",
        "Guardar",
        "es-ES",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert captured["think"] is False
