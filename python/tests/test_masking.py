from __future__ import annotations

from chat_translate import BRACKET_SENTINELS, DEFAULT_SENTINELS, mask_non_translatable


def test_round_trips_do_not_translate_spans_verbatim() -> None:
    raw = "hey @alice see https://x.com/pr/42 lgtm 👍 `npm run build` :tada:"
    m = mask_non_translatable(raw)

    assert "@alice" not in m.masked
    assert "https://" not in m.masked
    assert "👍" not in m.masked
    assert m.token_count == 5  # mention, url, emoji, code span, shortcode

    # Restoring the (untranslated) masked form reproduces the original exactly.
    assert m.restore(m.masked) == raw
    assert m.find_unrestored(m.masked) == []


def test_reports_dropped_tokens() -> None:
    m = mask_non_translatable("ping @bob now")
    assert m.token_count == 1

    eaten = "translated text without the placeholder"
    assert m.find_unrestored(eaten) == [0]


def test_keeps_trailing_punctuation_outside_url_token() -> None:
    raw = "see https://x.com/foo."
    m = mask_non_translatable(raw)
    assert m.restore(m.masked) == raw
    assert m.masked.endswith(".")  # the period was trimmed out of the span


def test_default_scheme_uses_private_use_area() -> None:
    m = mask_non_translatable("hey @alice")
    assert m.sentinels is DEFAULT_SENTINELS
    assert "⟦0⟧" not in m.masked
    assert m.restore(m.masked) == "hey @alice"


def test_custom_sentinel_scheme_round_trips_and_is_carried() -> None:
    m = mask_non_translatable("hey @alice", sentinels=BRACKET_SENTINELS)
    # The chosen scheme is recorded on the message and used by restore.
    assert m.sentinels is BRACKET_SENTINELS
    assert "⟦0⟧" in m.masked
    assert "@alice" not in m.masked
    assert m.restore(m.masked) == "hey @alice"
    assert m.find_unrestored(m.masked) == []
