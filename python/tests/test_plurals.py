from __future__ import annotations

import httpx

from chat_translate.icu import missing_plural_categories, parse_branches
from chat_translate.plurals import build_expand_prompt, expand_message, parse_forms


def _client(reply: str) -> httpx.Client:
    def handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": reply}})

    return httpx.Client(transport=httpx.MockTransport(handler))


SRC = "Imported {count, plural, one {# file} other {# files}}."
RU = "Импортировано {count, plural, one {# файл} other {# файла}}."


def test_prompt_gives_concrete_numbers_not_category_names() -> None:
    # A general model does not reliably know what CLDR "few" means; it does know
    # what form goes with 2, 3, 4.
    prompt = build_expand_prompt(
        SRC,
        parse_branches("{count, plural, one {# файл} other {# файла}}"),
        ["few", "many"],
        "ru-RU",
    )
    assert "2, 3, 4" in prompt
    assert "0, 5" in prompt
    assert "keep the # exactly where it is" in prompt


def test_parse_forms_reads_category_lines() -> None:
    reply = "few: # файла\nmany: # файлов\nnope: ignored"
    assert parse_forms(reply, ["few", "many"]) == {"few": "# файла", "many": "# файлов"}
    # Markdown bold and stray quoting are common and must not break parsing.
    assert parse_forms("**few**: `# файла`", ["few"]) == {"few": "# файла"}


def test_expansion_inserts_before_other() -> None:
    out = expand_message(SRC, RU, "ru-RU", client=_client("few: # файла\nmany: # файлов")).after
    sels = [s for s, _ in parse_branches(out[out.index("{count") : out.rindex("}") + 1])]
    # CLDR requires `other` last.
    assert sels[-1] == "other"
    assert "few" in sels and "many" in sels
    assert missing_plural_categories(out, "ru-RU") == {}


def test_rejects_a_branch_that_lost_its_number_slot() -> None:
    # The whole point is the count; a branch without # is worse than none.
    r = expand_message(SRC, RU, "ru-RU", client=_client("few: файла\nmany: файлов"))
    assert not r.ok
    assert "#" in (r.error or "")
    assert r.after == RU  # unchanged


def test_no_gap_is_a_no_op() -> None:
    r = expand_message(
        SRC,
        "Importiert {count, plural, one {# Datei} other {# Dateien}}.",
        "de-DE",
        client=_client("should not be called"),
    )
    assert r.after == "Importiert {count, plural, one {# Datei} other {# Dateien}}."
    assert not r.added


def test_unusable_reply_leaves_the_message_alone() -> None:
    r = expand_message(SRC, RU, "ru-RU", client=_client("I'm not sure about that."))
    assert not r.ok and r.after == RU
