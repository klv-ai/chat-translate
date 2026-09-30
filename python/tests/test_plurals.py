from __future__ import annotations

from chat_translate.icu import missing_plural_categories, parse_branches
from chat_translate.plurals import build_expand_prompt, expand_message, parse_forms
from chat_translate.provider.base import CompletionError


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str = "", *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls = 0

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        self.calls += 1
        if self.fail:
            raise CompletionError("down", self.name)
        return self.reply


SRC = "Imported {count, plural, one {# file} other {# files}}."
RU = "Импортировано {count, plural, one {# файл} other {# файла}}."


def test_prompt_gives_example_counts_per_category() -> None:
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
    # Markdown bold and quoting are tolerated.
    assert parse_forms("**few**: `# файла`", ["few"]) == {"few": "# файла"}


def test_expansion_inserts_before_other() -> None:
    out = expand_message(SRC, RU, "ru-RU", FakeLLM("few: # файла\nmany: # файлов")).after
    sels = [s for s, _ in parse_branches(out[out.index("{count") : out.rindex("}") + 1])]
    assert sels[-1] == "other"
    assert "few" in sels and "many" in sels
    assert missing_plural_categories(out, "ru-RU") == {}


def test_rejects_a_branch_that_lost_its_number_slot() -> None:
    r = expand_message(SRC, RU, "ru-RU", FakeLLM("few: файла\nmany: файлов"))
    assert not r.ok
    assert "#" in (r.error or "")
    assert r.after == RU


def test_no_gap_is_a_no_op() -> None:
    llm = FakeLLM("should not be called")
    r = expand_message(
        SRC, "Importiert {count, plural, one {# Datei} other {# Dateien}}.", "de-DE", llm
    )
    assert r.after == "Importiert {count, plural, one {# Datei} other {# Dateien}}."
    assert not r.added
    assert llm.calls == 0


def test_unusable_reply_leaves_the_message_alone() -> None:
    r = expand_message(SRC, RU, "ru-RU", FakeLLM("I'm not sure about that."))
    assert not r.ok and r.after == RU


def test_provider_error_leaves_the_message_alone() -> None:
    r = expand_message(SRC, RU, "ru-RU", FakeLLM(fail=True))
    assert not r.ok and r.after == RU
    assert "unreachable" in (r.error or "")
