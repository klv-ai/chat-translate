from __future__ import annotations

from chat_translate.icu import (
    arguments_of,
    find_arguments,
    mask_icu,
    missing_plural_categories,
    parse_branches,
    selectors_of,
    translate_icu,
)


# A translator that returns something obviously different, so a test failure
# shows *where* structure leaked rather than passing on an identity function.
# The signature mirrors the real one: (source_fragment, masked_fragment).
def shout(_source: str, masked: str) -> str:
    return masked.upper()


def test_finds_simple_and_nested_arguments() -> None:
    args = find_arguments("Run {id} · idle {minutes}m")
    assert [a.name for a in args] == ["id", "minutes"]
    assert all(a.arg_type is None for a in args)

    nested = find_arguments("Imported {count, plural, one {# file} other {# files}}.")
    assert len(nested) == 1  # the whole plural is ONE argument, braces and all
    assert nested[0].name == "count"
    assert nested[0].arg_type == "plural"
    assert nested[0].has_submessages


def test_parses_plural_branches_in_order() -> None:
    raw = "{count, plural, one {# file} other {# files}}"
    assert parse_branches(raw) == [("one", "# file"), ("other", "# files")]


def test_masking_hides_every_argument_from_the_provider() -> None:
    m = mask_icu("Run {id} · idle {minutes}m")
    assert "{id}" not in m.carrier.masked
    assert "{minutes}" not in m.carrier.masked
    assert m.carrier.restore(m.carrier.masked) == "Run {id} · idle {minutes}m"


def test_argument_names_survive_a_translator_that_would_mangle_them() -> None:
    # The real failure this guards: TranslateGemma renders {minutes} as
    # {minutos}, and the renderer then prints the placeholder instead of a value.
    out = translate_icu("Run {id} · idle {minutes}m", shout)
    assert "{id}" in out
    assert "{minutes}" in out
    assert arguments_of(out) == {"id", "minutes"}


def test_plural_skeleton_is_rebuilt_not_translated() -> None:
    src = "Imported {count, plural, one {# file} other {# files}}."
    out = translate_icu(src, shout)

    # Selectors and the argument name are written by us, so they stay lowercase
    # even though the translator uppercases everything it is given.
    assert "{count, plural, one {" in out
    assert " other {" in out
    assert selectors_of(out) == {"count": ["one", "other"]}
    # …while the branch prose really did go through the translator.
    assert "FILE" in out and "FILES" in out


def test_hash_number_slot_survives() -> None:
    out = translate_icu("{count, plural, one {# file} other {# files}}", shout)
    assert out.count("#") == 2


def test_message_with_no_prose_is_not_sent_to_the_provider() -> None:
    calls: list[str] = []

    def spy(_source: str, masked: str) -> str:
        calls.append(masked)
        return masked

    assert translate_icu("{done}/{total}", spy) == "{done}/{total}"
    assert calls == []


def test_reports_plural_categories_the_target_needs_and_source_lacks() -> None:
    src = "Imported {count, plural, one {# file} other {# files}}."

    # Spanish/German are covered by one+other as authored.
    assert missing_plural_categories(src, "de-DE") == {}
    # Russian needs few/many that an English source never defined.
    assert missing_plural_categories(src, "ru") == {"count": ["few", "many"]}
    # Japanese needs only `other`, which is present.
    assert missing_plural_categories(src, "ja-JP") == {}


def test_plain_message_is_translated_whole() -> None:
    assert translate_icu("Delete folder", shout) == "DELETE FOLDER"


def test_recovers_a_sentinel_the_model_rewrote_as_ascii_brackets() -> None:
    # Observed with translategemma:12b: inside a plural branch it renders
    # ⟦PH0⟧ as [PH0]. Without recovery the token never restores and it ships.
    def mangle(_source: str, masked: str) -> str:
        return masked.replace("⟦", "[").replace("⟧", "]")

    out = translate_icu("{count, plural, one {# file} other {# files}}", mangle)
    assert "[0]" not in out
    assert out.count("#") == 2


def test_leaves_a_literal_bracket_number_in_copy_alone() -> None:
    # "[0]" that the source really contains must not be mistaken for a mangled
    # sentinel — recovery only fires for an index whose sentinel is missing.
    def passthrough(_source: str, masked: str) -> str:
        return masked

    assert translate_icu("Row [0] failed", passthrough) == "Row [0] failed"


def test_branch_translation_sees_its_own_source_not_the_whole_message() -> None:
    seen: list[str] = []

    def spy(source: str, masked: str) -> str:
        seen.append(source)
        return masked

    translate_icu("Imported {count, plural, one {# file} other {# files}}.", spy)
    # The carrier, then each branch as its own fragment — so a caller can judge
    # punctuation drift per fragment.
    assert "# file" in seen and "# files" in seen


def test_placeholder_index_is_prefixed_so_it_does_not_read_as_a_number() -> None:
    # The failure this guards is subtle: a bare "⟦0⟧" is not mangled by the
    # tokenizer — it is faithfully TRANSLATED, because to the model it looks
    # like the number zero. "⟦0⟧% match" came back "0% de coincidencia" on
    # translategemma:12b. A prefixed index reads as a name and is copied.
    m = mask_icu("{pct}% match")
    assert "⟦PH0⟧" in m.carrier.masked
    assert "⟦0⟧" not in m.carrier.masked


def test_restores_a_number_slot_the_model_dropped_from_a_plural_branch() -> None:
    # Branches are tiny, and on short ones the model returns just the noun:
    # "⟦PH0⟧ attachments" came back "Anhänge", losing the count entirely.
    def drop_hash(_source: str, masked: str) -> str:
        return masked.replace("⟦PH0⟧", "").strip() or "X"

    out = translate_icu("{count, plural, one {# file} other {# files}}", drop_hash)
    assert out.count("#") == 2, out


def test_hash_repair_keeps_the_side_the_source_used() -> None:
    def drop_hash(_source: str, masked: str) -> str:
        return masked.replace("⟦PH0⟧", "").strip() or "X"

    trailing = translate_icu("{count, plural, other {files #}}", drop_hash)
    assert trailing.rstrip("}").rstrip().endswith("#"), trailing


def test_hash_repair_does_not_fire_when_the_slot_survived() -> None:
    def keep(_source: str, masked: str) -> str:
        return masked

    out = translate_icu("{count, plural, one {# file} other {# files}}", keep)
    assert out.count("#") == 2


def test_moves_a_number_slot_back_to_the_side_the_source_used() -> None:
    # "# sources" came back "Sources #": the model treats # as decoration and
    # re-places it. A third of plural branches per locale had this.
    def move_hash(_source: str, masked: str) -> str:
        return masked.replace("⟦PH0⟧ ", "") + " ⟦PH0⟧"

    out = translate_icu("{count, plural, other {# files}}", move_hash)
    inner = out[out.index("other {") + 7 : out.rindex("}}")]
    assert inner.strip().startswith("#"), out
