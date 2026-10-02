from __future__ import annotations

from chat_translate.catalog import (
    EntryResult,
    protect_terms,
    restore_terms,
    tidy,
    translate_catalog,
    verify,
)
from chat_translate.provider.base import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
)


class FakeProvider(BaseTranslationProvider):
    """Returns whatever the test scripted, so structure handling is testable
    without a model. Records what it was asked to translate."""

    name = "fake"

    def __init__(self, replies: dict[str, str] | None = None, *, fail: bool = False) -> None:
        self.replies = replies or {}
        self.fail = fail
        self.seen: list[str] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=False,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        self.seen.append(text)
        if self.fail:
            raise TranslationError("boom", self.name, retryable=True)
        return TranslateResult(text=self.replies.get(text, f"<{text}>"), detected_source_lang="en")

    def health_check(self) -> bool:
        return True


def test_tidy_removes_a_period_the_source_never_had() -> None:
    assert tidy("Delete folder", "Eliminar carpeta.") == "Eliminar carpeta"
    # …but keeps one the source did have.
    assert tidy("Are you sure?", "¿Estás seguro?") == "¿Estás seguro?"


def test_tidy_strips_preamble_and_wrapping_quotes() -> None:
    assert tidy("Save", "Translation: Guardar") == "Guardar"
    assert tidy("Save", '"Guardar"') == "Guardar"
    # A source that is itself quoted keeps its quotes.
    assert tidy('"After"', '"Después"') == '"Después"'


def test_tidy_preserves_deliberate_source_spacing() -> None:
    # " (container)" is padded on purpose — the layout depends on it.
    assert tidy(" (container)", "(contenedor)") == " (contenedor)"


def test_verify_rejects_a_lost_placeholder() -> None:
    assert verify("Run {id}", "Ejecutar") is not None
    assert "lost" in verify("Run {id}", "Ejecutar")  # type: ignore[operator]
    assert verify("Run {id}", "Ejecutar {id}") is None


def test_verify_rejects_a_renamed_placeholder() -> None:
    # The exact TranslateGemma failure: {minutes} comes back as {minutos}.
    reason = verify("idle {minutes}m", "inactivo {minutos}m")
    assert reason is not None and "invented" in reason


def test_verify_rejects_changed_plural_selectors() -> None:
    src = "{count, plural, one {# file} other {# files}}"
    bad = "{count, plural, uno {# archivo} otros {# archivos}}"
    assert verify(src, bad) is not None
    good = "{count, plural, one {# archivo} other {# archivos}}"
    assert verify(src, good) is None


def test_failed_entries_keep_english_rather_than_shipping_broken_output() -> None:
    provider = FakeProvider(fail=True)
    result = translate_catalog(["Delete folder"], provider, "es-ES")

    entry = result.entries[0]
    assert not entry.ok
    assert entry.value == "Delete folder"  # fell back, did not ship a break
    assert result.failures == [entry]


def test_placeholders_are_masked_before_reaching_the_provider() -> None:
    provider = FakeProvider()
    translate_catalog(["Run {id} · idle {minutes}m"], provider, "es-ES")

    sent = provider.seen[0]
    assert "{id}" not in sent and "{minutes}" not in sent
    assert "⟦PH0⟧" in sent and "⟦PH1⟧" in sent


def test_plural_gap_is_reported_without_failing_the_entry() -> None:
    provider = FakeProvider()
    src = "Imported {count, plural, one {# file} other {# files}}."
    result = translate_catalog([src], provider, "ru")

    entry = result.entries[0]
    assert entry.ok  # a gap is review work, not a failure
    assert entry.notes and "few" in entry.notes[0]


def test_result_maps_keys_to_values() -> None:
    provider = FakeProvider({"Save": "Guardar"})
    result = translate_catalog(["Save"], provider, "es-ES")
    assert result.translated == {"Save": "Guardar"}
    assert isinstance(result.entries[0], EntryResult)


def test_tidy_does_not_amputate_an_ellipsis() -> None:
    # "Loading…" ends in an ellipsis, so "Cargando..." is not sentence drift —
    # treating it as such produced "Cargando..". It is normalised to the
    # source's form (see test_tidy_mirrors_the_source_ellipsis_form), never
    # truncated.
    assert tidy("Loading…", "Cargando...") == "Cargando…"
    assert not tidy("Loading…", "Cargando...").endswith("..")


def test_verify_rejects_output_with_a_stranded_sentinel() -> None:
    assert verify("{count, plural, other {# files}}", "⟦0⟧ archivos") is not None


def test_branch_punctuation_drift_is_tidied_per_branch() -> None:
    # The model adds a period inside "# file" -> "# archivo."; the branch source
    # has none, so it must be removed even though the whole message ends in ".".
    class BranchDrift(FakeProvider):
        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            self.seen.append(text)
            return TranslateResult(text=f"{text} archivo.", detected_source_lang="en")

    result = translate_catalog(
        ["Imported {count, plural, one {# file} other {# files}}."], BranchDrift(), "es-ES"
    )
    value = result.entries[0].value
    assert "archivo.}" not in value  # no stray period inside a branch


def test_retries_a_transient_failure_instead_of_shipping_english() -> None:
    # A read timeout while the model loads is retryable; losing the entry to it
    # would ship English for a string that translates fine on the next try.
    class FlakyProvider(FakeProvider):
        calls = 0

        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            FlakyProvider.calls += 1
            if FlakyProvider.calls < 3:
                raise TranslationError("timed out", self.name, retryable=True)
            return TranslateResult(text="Guardar", detected_source_lang="en")

    result = translate_catalog(
        ["Save"], FlakyProvider(), "es-ES", attempts=3, retry_delay_seconds=0
    )
    assert result.entries[0].ok
    assert result.entries[0].value == "Guardar"
    assert FlakyProvider.calls == 3


def test_does_not_retry_a_permanent_failure() -> None:
    class HardFail(FakeProvider):
        calls = 0

        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            HardFail.calls += 1
            raise TranslationError("bad request", self.name, retryable=False)

    result = translate_catalog(["Save"], HardFail(), "es-ES", attempts=3, retry_delay_seconds=0)
    assert not result.entries[0].ok
    assert HardFail.calls == 1  # no point hammering a 400


def test_checkpoints_partial_progress_so_a_kill_does_not_lose_the_run() -> None:
    # A catalog run is thousands of sequential model calls. Writing only at the
    # end means an OOM or a Ctrl-C throws all of it away.
    snapshots: list[dict[str, str]] = []
    keys = [f"Key {i}" for i in range(5)]

    translate_catalog(
        keys,
        FakeProvider(),
        "es-ES",
        on_checkpoint=lambda done: snapshots.append(dict(done)),
        checkpoint_every=2,
    )

    # After 2, after 4, and a final flush.
    assert [len(s) for s in snapshots] == [2, 4, 5]


def test_checkpoint_never_contains_a_failed_entry() -> None:
    snapshots: list[dict[str, str]] = []
    translate_catalog(
        ["Save"],
        FakeProvider(fail=True),
        "es-ES",
        attempts=1,
        retry_delay_seconds=0,
        on_checkpoint=lambda done: snapshots.append(dict(done)),
    )
    assert snapshots == [{}]


def test_verify_rejects_a_lost_plural_number_slot() -> None:
    # translategemma:4b rewrites "one {# file}" as "one {Archivo 0}", which
    # renders a literal 0 where the count belongs. `#` is not an argument, so
    # only an explicit slot check catches it.
    src = "{count, plural, one {# file} other {# files}}"
    bad = "{count, plural, one {Archivo 0} other {0 archivos}}"
    reason = verify(src, bad)
    assert reason is not None and "#" in reason
    assert verify(src, "{count, plural, one {# archivo} other {# archivos}}") is None


# Real outputs from translategemma:12b that shipped into a UI before the
# UI-label prompt existed. Each renders a paragraph inside a button.
GLOSS_CASES = [
    (
        "Flagged",
        "Marcado. Señalizado. Identificado. (Dependiendo del contexto, "
        'también podría traducirse como "pendiente de revisión")',
    ),
    ("Chatterbox", "Charlatán.\nBocazas.\nCotilla.\nPersona locuaz.\nHabladora."),
    ("Get started", "Comience aquí. O bien: Empezar. O también: Póngase en marcha"),
    # No newlines at all — only the length ratio catches this one.
    (
        "hosted",
        "alojado/a organizado ofrecido presentado celebrado acogido "
        "gestionado administrado facilitado sostenido financiado impulsado "
        "promovido desarrollado distribuido transmitido publicado archivado",
    ),
]


def test_rejects_dictionary_style_output() -> None:
    for source, gloss in GLOSS_CASES:
        assert verify(source, gloss) is not None, f"missed gloss for {source!r}"


def test_does_not_reject_an_honest_translation_that_needs_more_words() -> None:
    # Spanish is routinely longer than English; the guard must not fire on that.
    for source, good in [
        ("Undo", "Deshacer"),
        ("Skip", "Omitir"),
        ("Sign out", "Cerrar sesión"),
        ("Delete folder", "Eliminar carpeta"),
        ("Save", "Guardar"),
        ("Inbox", "Bandeja de entrada"),
    ]:
        assert verify(source, good) is None, f"false positive on {source!r}"


def test_protected_terms_are_never_sent_to_the_provider() -> None:
    provider = FakeProvider()
    translate_catalog(
        ["Conversations inside stay in Chatterbox, just ungrouped."],
        provider,
        "es-ES",
        protected=["Chatterbox"],
    )
    assert "Chatterbox" not in provider.seen[0]


def test_protected_terms_come_back_verbatim() -> None:
    # The model faithfully translates a brand name unless it is masked:
    # "Chatterbox" -> "Charlatán", "Similie" -> "símil".
    class Translator(FakeProvider):
        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            self.seen.append(text)
            return TranslateResult(text=text.replace("in", "en"), detected_source_lang="en")

    result = translate_catalog(
        ["Open in Chatterbox"],
        Translator(),
        "es-ES",
        protected=["Chatterbox"],
    )
    assert "Chatterbox" in result.entries[0].value


def test_longer_protected_term_wins_over_a_substring() -> None:
    masked, mapping = protect_terms("Klavi Experts and Experts", ["Experts", "Klavi Experts"])
    assert "Klavi Experts" not in masked
    assert restore_terms(masked, mapping) == "Klavi Experts and Experts"


def test_a_string_that_is_only_a_protected_term_is_never_sent() -> None:
    # "Chatterbox" masks down to a lone sentinel; asked to translate that, the
    # model invented "Guardar" ("Save") and it shipped.
    provider = FakeProvider()
    result = translate_catalog(["Chatterbox"], provider, "es-ES", protected=["Chatterbox"])
    assert provider.seen == []
    assert result.entries[0].value == "Chatterbox"


def test_verify_rejects_a_translation_that_dropped_a_protected_term() -> None:
    assert verify("e.g., Similie", "Ejemplo", protected=["Similie"]) is not None
    assert verify("e.g., Similie", "p. ej., Similie", protected=["Similie"]) is None


def test_tidy_mirrors_the_source_ellipsis_form() -> None:
    # The model renders "…" as three dots about half the time. Harmless alone,
    # but it left es-ES with 136 entries one way and 1 the other.
    assert tidy("Loading…", "Cargando...") == "Cargando…"
    assert tidy("Search…", "Buscar ...") == "Buscar…"
    # A source that really uses three dots keeps them.
    assert tidy("Wait...", "Espere…") == "Espere..."
    # Nothing to mirror.
    assert tidy("Save", "Guardar") == "Guardar"


def test_tidy_mirrors_leading_case_for_plural_branches() -> None:
    # A branch is two tokens in isolation, so the UI-label prompt makes the
    # model Title-Case it: "# file" -> "# Файл". Wrong inside a sentence.
    assert tidy("# file", "# Файл", "ru-RU") == "# файл"
    assert tidy("# sources", "# Sources", "fr-FR") == "# sources"
    # German capitalises nouns — leave it alone.
    assert tidy("# file", "# Datei", "de-DE") == "# Datei"
    # A source that is itself capitalised keeps its capital.
    assert tidy("# Files", "# Dateien", "nl-NL") == "# Dateien"
    # No language given: no case mirroring.
    assert tidy("# file", "# Файл") == "# Файл"


# ---------------------------------------------------------------------------
# Stand-in names: the retry for a model that drops a leading sentinel
# ---------------------------------------------------------------------------

from chat_translate.catalog import from_stand_ins, to_stand_ins  # noqa: E402


def test_stand_ins_round_trip() -> None:
    named, mapping = to_stand_ins("⟦PH0⟧ shared by ⟦PH1⟧")
    assert named == "Zarvex shared by Quilmor"
    assert from_stand_ins("Zarvex, compartido por Quilmor.", mapping, "⟦PH0⟧ shared by ⟦PH1⟧") == \
        "⟦PH0⟧, compartido por ⟦PH1⟧."


def test_stand_ins_reject_a_dropped_repeated_or_transliterated_name() -> None:
    masked = "⟦PH0⟧ installed."
    _, mapping = to_stand_ins(masked)
    assert from_stand_ins("Instalado.", mapping, masked) is None
    assert from_stand_ins("Zarvex instalado, Zarvex.", mapping, masked) is None
    assert from_stand_ins("ザルベックス インストール済み", mapping, masked) is None


def test_stand_ins_reject_a_declined_name_but_not_cjk_or_a_glued_source() -> None:
    masked = "Connected to ⟦PH0⟧"
    _, mapping = to_stand_ins(masked)
    assert from_stand_ins("Подключено к Zarvexу", mapping, masked) is None
    assert from_stand_ins("Zarvexに接続済み", mapping, masked) == "⟦PH0⟧に接続済み"
    glued = "⟦PH0⟧d"
    _, mapping = to_stand_ins(glued)
    assert from_stand_ins("Zarvexd", mapping, glued) == "⟦PH0⟧d"


def test_stand_ins_unusable_without_sentinels_or_when_a_name_is_taken() -> None:
    assert to_stand_ins("(one per line)") is None
    assert to_stand_ins("⟦PH0⟧ met Zarvex") is None


def test_a_lost_leading_placeholder_is_recovered_with_stand_in_names() -> None:
    provider = FakeProvider({
        "⟦PH0⟧ installed.": "Instalado.",          # the sentinel is dropped
        "Zarvex installed.": "Zarvex instalado.",  # the name is kept
    })
    result = translate_catalog(["{language} installed."], provider, "es-ES")

    entry = result.entries[0]
    assert entry.ok and entry.stand_ins
    assert entry.value == "{language} instalado."
    assert result.recovered == [entry]


def test_a_failure_the_names_cannot_fix_is_not_retried() -> None:
    provider = FakeProvider({"Flagged": "Marcado. Señalizado. Identificado. Depende del contexto y del uso."})
    result = translate_catalog(["Flagged"], provider, "es-ES")

    assert not result.entries[0].ok
    assert provider.seen == ["Flagged"]  # one call: no stand-in retry


def test_verify_rejects_a_placeholder_fused_into_a_compound() -> None:
    assert verify("{errors} errors · idle", "{errors}-Fehler · Inaktiv") is not None
    assert verify(
        "{count, plural, one {# result} other {# results}}",
        "{count, plural, one {#-Ergebnis} other {#-Ergebnisse}}",
    ) is not None
    # Fine when the source compounds too, or nothing is glued
    assert verify("{n}-day trial", "{n}-Tage-Test") is None
    assert verify("{errors} errors", "{errors} Fehler") is None


def test_a_plural_number_slot_is_never_retried_with_a_stand_in() -> None:
    # The leading {name} is dropped, which earns the stand-in retry; the
    # plural branches inside it must still never see a name for their `#`.
    provider = FakeProvider({"⟦PH0⟧ has ⟦PH1⟧": "Tiene ⟦PH1⟧", "Zarvex has Quilmor": "Zarvex tiene Quilmor"})
    result = translate_catalog(["{name} has {count, plural, one {# file} other {# files}}"], provider, "es-ES")

    assert not result.entries[0].ok
    assert "Zarvex has Quilmor" in provider.seen          # the retry ran
    assert not any("Zarvex file" in s for s in provider.seen)
