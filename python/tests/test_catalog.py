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
        self.options: list[TranslateOptions] = []

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
        self.options.append(options)
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


def test_catalog_requests_use_the_ui_label_content_kind() -> None:
    provider = FakeProvider()
    translate_catalog(["Save"], provider, "es-ES")
    assert provider.options[0].content_kind == "ui_label"


def test_tidy_preserves_source_leading_and_trailing_spacing() -> None:
    assert tidy(" (container)", "(contenedor)") == " (contenedor)"


def test_verify_rejects_a_lost_placeholder() -> None:
    assert verify("Run {id}", "Ejecutar") is not None
    assert "lost" in verify("Run {id}", "Ejecutar")  # type: ignore[operator]
    assert verify("Run {id}", "Ejecutar {id}") is None


def test_verify_rejects_a_renamed_placeholder() -> None:
    reason = verify("idle {minutes}m", "inactivo {minutos}m")
    assert reason is not None and "invented" in reason


def test_verify_rejects_changed_plural_selectors() -> None:
    src = "{count, plural, one {# file} other {# files}}"
    bad = "{count, plural, uno {# archivo} otros {# archivos}}"
    assert verify(src, bad) is not None
    good = "{count, plural, one {# archivo} other {# archivos}}"
    assert verify(src, good) is None


def test_failed_entries_are_reported_and_excluded_from_translated() -> None:
    provider = FakeProvider(fail=True)
    result = translate_catalog(["Delete folder"], provider, "es-ES", retry_delay_seconds=0)

    entry = result.entries[0]
    assert not entry.ok
    assert entry.value == "Delete folder"
    assert result.failures == [entry]
    assert result.translated == {}


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
    assert entry.ok
    assert entry.notes and "few" in entry.notes[0]


def test_result_maps_keys_to_values() -> None:
    provider = FakeProvider({"Save": "Guardar"})
    result = translate_catalog(["Save"], provider, "es-ES")
    assert result.translated == {"Save": "Guardar"}
    assert isinstance(result.entries[0], EntryResult)


def test_tidy_does_not_truncate_an_ellipsis() -> None:
    assert tidy("Loading…", "Cargando...") == "Cargando…"
    assert not tidy("Loading…", "Cargando...").endswith("..")


def test_verify_rejects_output_with_a_stranded_sentinel() -> None:
    assert verify("{count, plural, other {# files}}", "⟦0⟧ archivos") is not None


def test_branch_punctuation_is_tidied_against_the_branch_source() -> None:
    # The branch source "# file" has no period, although the whole message does.
    class BranchDrift(FakeProvider):
        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            self.seen.append(text)
            return TranslateResult(text=f"{text} archivo.", detected_source_lang="en")

    result = translate_catalog(
        ["Imported {count, plural, one {# file} other {# files}}."], BranchDrift(), "es-ES"
    )
    value = result.entries[0].value
    assert "archivo.}" not in value  # no stray period inside a branch


def test_retries_a_retryable_failure() -> None:
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
    assert HardFail.calls == 1


def test_checkpoints_every_n_entries_and_at_the_end() -> None:
    snapshots: list[dict[str, str]] = []
    keys = [f"Key {i}" for i in range(5)]

    translate_catalog(
        keys,
        FakeProvider(),
        "es-ES",
        on_checkpoint=lambda done: snapshots.append(dict(done)),
        checkpoint_every=2,
    )

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
    src = "{count, plural, one {# file} other {# files}}"
    bad = "{count, plural, one {Archivo 0} other {0 archivos}}"
    reason = verify(src, bad)
    assert reason is not None and "#" in reason
    assert verify(src, "{count, plural, one {# archivo} other {# archivos}}") is None


# Short sources whose "translation" lists alternatives rather than giving one.
GLOSS_CASES = [
    (
        "Flagged",
        "Marcado. Señalizado. Identificado. (Dependiendo del contexto, "
        'también podría traducirse como "pendiente de revisión")',
    ),
    ("Chatter", "Charla.\nParloteo.\nCotilleo.\nConversación.\nCháchara."),
    ("Get started", "Comience aquí. O bien: Empezar. O también: Póngase en marcha"),
    # No newline; caught by the word-count ratio.
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


def test_does_not_reject_a_translation_that_needs_more_words() -> None:
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
        ["Conversations inside stay in Widgetbox, just ungrouped."],
        provider,
        "es-ES",
        protected=["Widgetbox"],
    )
    assert "Widgetbox" not in provider.seen[0]


def test_protected_terms_come_back_verbatim() -> None:
    class Translator(FakeProvider):
        def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
            self.seen.append(text)
            return TranslateResult(text=text.replace("in", "en"), detected_source_lang="en")

    result = translate_catalog(
        ["Open in Widgetbox"],
        Translator(),
        "es-ES",
        protected=["Widgetbox"],
    )
    assert "Widgetbox" in result.entries[0].value


def test_longer_protected_term_wins_over_a_substring() -> None:
    masked, mapping = protect_terms("Acme Notes and Notes", ["Notes", "Acme Notes"])
    assert "Acme Notes" not in masked
    assert restore_terms(masked, mapping) == "Acme Notes and Notes"


def test_a_string_that_is_only_a_protected_term_is_never_sent() -> None:
    provider = FakeProvider()
    result = translate_catalog(["Widgetbox"], provider, "es-ES", protected=["Widgetbox"])
    assert provider.seen == []
    assert result.entries[0].value == "Widgetbox"


def test_verify_rejects_a_translation_that_dropped_a_protected_term() -> None:
    assert verify("e.g., Acme Pro", "Ejemplo", protected=["Acme Pro"]) is not None
    assert verify("e.g., Acme Pro", "p. ej., Acme Pro", protected=["Acme Pro"]) is None


def test_tidy_mirrors_the_source_ellipsis_form() -> None:
    assert tidy("Loading…", "Cargando...") == "Cargando…"
    assert tidy("Search…", "Buscar ...") == "Buscar…"
    # A source that really uses three dots keeps them.
    assert tidy("Wait...", "Espere…") == "Espere..."
    # Nothing to mirror.
    assert tidy("Save", "Guardar") == "Guardar"


def test_tidy_mirrors_leading_case_for_plural_branches() -> None:
    assert tidy("# file", "# Файл", "ru-RU") == "# файл"
    assert tidy("# sources", "# Sources", "fr-FR") == "# sources"
    # German capitalises nouns — leave it alone.
    assert tidy("# file", "# Datei", "de-DE") == "# Datei"
    # A source that is itself capitalised keeps its capital.
    assert tidy("# Files", "# Dateien", "nl-NL") == "# Dateien"
    # No language given: no case mirroring.
    assert tidy("# file", "# Файл") == "# Файл"
