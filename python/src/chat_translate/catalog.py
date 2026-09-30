"""Translate a UI message catalog with a TranslationProvider.

Operates on natural-key catalogs, where each key is the source string and the
value is its translation::

    { "Delete folder": "Eliminar carpeta" }

Each result is checked by :func:`verify`; an entry that fails is reported with
an error and excluded from :attr:`CatalogResult.translated`. Successful output
is normalised by :func:`tidy`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from .icu import (
    arguments_of,
    has_stray_sentinel,
    hash_slots_of,
    missing_plural_categories,
    selectors_of,
    translate_icu,
)
from .masking import Sentinels, make_sentinels
from .provider.base import TranslateOptions, TranslationError, TranslationProvider

#: Sentinels for protected terms; distinct from the ICU placeholder sentinels.
TERM_SENTINELS = make_sentinels(chr(0x27E6) + "TM", chr(0x27E7))


def protect_terms(text: str, terms: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Replace each occurrence of a protected term with a sentinel.

    Longer terms are replaced first, so a term containing another is masked
    whole. Returns the masked text and a token -> term mapping for
    :func:`restore_terms`.
    """
    mapping: dict[str, str] = {}
    out = text
    for i, term in enumerate(sorted(terms, key=len, reverse=True)):
        if not term or term not in out:
            continue
        token = f"{TERM_SENTINELS.open}{i}{TERM_SENTINELS.close}"
        out = out.replace(term, token)
        mapping[token] = term
    return out, mapping


def restore_terms(text: str, mapping: dict[str, str]) -> str:
    out = text
    for token, term in mapping.items():
        out = out.replace(token, term)
    return out


#: Leading phrases removed by :func:`tidy` (compared case-insensitively).
_PREAMBLES = (
    "translation:",
    "here is the translation:",
    "sure, here is the translation:",
    "traducción:",
    "traduccion:",
)
_TERMINAL = ".!?:;…。！？⋯"
#: Languages that capitalise common nouns; :func:`tidy` does not mirror case for them.
_CAPITALISES_NOUNS = frozenset({"de", "lb"})


def tidy(source: str, translated: str, target_lang: str | None = None) -> str:
    """Normalise a translated UI string against its source.

    In order:

    * removes a leading preamble (see ``_PREAMBLES``);
    * removes a whole-string quote pair the source does not have;
    * removes terminal punctuation the source does not end with;
    * when *target_lang* is given and does not capitalise nouns, lowercases the
      first letter (after any leading ``#``) if the source's is lowercase;
    * matches the source's trailing ellipsis form (``…`` vs ``...``);
    * restores the source's leading and trailing whitespace.
    """
    out = translated.strip()

    lowered = out.lower()
    for pre in _PREAMBLES:
        if lowered.startswith(pre):
            out = out[len(pre) :].lstrip()
            break

    for open_q, close_q in (('"', '"'), ("'", "'"), ("«", "»"), ("“", "”")):
        if len(out) >= 2 and out.startswith(open_q) and out.endswith(close_q):
            if not (source.startswith(open_q) and source.endswith(close_q)):
                out = out[1:-1].strip()
            break

    if out and out[-1] in _TERMINAL and (not source or source[-1] not in _TERMINAL):
        out = out[:-1].rstrip()

    if target_lang and target_lang.split("-")[0].lower() not in _CAPITALISES_NOUNS:
        body = out.lstrip("# ").lstrip()
        src_body = source.lstrip("# ").lstrip()
        if body[:1].isupper() and src_body[:1].islower():
            at = out.index(body[0])
            out = out[:at] + body[0].lower() + out[at + 1 :]

    if source.rstrip().endswith("…") and out.rstrip().endswith("..."):
        out = out.rstrip()[:-3].rstrip() + "…"
    elif source.rstrip().endswith("...") and out.rstrip().endswith("…"):
        out = out.rstrip()[:-1].rstrip() + "..."

    return source[: len(source) - len(source.lstrip())] + out + source[len(source.rstrip()) :]


def _has_prose_outside(text: str, sentinels: Sentinels) -> bool:
    """True if any letter remains once sentinels are removed."""
    return any(ch.isalpha() for ch in sentinels.placeholder_re.sub("", text))


def _looks_like_a_dictionary_entry(source: str, translated: str) -> bool:
    """True if *translated* looks like a list of alternatives, not one translation.

    Fires when the output has a newline the source lacks, or when a source of
    at most three words produces at least ``max(8, 5 × source words)`` words.
    """
    if not translated.strip():
        return False
    if "\n" in translated and "\n" not in source:
        return True
    src_words = len(source.split())
    out_words = len(translated.split())
    return src_words <= 3 and out_words >= max(8, src_words * 5)


@dataclass(slots=True)
class EntryResult:
    key: str
    #: The translation, or the source key when ``error`` is set.
    value: str
    #: Why the entry was rejected, if it was.
    error: str | None = None
    #: Non-fatal notes, e.g. plural categories the target needs but the source lacks.
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None


def verify(source: str, translated: str, protected: Sequence[str] = ()) -> str | None:
    """Return a reason the translation is unusable, or None if it passes.

    Checks, in order: argument names, plural/select selectors, ``#`` slot
    counts, protected terms present in the source, dictionary-style output,
    leftover sentinels, and emptiness. Does not check meaning.
    """
    src_args = arguments_of(source)
    out_args = arguments_of(translated)
    if src_args != out_args:
        lost = sorted(src_args - out_args)
        gained = sorted(out_args - src_args)
        parts = []
        if lost:
            parts.append(f"lost {lost}")
        if gained:
            parts.append(f"invented {gained}")
        return "placeholders " + ", ".join(parts)

    src_sel = selectors_of(source)
    out_sel = selectors_of(translated)
    if src_sel != out_sel:
        return f"plural selectors changed: {src_sel} -> {out_sel}"

    src_hashes = hash_slots_of(source)
    out_hashes = hash_slots_of(translated)
    if src_hashes != out_hashes:
        return f"plural number slots (#) changed: {src_hashes} -> {out_hashes}"

    dropped = [t for t in protected if t in source and t not in translated]
    if dropped:
        return f"protected term(s) lost: {dropped}"

    if _looks_like_a_dictionary_entry(source, translated):
        return "reads as a dictionary entry, not a translation"

    if has_stray_sentinel(translated):
        return "a masked placeholder was mangled and could not be restored"

    if not translated.strip():
        return "empty translation"
    return None


@dataclass(slots=True)
class CatalogResult:
    entries: list[EntryResult]
    elapsed_seconds: float

    @property
    def translated(self) -> dict[str, str]:
        """Key -> translation for entries that passed verification."""
        return {e.key: e.value for e in self.entries if e.ok}

    @property
    def failures(self) -> list[EntryResult]:
        return [e for e in self.entries if not e.ok]

    @property
    def flagged(self) -> list[EntryResult]:
        return [e for e in self.entries if e.notes]


def translate_catalog(
    keys: Iterable[str],
    provider: TranslationProvider,
    target_lang: str,
    *,
    source_lang: str = "en",
    protected: Sequence[str] = (),
    attempts: int = 3,
    retry_delay_seconds: float = 2.0,
    on_progress: Callable[[int, EntryResult], None] | None = None,
    on_checkpoint: Callable[[dict[str, str]], None] | None = None,
    checkpoint_every: int = 25,
) -> CatalogResult:
    """Translate each key in order and verify the result.

    Requests use ``content_kind="ui_label"``. *protected* terms are masked
    before translation and restored afterwards; a fragment that is only
    protected terms is returned unchanged without calling the provider. A
    retryable :class:`TranslationError` is retried up to *attempts* times with a
    linearly increasing delay.

    *on_progress* is called after each entry with its index and result.
    *on_checkpoint* is called with all successful translations so far every
    *checkpoint_every* entries and once at the end.
    """
    started = time.monotonic()
    options = TranslateOptions(
        target_lang=target_lang, source_lang=source_lang, content_kind="ui_label"
    )

    def call(source_fragment: str, masked: str) -> str:
        guarded, terms = protect_terms(masked, protected)
        if terms and not _has_prose_outside(guarded, TERM_SENTINELS):
            return source_fragment
        last: TranslationError | None = None
        for attempt in range(attempts):
            try:
                raw = provider.translate(guarded, options).text
                return tidy(source_fragment, restore_terms(raw, terms), target_lang)
            except TranslationError as exc:
                if not exc.retryable:
                    raise
                last = exc
                if attempt < attempts - 1:
                    time.sleep(retry_delay_seconds * (attempt + 1))
        raise last if last else TranslationError("translation failed", "catalog")

    entries: list[EntryResult] = []
    for i, key in enumerate(keys):
        notes = [
            f"{target_lang} needs plural categories {list(cats)} for "
            f"{{{name}}} that the source does not define"
            for name, cats in missing_plural_categories(key, target_lang).items()
        ]
        try:
            candidate = translate_icu(key, call)
            reason = verify(key, candidate, protected)
            result = (
                EntryResult(key=key, value=key, error=reason, notes=notes)
                if reason
                else EntryResult(key=key, value=candidate, notes=notes)
            )
        except TranslationError as exc:
            result = EntryResult(key=key, value=key, error=str(exc), notes=notes)
        entries.append(result)
        if on_progress:
            on_progress(i, result)
        if on_checkpoint and checkpoint_every and (i + 1) % checkpoint_every == 0:
            on_checkpoint({e.key: e.value for e in entries if e.ok})

    if on_checkpoint:
        on_checkpoint({e.key: e.value for e in entries if e.ok})
    return CatalogResult(entries=entries, elapsed_seconds=time.monotonic() - started)
