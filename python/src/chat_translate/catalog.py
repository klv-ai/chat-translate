"""Translate a UI message catalog (natural-key JSON) with a TranslationProvider.

Built for svelte-i18n style catalogs, where the KEY is the English source string
and the value is the translation:

    { "Delete folder": "Eliminar carpeta" }

Two things separate this from chat translation:

* **Structure must survive exactly.** A chat message that loses a placeholder is
  ugly; a UI message that loses one renders ``{minutes}`` to a user, or throws.
  Every result is verified against its source (see :func:`verify`) and a failure
  keeps the English rather than shipping a broken string.

* **The model drifts on short labels.** TranslateGemma at temperature 0 renders
  "Delete folder" as "Eliminar carpeta." — a sentence period on a button. UI
  copy is not prose, so :func:`tidy` removes the additions the model makes and
  the source did not have.

Run it:

    python -m chat_translate.catalog en-US.json es-ES --out es-ES.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .icu import (
    arguments_of,
    has_stray_sentinel,
    hash_slots_of,
    missing_plural_categories,
    selectors_of,
    translate_icu,
)
from .masking import LLM_SENTINELS, Sentinels, make_sentinels
from .provider.base import TranslateOptions, TranslationError, TranslationProvider
from .provider.ollama import OllamaConfig, OllamaProvider

#: Separate sentinel space from the ICU one so indices cannot collide.
TERM_SENTINELS = make_sentinels(chr(0x27E6) + "TM", chr(0x27E7))


def protect_terms(text: str, terms: Sequence[str]) -> tuple[str, dict[str, str]]:
    """Mask product and brand names so the model copies them through.

    Without this a translator does its job and renders the name as a word:
    "Chatterbox" becomes "Charlatán", "Similie" becomes "símil". Longest term
    first so "Klavi Experts" is not half-consumed by "Experts".
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


#: Preambles an instruction-tuned model adds despite being told not to.
_PREAMBLES = (
    "translation:",
    "here is the translation:",
    "sure, here is the translation:",
    "traducción:",
    "traduccion:",
)
_TERMINAL = ".!?:;…。！？⋯"
#: Languages that capitalise common nouns, where mirroring English case is wrong.
_CAPITALISES_NOUNS = frozenset({"de", "lb"})


def tidy(source: str, translated: str, target_lang: str | None = None) -> str:
    """Strip what the model added and the source did not have.

    Deliberately conservative: it only removes a *leading* preamble, a wrapping
    quote pair, and terminal punctuation that the source lacked. It never adds
    anything and never touches the middle of the string.
    """
    out = translated.strip()

    lowered = out.lower()
    for pre in _PREAMBLES:
        if lowered.startswith(pre):
            out = out[len(pre) :].lstrip()
            break

    # A whole-string quote wrap, when the source was not quoted.
    for open_q, close_q in (('"', '"'), ("'", "'"), ("«", "»"), ("“", "”")):
        if len(out) >= 2 and out.startswith(open_q) and out.endswith(close_q):
            if not (source.startswith(open_q) and source.endswith(close_q)):
                out = out[1:-1].strip()
            break

    # Sentence punctuation on a label that never had it.
    if out and out[-1] in _TERMINAL and (not source or source[-1] not in _TERMINAL):
        out = out[:-1].rstrip()

    # Mirror the source's leading letter case. A plural branch is translated in
    # isolation — the provider sees "# file", two tokens — and the UI-label
    # prompt tells it that is a button or menu item, so it Title-Cases the
    # noun: 24-30 of ~34 branches per locale came back "# Файл", "# Source".
    # Inside a sentence that is simply wrong.
    #
    # Skipped for German, which capitalises every noun, and harmless for
    # scripts without case.
    if target_lang and target_lang.split("-")[0].lower() not in _CAPITALISES_NOUNS:
        body = out.lstrip("# ").lstrip()
        src_body = source.lstrip("# ").lstrip()
        if body[:1].isupper() and src_body[:1].islower():
            at = out.index(body[0])
            out = out[:at] + body[0].lower() + out[at + 1 :]

    # Mirror the source's ellipsis form. The model renders "…" (U+2026) as
    # three dots about half the time, which is invisible on its own but leaves
    # a locale inconsistent with itself — es-ES came out 136 one way and 1 the
    # other. Cosmetic, so normalise rather than reject.
    if source.rstrip().endswith("…") and out.rstrip().endswith("..."):
        out = out.rstrip()[:-3].rstrip() + "…"
    elif source.rstrip().endswith("...") and out.rstrip().endswith("…"):
        out = out.rstrip()[:-1].rstrip() + "..."

    # Match the source's leading/trailing spacing exactly — some catalog keys
    # carry deliberate spacing (" (container)") that layout depends on.
    return source[: len(source) - len(source.lstrip())] + out + source[len(source.rstrip()) :]


def _has_prose_outside(text: str, sentinels: Sentinels) -> bool:
    """Is there a word left once sentinels are removed?"""
    return any(ch.isalpha() for ch in sentinels.placeholder_re.sub("", text))


def _looks_like_a_dictionary_entry(source: str, translated: str) -> bool:
    """Did the model answer like a dictionary instead of a translator?

    Prompted as prose, a bare UI label makes TranslateGemma list every sense —
    "Skip" comes back as seven lines ending in "(Dependiendo del contexto…)" —
    and that paragraph then renders inside a button. The UI-label prompt stops
    it at the source; this stays as the net, because the failure is silent and
    ships a broken screen rather than an error.

    Deliberately shape-based, not phrase-based: a list of tell-tale connectives
    would need maintaining per language and would miss the next one.
    """
    if not translated.strip():
        return False
    # A UI string is one line. The source never has a newline (the catalog is
    # flat), so any newline is the model enumerating.
    if "\n" in translated and "\n" not in source:
        return True
    src_words = len(source.split())
    out_words = len(translated.split())
    # Short label, long answer. Generous headroom for languages that genuinely
    # need more words ("Undo" -> "Deshacer la acción"), so this only fires on a
    # blowup no honest translation reaches.
    return src_words <= 3 and out_words >= max(8, src_words * 5)


@dataclass(slots=True)
class EntryResult:
    key: str
    value: str
    #: Why this entry fell back to English, if it did.
    error: str | None = None
    #: Non-fatal notes (plural categories the target needs but English lacks).
    notes: list[str] = field(default_factory=list)
    #: Translated on the stand-in-name retry after the sentinels were lost.
    stand_ins: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def verify(source: str, translated: str, protected: Sequence[str] = ()) -> str | None:
    """Return a reason the translation is unusable, or None if it is sound.

    Checks structure, not meaning — the two failures that break rendering rather
    than merely reading badly.
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
        # Masking usually prevents this, but a model that eats the sentinel
        # takes the brand name with it — and a wrong product name is worse
        # than an untranslated string.
        return f"product term(s) lost: {dropped}"

    if _looks_like_a_dictionary_entry(source, translated):
        return "reads as a dictionary entry, not a translation"

    if has_stray_sentinel(translated):
        # The model rewrote a placeholder sentinel past recognition, so an
        # argument (or the `#` number slot) never made it back in.
        return "a masked placeholder was mangled and could not be restored"

    if not translated.strip():
        return "empty translation"
    return None


# Invented proper names, swapped in for the placeholder sentinels when a model
# drops those. translategemma:12b discards a LEADING sentinel as noise —
# "⟦PH0⟧ installed." comes back "Instalado." — which is most of what a catalog
# run used to leave in English (~13% of every locale). An unknown name is
# something it copies through and places grammatically: on strings that lost
# their sentinels, names survived 26/30 across es/ru/ja/de/tet, sentinels 7/30.
# They are only a retry: the sentinels are proven on everything else, and a
# name invites the model to treat the value as a noun ("Удалено: Zarvex.").
STAND_IN_NAMES = (
    "Zarvex", "Quilmor", "Brennat", "Toskiv", "Varneth", "Oskelyn", "Drumvar", "Pelquist",
)

# A placeholder-shaped failure — what the stand-in retry can fix
_PLACEHOLDER_FAILURES = (
    "placeholders lost",
    "plural number slots",
    "a masked placeholder was mangled",
)

# Letters of the scripts that inflect a noun by suffix. A stand-in that comes
# back with one glued on ("Zarvexа") was declined, and a value cannot be.
# CJK text sits directly against a name with no space, so it is not listed.
_INFLECTION_LETTER = re.compile(r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF]")


def to_stand_ins(masked: str) -> tuple[str, dict[str, str]] | None:
    """Swap each sentinel in ``masked`` for a stand-in name. Returns the text
    and a name -> sentinel map, or None when the scheme cannot be used: no
    sentinels, more of them than names, or a name already in the text."""
    ids = LLM_SENTINELS.placeholder_re.findall(masked)
    if not ids or max(int(i) for i in ids) >= len(STAND_IN_NAMES):
        return None
    if any(name in masked for name in STAND_IN_NAMES):
        return None
    mapping = {
        STAND_IN_NAMES[int(i)]: f"{LLM_SENTINELS.open}{i}{LLM_SENTINELS.close}" for i in set(ids)
    }
    return LLM_SENTINELS.placeholder_re.sub(lambda m: STAND_IN_NAMES[int(m.group(1))], masked), mapping


def from_stand_ins(text: str, mapping: dict[str, str], masked: str) -> str | None:
    """Put the sentinels back. None when a name was dropped, repeated,
    transliterated or declined — the value it stands for would be wrong."""
    for name, token in mapping.items():
        if text.count(name) != masked.count(token):
            return None
        for m in re.finditer(re.escape(name), text):
            after = text[m.end():m.end() + 1]
            if after and _INFLECTION_LETTER.match(after):
                # Unless the source glues one on too ("{dims}d")
                if not re.search(re.escape(token) + re.escape(after), masked):
                    return None
    for name, token in mapping.items():
        text = text.replace(name, token)
    return text


@dataclass(slots=True)
class CatalogResult:
    entries: list[EntryResult]
    elapsed_seconds: float

    @property
    def translated(self) -> dict[str, str]:
        """Only the entries that verified — a failure has no usable translation."""
        return {e.key: e.value for e in self.entries if e.ok}

    @property
    def failures(self) -> list[EntryResult]:
        return [e for e in self.entries if not e.ok]

    @property
    def flagged(self) -> list[EntryResult]:
        return [e for e in self.entries if e.notes]

    @property
    def recovered(self) -> list[EntryResult]:
        """Entries the stand-in-name retry saved from falling back to English."""
        return [e for e in self.entries if e.ok and e.stand_ins]


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
    """Translate each key, verifying structure and falling back to English.

    Sequential on purpose: the self-hosted path is one GPU, and hammering it
    with concurrent requests makes it slower, not faster (the same reasoning as
    ``BaseTranslationProvider.translate_batch``).

    Retries a *retryable* failure — a read timeout while the model loads, or a
    5xx — because losing a whole entry to a transient blip means shipping
    English for a string that was perfectly translatable.
    """
    started = time.monotonic()
    options = TranslateOptions(target_lang=target_lang, source_lang=source_lang)

    def call(source_fragment: str, masked: str, stand_ins: bool = False) -> str:
        # Brand names are masked here, not in the ICU layer: they are product
        # vocabulary, not message structure.
        guarded, terms = protect_terms(masked, protected)
        # "Chatterbox" masks down to a lone sentinel. Asked to translate that,
        # the model has nothing to work with and simply invents a word — it
        # answered "Guardar" ("Save"). There is nothing to translate here.
        if terms and not _has_prose_outside(guarded, TERM_SENTINELS):
            return source_fragment
        named = to_stand_ins(guarded) if stand_ins else None
        last: TranslationError | None = None
        for attempt in range(attempts):
            try:
                raw = provider.translate(named[0] if named else guarded, options).text
                if named:
                    restored = from_stand_ins(raw, named[1], guarded)
                    if restored is None:
                        raise TranslationError("a stand-in name did not come back intact", "catalog")
                    raw = restored
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
            f"{{{name}}} that the English source does not define"
            for name, cats in missing_plural_categories(key, target_lang).items()
        ]
        try:
            candidate = translate_icu(key, call)
            reason = verify(key, candidate, protected)
            stand_ins = False
            if reason and reason.startswith(_PLACEHOLDER_FAILURES):
                # Second chance with names in place of the sentinels it lost
                try:
                    retry = translate_icu(key, lambda src, masked: call(src, masked, stand_ins=True))
                    if verify(key, retry, protected) is None:
                        candidate, reason, stand_ins = retry, None, True
                except TranslationError:
                    pass  # keep the first failure: it says what went wrong
            result = (
                EntryResult(key=key, value=key, error=reason, notes=notes)
                if reason
                else EntryResult(key=key, value=candidate, notes=notes, stand_ins=stand_ins)
            )
        except TranslationError as exc:
            result = EntryResult(key=key, value=key, error=str(exc), notes=notes)
        entries.append(result)
        if on_progress:
            on_progress(i, result)
        # Persist as we go. A catalog run is long (thousands of sequential
        # model calls); without this, a kill or an OOM throws away everything
        # done so far, and resume has nothing to resume from.
        if on_checkpoint and checkpoint_every and (i + 1) % checkpoint_every == 0:
            on_checkpoint({e.key: e.value for e in entries if e.ok})

    if on_checkpoint:
        on_checkpoint({e.key: e.value for e in entries if e.ok})
    return CatalogResult(entries=entries, elapsed_seconds=time.monotonic() - started)


def _load(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m chat_translate.catalog",
        description="Translate a natural-key UI message catalog.",
    )
    ap.add_argument("source", type=Path, help="source catalog JSON (keys are English)")
    ap.add_argument("target_lang", help="target locale, e.g. es-ES")
    ap.add_argument(
        "--out", type=Path, help="output JSON (default: <target_lang>.json beside source)"
    )
    ap.add_argument("--source-lang", default="en")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", default="translategemma:12b")
    ap.add_argument("--limit", type=int, help="translate only the first N new keys (for a pilot)")
    ap.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="seconds to wait per request (a 12B model needs more than the chat default)",
    )
    ap.add_argument(
        "--attempts", type=int, default=3, help="tries per fragment on a retryable error"
    )
    ap.add_argument(
        "--keep-alive",
        default="60m",
        help="how long Ollama holds the model resident between requests "
        "(a reload mid-run costs minutes on a loaded box)",
    )
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=256,
        help="cap on generated tokens; a UI string never needs many",
    )
    ap.add_argument(
        "--protect",
        type=Path,
        help="JSON file of product/brand terms to keep verbatim "
        "(array, or an object with a 'protectedTerms' key)",
    )
    ap.add_argument(
        "--redo",
        action="store_true",
        help="retranslate keys already present in the output instead of resuming",
    )
    args = ap.parse_args(argv)

    source = _load(args.source)
    out_path = args.out or args.source.with_name(f"{args.target_lang}.json")
    existing = _load(out_path) if out_path.exists() else {}

    pending = [k for k in source if args.redo or k not in existing]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print(f"Nothing to do — {out_path.name} already covers all {len(source)} keys.")
        return 0

    protected: list[str] = []
    if args.protect and args.protect.exists():
        raw = json.loads(args.protect.read_text(encoding="utf-8"))
        # `maskTerms` is the subset worth masking; the rest of `protectedTerms`
        # survives on its own and masking it only invites a dropped token.
        protected = (
            raw
            if isinstance(raw, list)
            else list(raw.get("maskTerms") or raw.get("protectedTerms", []))
        )
        print(f"Protecting {len(protected)} product term(s) from translation.")

    provider = OllamaProvider(
        OllamaConfig(
            host=args.host,
            model=args.model,
            timeout_seconds=args.timeout,
            keep_alive=args.keep_alive,
            max_output_tokens=args.max_output_tokens,
            # A catalog is labels, not prose — see build_ui_label_prompt.
            prompt_style="ui_label",
        )
    )
    if not provider.health_check():
        print(
            f"Model {args.model!r} is not available at {args.host}.\n"
            f"Check `ollama list` and that the daemon is running.",
            file=sys.stderr,
        )
        return 2

    total = len(pending)
    print(f"Translating {total} key(s) to {args.target_lang} with {args.model}…")

    def progress(i: int, entry: EntryResult) -> None:
        flag = "!" if not entry.ok else " "
        print(f"  {flag} [{i + 1}/{total}] {entry.key[:60]!r} -> {entry.value[:60]!r}", flush=True)

    def write(done: dict[str, str]) -> None:
        merged = {**existing, **done}
        ordered = {k: merged[k] for k in source if k in merged}
        # Write-then-rename so a kill mid-write cannot truncate the catalog.
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(out_path)

    result = translate_catalog(
        pending,
        provider,
        args.target_lang,
        source_lang=args.source_lang,
        protected=protected,
        attempts=args.attempts,
        on_progress=progress,
        on_checkpoint=write,
    )

    # Only successful entries are written. A missing key is not a hole: the
    # catalog uses natural keys, so svelte-i18n renders the English key itself —
    # the same thing a fallback entry would have shown. Omitting them keeps the
    # file honest about what was translated AND lets the next run retry them,
    # which a written-back English value would silently prevent forever.
    merged = {**existing, **result.translated}
    ordered = {k: merged[k] for k in source if k in merged}

    rate = total / result.elapsed_seconds if result.elapsed_seconds else 0.0
    print(
        f"\nWrote {len(ordered)}/{len(source)} keys to {out_path} "
        f"({result.elapsed_seconds:.0f}s, {rate:.2f} keys/s)."
    )
    if result.failures:
        print(
            f"\n{len(result.failures)} left untranslated (they render as English) "
            f"— rerun to retry them:"
        )
        for e in result.failures[:20]:
            print(f"  {e.key[:70]!r}: {e.error}")
    if result.recovered:
        print(
            f"\n{len(result.recovered)} kept their placeholders only on the stand-in-name "
            f"retry — worth a look in review."
        )
    if result.flagged:
        print(f"\n{len(result.flagged)} need a plural review for {args.target_lang}:")
        for e in result.flagged[:20]:
            print(f"  {e.key[:70]!r}: {e.notes[0]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
