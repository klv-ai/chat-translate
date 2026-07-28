"""ICU MessageFormat support for translating UI catalogs.

The chat path masks @mentions, URLs, and emoji. A UI catalog brings a different
class of do-not-translate span: **ICU arguments**.

    "Run {id} · idle {minutes}m"
    "Imported {count, plural, one {# file} other {# files}}."

These break in two distinct ways, so they need two distinct treatments:

* A **simple argument** (``{minutes}``) is a lookup key into the values the
  caller passes at render time. Its name is code, not copy. Handed to a
  translator verbatim it comes back localized — TranslateGemma turns
  ``{minutes}`` into ``{minutos}`` — and the renderer then finds no matching
  value and prints the placeholder. So: mask it whole.

* A **plural / select argument** is both. The skeleton (``count, plural,
  one {…} other {…}``) is code, but each branch holds real copy that MUST be
  translated, and translated as its own short message. So: rebuild the skeleton
  programmatically and recurse into the branches.

Everything here is pure and provider-agnostic — it produces masked text for any
:class:`TranslationProvider` and reassembles the result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .masking import LLM_SENTINELS, MaskedMessage, Sentinels

#: Argument types whose body is a set of named sub-messages rather than a value.
SUBMESSAGE_TYPES = frozenset({"plural", "selectordinal", "select"})


@dataclass(frozen=True, slots=True)
class Argument:
    """One balanced ``{...}`` span in a message."""

    start: int
    end: int
    raw: str
    name: str
    #: "plural" / "select" / "selectordinal", or None for a simple argument.
    arg_type: str | None

    @property
    def has_submessages(self) -> bool:
        return self.arg_type in SUBMESSAGE_TYPES


def find_arguments(text: str) -> list[Argument]:
    """Top-level ``{...}`` spans, brace-balanced.

    A regex cannot do this: plural bodies nest braces (``{count, plural, one
    {# file}}``), so the scan has to count depth.
    """
    out: list[Argument] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue  # stray close brace — leave it to the caller
            depth -= 1
            if depth == 0 and start >= 0:
                raw = text[start : i + 1]
                name, arg_type = _split_head(raw)
                out.append(Argument(start=start, end=i + 1, raw=raw, name=name, arg_type=arg_type))
                start = -1
    return out


def _split_head(raw: str) -> tuple[str, str | None]:
    """``{count, plural, …}`` -> ("count", "plural"); ``{minutes}`` -> ("minutes", None)."""
    inner = raw[1:-1]
    head, sep, rest = inner.partition(",")
    if not sep:
        return inner.strip(), None
    arg_type = rest.partition(",")[0].strip()
    return head.strip(), arg_type or None


def parse_branches(raw: str) -> list[tuple[str, str]]:
    """Branches of a plural/select argument as ``[(selector, submessage), …]``.

    Order is preserved so the rebuilt message keeps the author's branch order.
    """
    inner = raw[1:-1]
    # Skip "name, type," — the remainder is the branch list.
    _, _, rest = inner.partition(",")
    _, _, body = rest.partition(",")

    branches: list[tuple[str, str]] = []
    i = 0
    n = len(body)
    while i < n:
        while i < n and body[i].isspace():
            i += 1
        sel_start = i
        while i < n and not body[i].isspace() and body[i] != "{":
            i += 1
        selector = body[sel_start:i].strip()
        while i < n and body[i].isspace():
            i += 1
        if i >= n or body[i] != "{":
            break
        depth = 0
        msg_start = i + 1
        while i < n:
            if body[i] == "{":
                depth += 1
            elif body[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if selector:
            branches.append((selector, body[msg_start:i]))
        i += 1
    return branches


@dataclass(slots=True)
class MaskedIcu:
    """A message split into a translatable carrier plus its ICU structure.

    ``carrier`` is what goes to the provider: prose with every argument replaced
    by a sentinel. ``branches`` are the plural/select sub-messages, each itself a
    :class:`MaskedIcu`, translated separately and slotted back in.
    """

    carrier: MaskedMessage
    #: (token index, argument) for every masked argument, in token order.
    arguments: list[tuple[int, Argument]] = field(default_factory=list)
    #: token index -> [(selector, nested)] for plural/select arguments.
    branches: dict[int, list[tuple[str, MaskedIcu]]] = field(default_factory=dict)

    @property
    def carrier_has_prose(self) -> bool:
        """Is there a word left once the arguments are masked out?

        Stricter than :meth:`MaskedMessage.has_translatable`, which counts any
        leftover non-space. That is right for chat, where "?!" carries meaning,
        but a UI carrier like ``{done}/{total}`` is pure layout — sending it to a
        model can only invite damage.
        """
        stripped = self.carrier.sentinels.placeholder_re.sub("", self.carrier.masked)
        return any(ch.isalpha() for ch in stripped)


#: The number placeholder inside a plural branch ("# files"). Masked like an
#: argument so the model cannot drop or relocate it.
HASH_TOKEN = "#"


def mask_icu(text: str, sentinels: Sentinels = LLM_SENTINELS) -> MaskedIcu:
    """Replace every ICU argument with a sentinel, recursing into plural branches.

    Defaults to the bracket scheme: this path always targets an instruction-tuned
    LLM, whose tokenizer eats the Private-Use-Area default.
    """
    work = sentinels.strip_re.sub("", text)
    args = find_arguments(work)

    tokens: list[str] = []
    arguments: list[tuple[int, Argument]] = []
    branches: dict[int, list[tuple[str, MaskedIcu]]] = {}

    pieces: list[str] = []
    cursor = 0
    for arg in args:
        pieces.append(work[cursor : arg.start])
        idx = len(tokens)
        tokens.append(arg.raw)
        arguments.append((idx, arg))
        if arg.has_submessages:
            branches[idx] = [
                (sel, mask_icu(msg, sentinels)) for sel, msg in parse_branches(arg.raw)
            ]
        pieces.append(f"{sentinels.open}{idx}{sentinels.close}")
        cursor = arg.end
    pieces.append(work[cursor:])

    # `#` is the plural number slot; protect it exactly like an argument.
    carrier_text = "".join(pieces)
    if HASH_TOKEN in carrier_text:
        idx = len(tokens)
        tokens.append(HASH_TOKEN)
        carrier_text = carrier_text.replace(HASH_TOKEN, f"{sentinels.open}{idx}{sentinels.close}")

    return MaskedIcu(
        carrier=MaskedMessage(masked=carrier_text, sentinels=sentinels, _tokens=tokens),
        arguments=arguments,
        branches=branches,
    )


#: Callback that turns one fragment into its translation.
#:
#: Takes ``(source_fragment, masked_fragment)``. The source is passed too because
#: a caller usually needs it to clean the result — punctuation drift is judged
#: against the fragment the model actually saw, and for a plural branch that is
#: the branch, not the whole message.
TranslateFn = Callable[[str, str], str]


#: How an instruction-tuned model mangles ``⟦PH0⟧`` when it does not pass it
#: through verbatim. Observed with translategemma:12b, which rewrites the
#: mathematical brackets as ASCII ones — most often inside plural branches, and
#: occasionally drops them entirely leaving the bare ``PH0``.
_SENTINEL_LOOKALIKES = (
    "[PH{i}]",
    "［PH{i}］",
    "【PH{i}】",
    "〈PH{i}〉",
    "⟨PH{i}⟩",
    "(PH{i})",
    "[ PH{i} ]",
    "PH{i}",
)


def recover_sentinels(text: str, masked: MaskedMessage) -> str:
    """Put back sentinels the model rewrote into a look-alike form.

    Only rewrites an index whose real sentinel is ABSENT from the output, so a
    literal "[0]" in translated copy is never clobbered.
    """
    out = text
    for i in range(masked.token_count):
        proper = f"{masked.sentinels.open}{i}{masked.sentinels.close}"
        if proper in out:
            continue
        for shape in _SENTINEL_LOOKALIKES:
            candidate = shape.format(i=i)
            if candidate in out:
                out = out.replace(candidate, proper)
                break
    return out


def translate_icu(text: str, translate: TranslateFn, sentinels: Sentinels = LLM_SENTINELS) -> str:
    """Translate an ICU message, preserving its structure exactly.

    The carrier sentence goes to the provider with its arguments masked, so it
    keeps full sentence context. Each plural/select branch is translated as its
    own short message and the skeleton is rebuilt in code — never by the model —
    so selectors and argument names cannot drift.

    A carrier with no prose (``"{count} {label}"``) is skipped entirely: there is
    nothing to translate and a round trip could only introduce damage.
    """
    masked = mask_icu(text, sentinels)

    translated_carrier = (
        recover_sentinels(translate(text, masked.carrier.masked), masked.carrier)
        if masked.carrier_has_prose
        else masked.carrier.masked
    )

    # Rebuild each argument. The skeleton is written here, in code — the model
    # only ever sees branch prose, so a selector or argument name cannot drift.
    rebuilt: dict[int, str] = {}
    for idx, arg in masked.arguments:
        if idx not in masked.branches:
            rebuilt[idx] = arg.raw
            continue
        parts: list[str] = []
        for selector, branch_source in parse_branches(arg.raw):
            translated_branch = translate_icu(branch_source, translate, sentinels)
            translated_branch = _restore_hash(branch_source, translated_branch)
            parts.append(f"{selector} {{{translated_branch}}}")
        rebuilt[idx] = f"{{{arg.name}, {arg.arg_type}, {' '.join(parts)}}}"

    out = translated_carrier
    for idx, _ in masked.arguments:
        out = out.replace(f"{sentinels.open}{idx}{sentinels.close}", rebuilt[idx])
    # Restore whatever is left — in practice the `#` number slot.
    return masked.carrier.restore(out)


def has_stray_sentinel(text: str, sentinels: Sentinels = LLM_SENTINELS) -> bool:
    """Did a sentinel survive into finished output?

    It should never happen — every token is restored — so if one is left, the
    model mangled it beyond :func:`recover_sentinels` and the string is unusable.
    """
    return bool(sentinels.strip_re.search(text))


def _restore_hash(branch_source: str, translated: str) -> str:
    """Put back a `#` number slot the model dropped from a plural branch.

    Branches are tiny — "# file", "# attachments" — and on the short ones the
    model returns just the noun: "⟦PH0⟧ attachments" came back "Anhänge". The
    slot is not decoration; without it the count never renders.

    Reinserted on the side it occupied in English. That is right for every
    locale here (all of them put the number before the noun) and is why this
    repairs rather than guesses: if the source did not lead with `#`, neither
    does the repair.
    """
    want = branch_source.count(HASH_TOKEN)
    if want == 0:
        return translated
    stripped = translated.strip()
    if not stripped:
        return translated
    leading = branch_source.lstrip().startswith(HASH_TOKEN)

    # Present but on the wrong side. "# sources" came back "Sources #" — the
    # model reads `#` as decoration and re-places it. A count reads before its
    # noun in every locale here, and mirroring the source is right regardless.
    if translated.count(HASH_TOKEN) >= want:
        if leading and not stripped.startswith(HASH_TOKEN) and stripped.endswith(HASH_TOKEN):
            return f"{HASH_TOKEN} {stripped[:-1].strip()}"
        return translated

    return f"{HASH_TOKEN} {stripped}" if leading else f"{stripped} {HASH_TOKEN}"


def arguments_of(text: str) -> set[str]:
    """Every argument name in a message, including inside plural branches.

    Used to verify a translation still resolves: if a name went missing, the
    renderer would print the placeholder instead of the value.
    """
    names: set[str] = set()
    for arg in find_arguments(text):
        names.add(arg.name)
        if arg.has_submessages:
            for _, msg in parse_branches(arg.raw):
                names |= arguments_of(msg)
    return names


def hash_slots_of(text: str) -> dict[str, list[int]]:
    """How many ``#`` number slots each plural branch holds, per argument.

    ``#`` is not an argument, so :func:`arguments_of` cannot see it — yet losing
    one is just as bad. translategemma:4b rewrites ``one {# file}`` as
    ``one {Archivo 0}``, which renders a literal 0 where the count belongs.
    """
    out: dict[str, list[int]] = {}
    for arg in find_arguments(text):
        if not arg.has_submessages:
            continue
        out[arg.name] = [msg.count(HASH_TOKEN) for _, msg in parse_branches(arg.raw)]
        for _, msg in parse_branches(arg.raw):
            out.update(hash_slots_of(msg))
    return out


def selectors_of(text: str) -> dict[str, list[str]]:
    """Plural/select selectors per argument name, e.g. ``{"count": ["one", "other"]}``."""
    out: dict[str, list[str]] = {}
    for arg in find_arguments(text):
        if not arg.has_submessages:
            continue
        out[arg.name] = [sel for sel, _ in parse_branches(arg.raw)]
        for _, msg in parse_branches(arg.raw):
            out.update(selectors_of(msg))
    return out


# CLDR plural categories per language, for the languages this product ships.
# English source messages only ever carry `one`/`other`; a target language with
# more categories cannot be filled in by translating those two, so the catalog
# tool reports them for a human rather than silently under-specifying.
PLURAL_CATEGORIES: dict[str, frozenset[str]] = {
    "en": frozenset({"one", "other"}),
    "de": frozenset({"one", "other"}),
    "nl": frozenset({"one", "other"}),
    "es": frozenset({"one", "many", "other"}),
    "it": frozenset({"one", "many", "other"}),
    "pt": frozenset({"one", "many", "other"}),
    "fr": frozenset({"one", "many", "other"}),
    "ja": frozenset({"other"}),
    "ko": frozenset({"other"}),
    "zh": frozenset({"other"}),
    "th": frozenset({"other"}),
    "vi": frozenset({"other"}),
    "id": frozenset({"other"}),
    "tr": frozenset({"one", "other"}),
    "ru": frozenset({"one", "few", "many", "other"}),
    "pl": frozenset({"one", "few", "many", "other"}),
    "cs": frozenset({"one", "few", "many", "other"}),
    "uk": frozenset({"one", "few", "many", "other"}),
    "ar": frozenset({"zero", "one", "two", "few", "many", "other"}),
    "hi": frozenset({"one", "other"}),
    "tet": frozenset({"other"}),
    "fil": frozenset({"one", "other"}),
}


def missing_plural_categories(text: str, target_lang: str) -> dict[str, Sequence[str]]:
    """Categories the target language needs that the source message does not define.

    Empty for most European targets (``one``/``other`` covers them). Non-empty
    for Slavic and Arabic, where a correct plural needs branches English never
    had — a translation cannot invent them, so they are surfaced as review work.
    """
    lang = target_lang.split("-")[0].lower()
    needed = PLURAL_CATEGORIES.get(lang)
    if needed is None:
        return {}
    out: dict[str, Sequence[str]] = {}
    for name, selectors in selectors_of(text).items():
        # `=0` / `=1` exact matches are not categories; strip them before comparing.
        present = {s for s in selectors if not s.startswith("=")}
        if "other" not in present:
            continue  # a `select`, not a plural — categories do not apply
        gap = sorted(needed - present)
        if gap:
            out[name] = gap
    return out
