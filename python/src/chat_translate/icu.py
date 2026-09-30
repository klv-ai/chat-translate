"""ICU MessageFormat support for translating UI catalogs.

* A simple argument (``{minutes}``) is masked whole and restored verbatim.
* A plural/select argument (``{count, plural, one {# file} other {# files}}``)
  has its skeleton rebuilt in code; only the branch text is translated, each
  branch as its own fragment. The ``#`` number slot is masked like an argument.

Pure and provider-agnostic: callers supply a :data:`TranslateFn`.
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
    """Top-level ``{...}`` spans, brace-balanced (plural bodies nest braces).
    Unmatched closing braces are ignored."""
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
        """True if any letter remains once the arguments are masked out.

        Stricter than :meth:`MaskedMessage.has_translatable`, which counts any
        non-space character; ``{done}/{total}`` has no prose.
        """
        stripped = self.carrier.sentinels.placeholder_re.sub("", self.carrier.masked)
        return any(ch.isalpha() for ch in stripped)


#: The number placeholder inside a plural branch ("# files").
HASH_TOKEN = "#"


def mask_icu(text: str, sentinels: Sentinels = LLM_SENTINELS) -> MaskedIcu:
    """Replace every ICU argument and ``#`` slot with a sentinel, recursing into
    plural/select branches. Existing sentinel delimiters in *text* are removed
    first."""
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
#: Takes ``(source_fragment, masked_fragment)`` and returns the translated masked
#: fragment. For a plural branch, ``source_fragment`` is the branch text only.
TranslateFn = Callable[[str, str], str]


#: Alternate renderings of a sentinel that :func:`recover_sentinels` maps back.
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
    """Replace look-alike forms of a sentinel with the real sentinel.

    Only indices whose real sentinel is absent from *text* are rewritten.
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
    """Translate an ICU message, preserving argument names, selectors and skeleton.

    The carrier (the message with every argument masked) is passed to
    *translate* once; each plural/select branch is translated recursively as its
    own fragment, and the skeleton is rebuilt from the source. A carrier with no
    prose is not passed to *translate*.
    """
    masked = mask_icu(text, sentinels)

    translated_carrier = (
        recover_sentinels(translate(text, masked.carrier.masked), masked.carrier)
        if masked.carrier_has_prose
        else masked.carrier.masked
    )

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
    """True if a sentinel delimiter remains in *text*, i.e. a placeholder could
    not be restored."""
    return bool(sentinels.strip_re.search(text))


def _restore_hash(branch_source: str, translated: str) -> str:
    """Ensure a translated plural branch has its ``#`` slot on the source's side.

    If the slot is missing it is added at the start when the source branch
    starts with ``#``, otherwise at the end. If present only at the end while
    the source leads with it, it is moved to the start. Assumes the target
    language places the number on the same side as the source.
    """
    want = branch_source.count(HASH_TOKEN)
    if want == 0:
        return translated
    stripped = translated.strip()
    if not stripped:
        return translated
    leading = branch_source.lstrip().startswith(HASH_TOKEN)

    if translated.count(HASH_TOKEN) >= want:
        if leading and not stripped.startswith(HASH_TOKEN) and stripped.endswith(HASH_TOKEN):
            return f"{HASH_TOKEN} {stripped[:-1].strip()}"
        return translated

    return f"{HASH_TOKEN} {stripped}" if leading else f"{stripped} {HASH_TOKEN}"


def arguments_of(text: str) -> set[str]:
    """Every argument name in a message, including inside plural branches."""
    names: set[str] = set()
    for arg in find_arguments(text):
        names.add(arg.name)
        if arg.has_submessages:
            for _, msg in parse_branches(arg.raw):
                names |= arguments_of(msg)
    return names


def hash_slots_of(text: str) -> dict[str, list[int]]:
    """How many ``#`` number slots each plural branch holds, per argument."""
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


# CLDR cardinal plural categories for a subset of languages. Languages not
# listed here are skipped by :func:`missing_plural_categories`.
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
    """Plural categories *target_lang* needs that each plural argument in *text*
    does not define, keyed by argument name. Exact-match selectors (``=0``) are
    ignored, as are arguments without an ``other`` branch (``select``). Empty
    for languages not in :data:`PLURAL_CATEGORIES`."""
    lang = target_lang.split("-")[0].lower()
    needed = PLURAL_CATEGORIES.get(lang)
    if needed is None:
        return {}
    out: dict[str, Sequence[str]] = {}
    for name, selectors in selectors_of(text).items():
        present = {s for s in selectors if not s.startswith("=")}
        if "other" not in present:
            continue
        gap = sorted(needed - present)
        if gap:
            out[name] = gap
    return out
