r"""Do-not-translate masking.

@mentions, emails, URLs, #channels, emoji, and code are replaced with opaque
sentinels, translated around, and restored verbatim. Naive translation mangles
all of these.

The sentinel *delimiters* are a per-provider concern (see ``Sentinels``): the
default Private-Use-Area scheme survives NMT engines (DeepL) almost perfectly,
but instruction-tuned LLM tokenizers (TranslateGemma) drop PUA code points, so
those providers override to a visible bracket scheme. The choice rides on the
``MaskedMessage`` so ``restore`` always matches how the text was masked.

Uses the third-party ``regex`` module (not stdlib ``re``) because the emoji
rules need Unicode properties like ``\p{Extended_Pictographic}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import regex as re


@dataclass(frozen=True, slots=True)
class Sentinels:
    """A placeholder delimiter scheme: ``<open><id><close>`` plus the compiled
    regexes derived from it. Build via :func:`make_sentinels`."""

    open: str
    close: str
    placeholder_re: re.Pattern[str]
    strip_re: re.Pattern[str]


def make_sentinels(open_: str, close_: str) -> Sentinels:
    # `strip_re` matches either delimiter string, so multi-character delimiters work.
    return Sentinels(
        open=open_,
        close=close_,
        placeholder_re=re.compile(re.escape(open_) + r"(\d+)" + re.escape(close_)),
        strip_re=re.compile(re.escape(open_) + "|" + re.escape(close_)),
    )


# Default: Private-Use-Area code points. Invisible, virtually never typed, and
# preserved ~100% by NMT engines (DeepL). Spelled with chr() to keep the source
# pure ASCII. This is the broad-base default; providers may override.
DEFAULT_SENTINELS = make_sentinels(chr(0xE000), chr(0xE001))

# Override for instruction-tuned LLM backends: mathematical white square
# brackets (U+27E6 ⟦ / U+27E7 ⟧). Still vanishingly rare in chat, but — unlike
# PUA — they survive TranslateGemma's tokenizer instead of being silently eaten.
BRACKET_SENTINELS = make_sentinels(chr(0x27E6), chr(0x27E7))

# Bracket sentinels with a "PH" prefix (⟦PH0⟧), used by the ICU catalog path.
# The prefix keeps an LLM from reading the index as a number and translating it.
LLM_SENTINELS = make_sentinels(chr(0x27E6) + "PH", chr(0x27E7))

# Back-compat module-level aliases (the default scheme).
SENTINEL_OPEN = DEFAULT_SENTINELS.open
SENTINEL_CLOSE = DEFAULT_SENTINELS.close
#: Matches a masked token in the DEFAULT scheme. Per-message logic should prefer
#: ``MaskedMessage`` helpers, which use that message's own scheme.
PLACEHOLDER_RE = DEFAULT_SENTINELS.placeholder_re

# Emoji pattern: keep the \p{...} property tokens as regex syntax, but spell the
# code-point ranges with real characters via chr() so we don't depend on the
# regex flavour's \x{...} escape handling.
_PICTO = r"\p{Extended_Pictographic}"
_RI = r"\p{Regional_Indicator}"
_SKIN = f"{chr(0x1F3FB)}-{chr(0x1F3FF)}{chr(0xFE0F)}"  # skin-tone modifiers + VS-16
_ZWJ = chr(0x200D)
_EMOJI = f"{_PICTO}[{_SKIN}]*(?:{_ZWJ}{_PICTO}[{_SKIN}]*)*|{_RI}{{2}}"

_TRAILING_PUNCT = re.compile(r"[.,!?;:)\]]+$")


@dataclass(frozen=True, slots=True)
class MaskRule:
    name: str
    pattern: re.Pattern[str]
    #: Trim trailing sentence punctuation out of the masked span (URLs).
    trim_trailing_punct: bool = False


# Order matters. Structural spans (code) are masked first so later rules cannot
# reach inside them; each masked span becomes a token no later rule can match.
DEFAULT_RULES: tuple[MaskRule, ...] = (
    MaskRule("fenced_code", re.compile(r"```[\s\S]*?```")),
    MaskRule("inline_code", re.compile(r"`[^`\n]+`")),
    MaskRule("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    MaskRule(
        "url",
        re.compile(r"\b(?:https?://|www\.)[^\s<]+", re.IGNORECASE),
        trim_trailing_punct=True,
    ),
    # @mention / #channel: only when preceded by whitespace or start-of-string,
    # so we don't eat the "@" inside an email or a mid-word "#".
    MaskRule("mention", re.compile(r"(?<![^\s])@[\w.\-]+")),
    MaskRule("channel", re.compile(r"(?<![^\s])#[\w\-]+")),
    MaskRule("shortcode_emoji", re.compile(r":[a-z0-9_+\-]+:", re.IGNORECASE)),
    MaskRule("unicode_emoji", re.compile(_EMOJI)),
)


@dataclass(slots=True)
class MaskedMessage:
    #: Text with non-translatable spans replaced by sentinels.
    masked: str
    #: The delimiter scheme used — restore/find_unrestored key off this so they
    #: always match how this message was masked.
    sentinels: Sentinels = DEFAULT_SENTINELS
    _tokens: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def restore(self, translated: str) -> str:
        """Re-insert the originals into a translated string."""

        def _sub(m: re.Match[str]) -> str:
            i = int(m.group(1))
            return self._tokens[i] if 0 <= i < len(self._tokens) else ""

        restored: str = self.sentinels.placeholder_re.sub(_sub, translated)
        return restored

    def find_unrestored(self, translated: str) -> list[int]:
        """Token ids that did NOT survive translation (should be empty)."""
        seen = {int(m.group(1)) for m in self.sentinels.placeholder_re.finditer(translated)}
        return [i for i in range(len(self._tokens)) if i not in seen]

    def has_translatable(self) -> bool:
        """False when the message is pure placeholders (emoji / mention / url /
        code) with nothing left to translate."""
        stripped: str = self.sentinels.placeholder_re.sub("", self.masked)
        return stripped.strip() != ""

    def detection_text(self) -> str:
        """Masked text with placeholders blanked to spaces — the input to
        language detection (sentinels would otherwise skew short messages)."""
        blanked: str = self.sentinels.placeholder_re.sub(" ", self.masked)
        return blanked


def mask_non_translatable(
    raw: str,
    rules: Sequence[MaskRule] = DEFAULT_RULES,
    sentinels: Sentinels = DEFAULT_SENTINELS,
) -> MaskedMessage:
    """Pure, cheap, and parametrised by the delimiter scheme. Call it once per
    message — the fan-out layer reuses the single masked form across every target
    language and only re-runs ``restore`` per viewer."""
    # Defensive: drop any pre-existing sentinels so user input can't collide.
    work = sentinels.strip_re.sub("", raw)
    tokens: list[str] = []

    for rule in rules:

        def _replace(m: re.Match[str], rule: MaskRule = rule) -> str:
            value = m.group(0)
            tail = ""
            if rule.trim_trailing_punct:
                t = _TRAILING_PUNCT.search(value)
                if t:
                    tail = t.group(0)
                    value = value[: -len(tail)]
            idx = len(tokens)
            tokens.append(value)
            return f"{sentinels.open}{idx}{sentinels.close}{tail}"

        work = rule.pattern.sub(_replace, work)

    return MaskedMessage(masked=work, sentinels=sentinels, _tokens=tokens)
