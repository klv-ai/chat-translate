r"""Do-not-translate masking.

@mentions, emails, URLs, #channels, emoji, and code are replaced with opaque
Private-Use-Area sentinels, translated around, and restored verbatim. Naive
translation mangles all of these.

Uses the third-party ``regex`` module (not stdlib ``re``) because the emoji
rules need Unicode properties like ``\p{Extended_Pictographic}``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import regex as re

# Private Use Area sentinels — they virtually never appear in real chat text.
# Spelled with chr() so the source stays pure ASCII (no invisible characters).
SENTINEL_OPEN = chr(0xE000)
SENTINEL_CLOSE = chr(0xE001)

_STRIP_SENTINELS = re.compile(f"[{SENTINEL_OPEN}{SENTINEL_CLOSE}]")
#: Matches a masked token: <PUA-open><id><PUA-close>. Exposed for the chat /
#: fan-out layers to strip or detect placeholders.
PLACEHOLDER_RE = re.compile(SENTINEL_OPEN + r"(\d+)" + SENTINEL_CLOSE)

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
    _tokens: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    def restore(self, translated: str) -> str:
        """Re-insert the originals into a translated string."""

        def _sub(m: re.Match[str]) -> str:
            i = int(m.group(1))
            return self._tokens[i] if 0 <= i < len(self._tokens) else ""

        restored: str = PLACEHOLDER_RE.sub(_sub, translated)
        return restored

    def find_unrestored(self, translated: str) -> list[int]:
        """Token ids that did NOT survive translation (should be empty)."""
        seen = {int(m.group(1)) for m in PLACEHOLDER_RE.finditer(translated)}
        return [i for i in range(len(self._tokens)) if i not in seen]


def mask_non_translatable(raw: str, rules: Sequence[MaskRule] = DEFAULT_RULES) -> MaskedMessage:
    """Pure, provider-independent, and cheap. Call it once per message — the
    fan-out layer reuses the single masked form across every target language and
    only re-runs ``restore`` per viewer."""
    # Defensive: drop any pre-existing sentinels so user input can't collide.
    work = _STRIP_SENTINELS.sub("", raw)
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
            return f"{SENTINEL_OPEN}{idx}{SENTINEL_CLOSE}{tail}"

        work = rule.pattern.sub(_replace, work)

    return MaskedMessage(masked=work, _tokens=tokens)
