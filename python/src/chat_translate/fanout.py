"""Fan-out + cache layer.

A chat message is authored once but viewed by many people, each in their own
language. This turns "one message" into "one translation per distinct language
currently being viewed" — masking once, translating the masked form into each
target (bounded concurrency), and caching on the masked text.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from .chat import LanguageDetector, ViewerTranslateOptions, ViewerTranslation, _same_lang
from .masking import DEFAULT_RULES, PLACEHOLDER_RE, MaskRule, mask_non_translatable
from .provider import (
    Formality,
    LanguageCode,
    ProviderCapabilities,
    TranslateOptions,
    TranslationError,
    TranslationProvider,
)


@dataclass(slots=True)
class CachedTranslation:
    #: Provider output BEFORE restore — still carrying sentinels (restore is
    #: per-message, so it must not be baked into the cached value).
    masked_translation: str
    detected_source_lang: LanguageCode


class TranslationCache(Protocol):
    def get(self, key: str) -> CachedTranslation | None: ...

    def set(self, key: str, value: CachedTranslation) -> None: ...


class InMemoryTranslationCache:
    """Thread-safe LRU+TTL cache, fine for a single process. Swap in a shared
    store (Redis) for multi-instance; the key is already a flat string."""

    def __init__(
        self,
        *,
        max_entries: int = 10_000,
        ttl_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store: OrderedDict[str, tuple[CachedTranslation, float]] = OrderedDict()
        self._max = max_entries
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()

    def get(self, key: str) -> CachedTranslation | None:
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            value, expires = hit
            if expires != float("inf") and self._clock() > expires:
                del self._store[key]
                return None
            self._store.move_to_end(key)  # LRU touch
            return value

    def set(self, key: str, value: CachedTranslation) -> None:
        expires = self._clock() + self._ttl if self._ttl else float("inf")
        with self._lock:
            self._store[key] = (value, expires)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)  # evict oldest


def _cache_key(
    masked: str,
    source: LanguageCode | None,
    target: LanguageCode,
    formality: Formality | None,
    glossary_id: str | None,
) -> str:
    # NUL separates fields so no value can bleed into another.
    return "\x00".join(
        ["v1", source or "auto", target, formality or "-", glossary_id or "-", masked]
    )


@dataclass(slots=True)
class RoomStats:
    targets: int
    cache_hits: int = 0
    translated: int = 0
    short_circuited: int = 0


@dataclass(slots=True)
class RoomTranslation:
    #: target language -> result for any viewer reading in that language.
    by_language: dict[LanguageCode, ViewerTranslation]
    detected_source_lang: LanguageCode
    stats: RoomStats


class RoomTranslator:
    def __init__(
        self,
        provider: TranslationProvider,
        *,
        detector: LanguageDetector | None = None,
        default_formality: Formality = "prefer_less",
        rules: Sequence[MaskRule] = DEFAULT_RULES,
        cache: TranslationCache | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self._provider = provider
        self._caps = provider.capabilities()
        self._detector = detector
        self._default_formality = default_formality
        self._rules = rules
        self._cache = cache
        # Parallel for native-batch providers (DeepL), serial for one GPU.
        self._max_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else (8 if self._caps.native_batch else 1)
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    def _resolve_source(
        self, detect_input: str, explicit: LanguageCode | None = None
    ) -> LanguageCode | None:
        if explicit:
            return explicit
        if self._detector is not None:
            return self._detector.detect(detect_input)
        if self._caps.auto_detect_source:
            return None
        raise TranslationError(
            f"{self._provider.name} cannot auto-detect and no detector is configured",
            self._provider.name,
        )

    def translate_for_room(
        self,
        raw: str,
        viewed_languages: Iterable[LanguageCode],
        opts: ViewerTranslateOptions | None = None,
    ) -> RoomTranslation:
        opts = opts or ViewerTranslateOptions()
        m = mask_non_translatable(raw, self._rules)
        targets = list(dict.fromkeys(viewed_languages))  # distinct, order-preserving
        by_language: dict[LanguageCode, ViewerTranslation] = {}
        stats = RoomStats(targets=len(targets))

        # Short-circuit A: nothing translatable (pure emoji / mention / code / url).
        if PLACEHOLDER_RE.sub("", m.masked).strip() == "":
            for t in targets:
                by_language[t] = ViewerTranslation(
                    text=raw, detected_source_lang=opts.source_lang or t, translated=False
                )
            stats.short_circuited = len(targets)
            return RoomTranslation(
                by_language=by_language,
                detected_source_lang=opts.source_lang or (targets[0] if targets else ""),
                stats=stats,
            )

        detect_input = PLACEHOLDER_RE.sub(" ", m.masked)
        source_lang = self._resolve_source(detect_input, opts.source_lang)
        detected_source_lang = source_lang or ""
        formality: Formality | None = (
            (opts.formality or self._default_formality) if self._caps.formality else None
        )

        # Short-circuit B: viewers already on the source language get the original.
        need: list[LanguageCode] = []
        for t in targets:
            if source_lang and _same_lang(source_lang, t):
                by_language[t] = ViewerTranslation(
                    text=raw, detected_source_lang=source_lang, translated=False
                )
                stats.short_circuited += 1
            else:
                need.append(t)

        def _do(target: LanguageCode) -> tuple[LanguageCode, str, LanguageCode, bool]:
            key = _cache_key(m.masked, source_lang, target, formality, opts.glossary_id)
            cached = self._cache.get(key) if self._cache else None
            if cached is not None:
                return target, cached.masked_translation, cached.detected_source_lang, True
            res = self._provider.translate(
                m.masked,
                TranslateOptions(
                    target_lang=target,
                    source_lang=source_lang,
                    formality=formality,
                    context=opts.context,
                    glossary_id=opts.glossary_id,
                ),
            )
            if self._cache:
                self._cache.set(
                    key,
                    CachedTranslation(
                        masked_translation=res.text, detected_source_lang=res.detected_source_lang
                    ),
                )
            return target, res.text, res.detected_source_lang, False

        results: list[tuple[LanguageCode, str, LanguageCode, bool]] = []
        if need:
            workers = max(1, min(self._max_concurrency, len(need)))
            if workers == 1:
                results = [_do(t) for t in need]
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    results = list(ex.map(_do, need))

        # Mutate shared state sequentially (no races with the parallel _do calls).
        for target, masked_translation, det, hit in results:
            if hit:
                stats.cache_hits += 1
            else:
                stats.translated += 1
            if det:
                detected_source_lang = det
            by_language[target] = ViewerTranslation(
                text=m.restore(masked_translation),
                detected_source_lang=det or source_lang or target,
                translated=True,
                unrestored_tokens=m.find_unrestored(masked_translation),
            )

        return RoomTranslation(
            by_language=by_language, detected_source_lang=detected_source_lang, stats=stats
        )


def create_room_translator(
    provider: TranslationProvider,
    *,
    detector: LanguageDetector | None = None,
    default_formality: Formality = "prefer_less",
    rules: Sequence[MaskRule] = DEFAULT_RULES,
    cache: TranslationCache | None = None,
    max_concurrency: int | None = None,
) -> RoomTranslator:
    return RoomTranslator(
        provider,
        detector=detector,
        default_formality=default_formality,
        rules=rules,
        cache=cache,
        max_concurrency=max_concurrency,
    )
