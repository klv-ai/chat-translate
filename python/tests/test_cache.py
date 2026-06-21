from __future__ import annotations

from chat_translate import CachedTranslation, InMemoryTranslationCache


def _entry(s: str) -> CachedTranslation:
    return CachedTranslation(masked_translation=s, detected_source_lang="en")


def test_evicts_least_recently_used() -> None:
    cache = InMemoryTranslationCache(max_entries=2)
    cache.set("a", _entry("A"))
    cache.set("b", _entry("B"))

    got_a = cache.get("a")  # touch a -> b is now LRU
    assert got_a is not None and got_a.masked_translation == "A"

    cache.set("c", _entry("C"))  # size 3 > 2 -> evict b
    assert cache.get("b") is None
    assert cache.get("a") is not None
    assert cache.get("c") is not None


def test_expires_past_ttl() -> None:
    now: list[float] = [1000.0]
    cache = InMemoryTranslationCache(ttl_seconds=10.0, clock=lambda: now[0])
    cache.set("k", _entry("V"))

    got = cache.get("k")
    assert got is not None and got.masked_translation == "V"

    now[0] += 15.0
    assert cache.get("k") is None


def test_never_expires_without_ttl() -> None:
    now: list[float] = [0.0]
    cache = InMemoryTranslationCache(clock=lambda: now[0])
    cache.set("k", _entry("V"))

    now[0] += 10_000_000.0
    got = cache.get("k")
    assert got is not None and got.masked_translation == "V"
