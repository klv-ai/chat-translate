import { afterEach, describe, expect, it, vi } from 'vitest';
import { InMemoryTranslationCache } from '../src/chat-fanout-cache';

const entry = (s: string) => ({ maskedTranslation: s, detectedSourceLang: 'en' });

describe('InMemoryTranslationCache', () => {
  afterEach(() => vi.useRealTimers());

  it('evicts the least-recently-used entry past maxEntries', () => {
    const cache = new InMemoryTranslationCache({ maxEntries: 2 });
    cache.set('a', entry('A'));
    cache.set('b', entry('B'));
    expect(cache.get('a')?.maskedTranslation).toBe('A'); // touch a → b is now LRU
    cache.set('c', entry('C')); // size 3 > 2 → evict b

    expect(cache.get('b')).toBeUndefined();
    expect(cache.get('a')?.maskedTranslation).toBe('A');
    expect(cache.get('c')?.maskedTranslation).toBe('C');
  });

  it('expires entries once their TTL elapses', () => {
    vi.useFakeTimers();
    const cache = new InMemoryTranslationCache({ ttlMs: 1000 });
    cache.set('k', entry('V'));
    expect(cache.get('k')?.maskedTranslation).toBe('V');

    vi.advanceTimersByTime(1500);
    expect(cache.get('k')).toBeUndefined();
  });

  it('never expires when no TTL is configured', () => {
    vi.useFakeTimers();
    const cache = new InMemoryTranslationCache();
    cache.set('k', entry('V'));
    vi.advanceTimersByTime(10_000_000);
    expect(cache.get('k')?.maskedTranslation).toBe('V');
  });
});
