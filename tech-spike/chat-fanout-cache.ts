/**
 * chat-fanout-cache.ts
 *
 * The fan-out + cache layer, sitting ABOVE chat-translation-layer.ts.
 *
 * A chat message is authored ONCE but viewed by many people, each in their own
 * language. This layer turns "one message" into "one translation per distinct
 * language currently being viewed in the room" — never per viewer, never into
 * languages nobody is reading.
 *
 * Two efficiencies do the heavy lifting:
 *
 *   1. Mask ONCE, restore PER TARGET. maskNonTranslatable() runs a single time;
 *      the resulting masked form is translated to every target, and each
 *      target's restore() re-inserts that message's own @mentions / URLs. The
 *      expensive, provider-bound work (translation) is what fans out; the cheap
 *      string work is all that repeats.
 *
 *   2. Cache on the MASKED text, not the raw text. Two messages that differ
 *      only by which user they @-mention or which URL they link mask to the
 *      same form — so "great work \uE000 0 \uE001" hits cache whether the
 *      mention was @alice or @bob. That is the whole reason masking happens
 *      before caching.
 */

import {
  maskNonTranslatable,
  DEFAULT_RULES,
  type LanguageDetector,
  type ViewerTranslation,
  type ViewerTranslateOptions,
} from './chat-translation-layer';
import {
  type TranslationProvider,
  type LanguageCode,
  type Formality,
  TranslationError,
} from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. Cache contract + a default in-memory implementation
// ─────────────────────────────────────────────────────────────

/**
 * What we cache is the provider's output BEFORE restore — i.e. still carrying
 * sentinels. Restore is per-message (different mentions), so it must NOT be
 * baked into the cached value.
 */
export interface CachedTranslation {
  maskedTranslation: string;
  detectedSourceLang: LanguageCode;
}

/**
 * Injectable so you can back it with an in-process LRU (below), Redis, or
 * anything else. Sync or async both satisfy the interface.
 */
export interface TranslationCache {
  get(
    key: string,
  ): Promise<CachedTranslation | undefined> | CachedTranslation | undefined;
  set(key: string, value: CachedTranslation): Promise<void> | void;
}

export interface InMemoryCacheOptions {
  /** Hard cap on entries; oldest-used evicted past this. Default 10k. */
  maxEntries?: number;
  /** Per-entry time-to-live in ms. Default: no expiry. */
  ttlMs?: number;
}

/**
 * A small LRU+TTL cache, fine for a single process. For a multi-instance
 * deployment swap in a shared store (Redis) so the cache is warm across nodes —
 * the key format below is already a flat string, so hash it and you're done.
 */
export class InMemoryTranslationCache implements TranslationCache {
  private store = new Map<string, { value: CachedTranslation; expires: number }>();
  constructor(private readonly opts: InMemoryCacheOptions = {}) {}

  get(key: string): CachedTranslation | undefined {
    const hit = this.store.get(key);
    if (!hit) return undefined;
    if (hit.expires !== Infinity && Date.now() > hit.expires) {
      this.store.delete(key);
      return undefined;
    }
    // LRU touch: re-insert so it becomes most-recently-used.
    this.store.delete(key);
    this.store.set(key, hit);
    return hit.value;
  }

  set(key: string, value: CachedTranslation): void {
    const ttl = this.opts.ttlMs ?? 0;
    this.store.set(key, {
      value,
      expires: ttl > 0 ? Date.now() + ttl : Infinity,
    });
    const max = this.opts.maxEntries ?? 10_000;
    while (this.store.size > max) {
      const oldest = this.store.keys().next().value;
      if (oldest === undefined) break;
      this.store.delete(oldest);
    }
  }
}

/**
 * Flat, collision-free key. NUL separates fields so no field value can bleed
 * into another. The masked text is included whole for correctness; a Redis
 * adapter should hash the whole string (the masked body can be long).
 */
function cacheKey(
  masked: string,
  source: LanguageCode | null,
  target: LanguageCode,
  formality: Formality | undefined,
  glossaryId: string | undefined,
): string {
  return ['v1', source ?? 'auto', target, formality ?? '-', glossaryId ?? '-', masked].join(
    '\u0000',
  );
}

// ─────────────────────────────────────────────────────────────
// 2. Bounded concurrency helper
// ─────────────────────────────────────────────────────────────

/**
 * Run tasks with at most `limit` in flight. For DeepL (managed, parallel-
 * friendly) a handful of concurrent calls cut latency. For the self-hosted
 * single-GPU path, limit 1 keeps things sane — see maxConcurrency default.
 */
async function runBounded(tasks: (() => Promise<void>)[], limit: number): Promise<void> {
  let i = 0;
  const lanes = Math.max(1, Math.min(limit, tasks.length));
  const workers = Array.from({ length: lanes }, async () => {
    while (i < tasks.length) {
      const idx = i++;
      await tasks[idx]();
    }
  });
  await Promise.all(workers);
}

// ─────────────────────────────────────────────────────────────
// 3. The room translator
// ─────────────────────────────────────────────────────────────

export interface RoomTranslatorConfig {
  /**
   * Strongly recommended even on the DeepL deployment. With a detector we can
   * (a) skip viewers who already speak the source language and (b) cache under
   * the *concrete* source instead of "auto", both big wins. Without one, the
   * DeepL path still works but pays a call for every target. Required outright
   * for TranslateGemma (it cannot auto-detect).
   */
  detector?: LanguageDetector;
  defaultFormality?: Formality;
  rules?: Parameters<typeof maskNonTranslatable>[1];
  /** Omit to disable caching. */
  cache?: TranslationCache;
  /** Default: caps.nativeBatch ? 8 : 1 (parallel for DeepL, serial for one GPU). */
  maxConcurrency?: number;
}

export interface RoomStats {
  /** Distinct target languages requested. */
  targets: number;
  /** Served from cache (no provider call). */
  cacheHits: number;
  /** Actually sent to the provider. */
  translated: number;
  /** Returned as-is (same language as source, or nothing translatable). */
  shortCircuited: number;
}

export interface RoomTranslation {
  /** target language → result for any viewer reading in that language. */
  byLanguage: Map<LanguageCode, ViewerTranslation>;
  detectedSourceLang: LanguageCode;
  stats: RoomStats;
}

/** Compare base subtags: "pt-BR" and "pt" count as the same language. */
function sameLang(a: LanguageCode, b: LanguageCode): boolean {
  return a.split('-')[0].toLowerCase() === b.split('-')[0].toLowerCase();
}

const PLACEHOLDER = /\uE000\d+\uE001/g;

export function createRoomTranslator(
  provider: TranslationProvider,
  config: RoomTranslatorConfig = {},
) {
  const caps = provider.capabilities();
  const rules = config.rules ?? DEFAULT_RULES;
  const defaultFormality: Formality = config.defaultFormality ?? 'prefer_less';
  const cache = config.cache;
  const maxConcurrency = config.maxConcurrency ?? (caps.nativeBatch ? 8 : 1);

  // Source resolution, ordered so a configured detector is always preferred:
  // it gives us routing (same-lang skip) and concrete cache keys for BOTH
  // backends. Auto-detect is the fallback only when no detector is supplied.
  async function resolveSource(
    detectInput: string,
    explicit?: LanguageCode,
  ): Promise<LanguageCode | null> {
    if (explicit) return explicit;
    if (config.detector) return await config.detector.detect(detectInput);
    if (caps.autoDetectSource) return null; // provider detects; no pre-skip
    throw new TranslationError(
      `${provider.name} cannot auto-detect and no detector is configured`,
      provider.name,
    );
  }

  /**
   * Translate one freshly-posted message for every language currently being
   * viewed in the room. Pass only the languages people are actually reading —
   * not every language the room has ever contained.
   */
  async function translateForRoom(
    raw: string,
    viewedLanguages: Iterable<LanguageCode>,
    opts: ViewerTranslateOptions = {},
  ): Promise<RoomTranslation> {
    const { masked, restore, findUnrestored } = maskNonTranslatable(raw, rules);
    const targets = [...new Set(viewedLanguages)];
    const byLanguage = new Map<LanguageCode, ViewerTranslation>();
    const stats: RoomStats = {
      targets: targets.length,
      cacheHits: 0,
      translated: 0,
      shortCircuited: 0,
    };

    // Short-circuit A: nothing translatable (pure emoji / mention / code / url).
    if (masked.replace(PLACEHOLDER, '').trim() === '') {
      for (const t of targets) {
        byLanguage.set(t, {
          text: raw,
          detectedSourceLang: opts.sourceLang ?? t,
          translated: false,
          unrestoredTokens: [],
        });
      }
      stats.shortCircuited = targets.length;
      return { byLanguage, detectedSourceLang: opts.sourceLang ?? targets[0] ?? '', stats };
    }

    const detectInput = masked.replace(PLACEHOLDER, ' ');
    const sourceLang = await resolveSource(detectInput, opts.sourceLang);
    let detectedSourceLang = sourceLang ?? '';

    // Formality is target-independent — resolve once for the call AND the key.
    const formality: Formality | undefined = caps.formality
      ? opts.formality ?? defaultFormality
      : undefined;

    // Short-circuit B: viewers already on the source language get the original.
    // Only possible when we know the source up front (explicit or detector).
    const need: LanguageCode[] = [];
    for (const t of targets) {
      if (sourceLang && sameLang(sourceLang, t)) {
        byLanguage.set(t, {
          text: raw,
          detectedSourceLang: sourceLang,
          translated: false,
          unrestoredTokens: [],
        });
        stats.shortCircuited++;
      } else {
        need.push(t);
      }
    }

    // One translation per remaining distinct target, cache-aware, bounded.
    const tasks = need.map((target) => async () => {
      const key = cacheKey(masked, sourceLang, target, formality, opts.glossaryId);

      let maskedTranslation: string;
      let det: LanguageCode;

      const cached = cache ? await cache.get(key) : undefined;
      if (cached) {
        maskedTranslation = cached.maskedTranslation;
        det = cached.detectedSourceLang;
        stats.cacheHits++;
      } else {
        const res = await provider.translate(masked, {
          sourceLang,
          targetLang: target,
          formality,
          context: opts.context,
          glossaryId: opts.glossaryId,
        });
        maskedTranslation = res.text;
        det = res.detectedSourceLang;
        stats.translated++;
        if (cache) await cache.set(key, { maskedTranslation, detectedSourceLang: det });
      }

      if (det) detectedSourceLang = det;
      byLanguage.set(target, {
        text: restore(maskedTranslation),
        detectedSourceLang: det || sourceLang || target,
        translated: true,
        unrestoredTokens: findUnrestored(maskedTranslation),
      });
    });

    await runBounded(tasks, maxConcurrency);

    return { byLanguage, detectedSourceLang, stats };
  }

  return { capabilities: caps, translateForRoom };
}

/* ─────────────────────────────────────────────────────────────
 * Usage:
 *
 *   import { createProvider, configFromEnv } from './translation-provider';
 *
 *   const provider = createProvider(configFromEnv());
 *   const room = createRoomTranslator(provider, {
 *     detector: { detect: (t) => franc(t) }, // recommended on every backend
 *     cache: new InMemoryTranslationCache({ maxEntries: 50_000, ttlMs: 3_600_000 }),
 *     defaultFormality: 'prefer_less',
 *   });
 *
 *   // A room where people are reading in German, French, and English:
 *   const out = await room.translateForRoom(
 *     'hey @alice ship it 🚀 see https://x.com/pr/42',
 *     ['de', 'fr', 'en'],
 *     { context: recentMessages.join('\n') },
 *   );
 *
 *   out.byLanguage.get('de')!.text;     // German, mention + 🚀 + URL intact
 *   out.detectedSourceLang;             // 'en' (the author wrote English)
 *   out.byLanguage.get('en')!.translated; // false — English viewer, source was English
 *   out.stats;                          // { targets: 3, translated: 2, shortCircuited: 1, ... }
 * ───────────────────────────────────────────────────────────── */
