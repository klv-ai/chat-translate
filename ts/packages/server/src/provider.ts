import {
  createChatTranslator,
  createProvider,
  createRoomTranslator,
  eldLanguageDetector,
  InMemoryTranslationCache,
  type LanguageDetector,
  type TranslationProvider,
} from '@chat-translate/sdk';
// Static entry: ships a preloaded ngrams database, so `detect()` is synchronous
// and needs no async load. Its instance already matches the SDK's EldLike shape
// (detect + setLanguageSubset), so it can be passed through without an adapter.
import { eld } from 'eld/large';
import type { ServerConfig } from './config';

export interface Translators {
  provider: TranslationProvider;
  chat: ReturnType<typeof createChatTranslator>;
  room: ReturnType<typeof createRoomTranslator>;
}

function buildDetector(cfg: ServerConfig): LanguageDetector {
  // Load-bearing on the TranslateGemma path (it cannot auto-detect a source);
  // a useful optimization on the DeepL path (same-language skip + concrete
  // cache keys). `fallback` is returned when ELD has no confident opinion.
  return eldLanguageDetector(eld, {
    subset: cfg.detectorSubset,
    fallback: cfg.defaultUiLang,
  });
}

/** Build the provider + chat/room translators once at boot from config. */
export function buildTranslators(cfg: ServerConfig): Translators {
  const provider = createProvider(cfg.deployment);
  const detector = buildDetector(cfg);
  const cache = new InMemoryTranslationCache({ maxEntries: 50_000, ttlMs: 3_600_000 });
  return {
    provider,
    chat: createChatTranslator(provider, { detector }),
    room: createRoomTranslator(provider, { detector, cache }),
  };
}
