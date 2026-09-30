/**
 * detector.ts — adapters bridging a confidence-bearing detector down to the
 * bare-code `LanguageDetector` the chat / room layers consume.
 *
 * Why this exists: ingest-time `resolveSource` needs a `ConfidenceDetector`
 * (returns `{ lang, reliable, score }`), but `createChatTranslator` /
 * `createRoomTranslator` only want a `LanguageDetector` (returns a concrete
 * code). `eldDetector` produces the former; these helpers narrow it to the
 * latter, supplying a concrete `fallback` for the case where the detector has
 * no opinion (ELD can return `null`) — the chat layer must never receive null.
 *
 * The `eld` instance is INJECTED (never imported here) so the SDK core stays
 * dependency-free; `eld` is an optional peer dependency. Reach this module via
 * the `@chat-translate/sdk/detector` subpath.
 */

import type { LanguageDetector } from './chat-translation-layer';
import { type ConfidenceDetector, eldDetector } from './resolve-source';
import type { LanguageCode } from './translation-provider';

/** Narrow a ConfidenceDetector to a bare-code LanguageDetector. */
export function toLanguageDetector(
  detector: ConfidenceDetector,
  fallback: LanguageCode,
): LanguageDetector {
  return {
    async detect(text: string): Promise<LanguageCode> {
      const r = await detector.detect(text);
      return r.lang ?? fallback;
    },
  };
}

/** The structural shape `eldDetector` expects of an injected `eld` instance. */
export type EldLike = Parameters<typeof eldDetector>[0];

export interface EldLanguageDetectorOptions {
  /** Constrain detection to the room's plausible languages (ISO 639-1). */
  subset?: LanguageCode[];
  /** Code returned when ELD has no confident opinion. */
  fallback: LanguageCode;
}

/**
 * Convenience: build a bare-code `LanguageDetector` straight from an injected
 * `eld` instance, applying an optional language subset and a required fallback.
 */
export function eldLanguageDetector(
  eld: EldLike,
  opts: EldLanguageDetectorOptions,
): LanguageDetector {
  return toLanguageDetector(eldDetector(eld, opts.subset), opts.fallback);
}
