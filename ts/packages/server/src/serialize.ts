import type { LanguageCode, RoomTranslation, ViewerTranslation } from '@chat-translate/sdk';

/** JSON-safe shape of a RoomTranslation (the `byLanguage` Map flattened). */
export interface SerializedRoomTranslation {
  detectedSourceLang: LanguageCode;
  stats: RoomTranslation['stats'];
  byLanguage: Record<LanguageCode, ViewerTranslation>;
}

/**
 * `RoomTranslation.byLanguage` is a `Map`, which `JSON.stringify` turns into
 * `{}`. Flatten it to a plain object keyed by language code before sending.
 */
export function serializeRoomTranslation(rt: RoomTranslation): SerializedRoomTranslation {
  return {
    detectedSourceLang: rt.detectedSourceLang,
    stats: rt.stats,
    byLanguage: Object.fromEntries(rt.byLanguage),
  };
}
