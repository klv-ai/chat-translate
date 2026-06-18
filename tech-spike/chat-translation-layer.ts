/**
 * chat-translation-layer.ts
 *
 * The provider-agnostic chat layer that sits ABOVE the TranslationProvider
 * boundary. This is where the actual product engineering lives — the provider
 * underneath just translates a string.
 *
 * Responsibilities (this file):
 *   1. Do-not-translate masking: @mentions, emails, URLs, #channels, emoji,
 *      and code are replaced with opaque sentinels, translated around, and
 *      restored verbatim. Naive translation mangles all of these.
 *   2. Capability-driven orchestration: detect the source language ourselves
 *      when the backend can't (TranslateGemma), let it auto-detect when it can
 *      (DeepL) — so both providers behave identically from here up.
 *   3. Cheap short-circuits: skip the network entirely when there is nothing
 *      to translate, or when the viewer already speaks the source language.
 *
 * Deliberately NOT here (fast-follow): per-viewer fan-out and caching. The
 * seams for both are set up below — see createChatTranslator().mask and the
 * note on translateForViewer.
 */

import {
  type TranslationProvider,
  type TranslateOptions,
  type LanguageCode,
  type Formality,
  TranslationError,
} from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. Masking
// ─────────────────────────────────────────────────────────────

/**
 * Sentinels come from the Unicode Private Use Area — they virtually never
 * appear in real chat text, so collisions are negligible and we strip any
 * that somehow arrive in user input before masking.
 *
 * Robustness note: this opaque-token scheme survives NMT (DeepL) extremely
 * well — the engine treats `\uE000 12 \uE001` as a proper noun and leaves it
 * alone, even reordering it to the grammatically correct spot, which is what
 * we want. A raw LLM (TranslateGemma) is *mostly* reliable but can
 * occasionally drop or alter a sentinel; unrestoredTokens (below) surfaces
 * that so the caller can fall back. If you want belt-and-suspenders on the
 * DeepL deployment, its native `tag_handling`/`ignore_tags` is stronger — but
 * that's provider-specific, so we don't depend on it here.
 */
const SENTINEL_OPEN = '\uE000';
const SENTINEL_CLOSE = '\uE001';

interface MaskRule {
  name: string;
  pattern: RegExp; // must be global
  /** Trim trailing sentence punctuation out of the masked span (URLs). */
  trimTrailingPunct?: boolean;
}

/**
 * Order matters. Structural spans (code) are masked first so later rules can't
 * reach inside them; by the time URLs/mentions run, code is already an opaque
 * sentinel. Each masked span becomes a token no later rule can match.
 */
export const DEFAULT_RULES: MaskRule[] = [
  { name: 'fenced_code', pattern: /```[\s\S]*?```/g },
  { name: 'inline_code', pattern: /`[^`\n]+`/g },
  { name: 'email', pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g },
  { name: 'url', pattern: /\b(?:https?:\/\/|www\.)[^\s<]+/gi, trimTrailingPunct: true },
  // @mention / #channel: only when preceded by whitespace or start-of-string,
  // so we don't eat the "@" inside an email or a mid-word "#".
  { name: 'mention', pattern: /(?<![^\s])@[\w.\-]+/g },
  { name: 'channel', pattern: /(?<![^\s])#[\w\-]+/g },
  // :shortcode: emoji (Slack/Discord style) — literal text, never translate.
  { name: 'shortcode_emoji', pattern: /:[a-z0-9_+\-]+:/gi },
  // Unicode emoji (approximate: base pictographic + modifiers/ZWJ, or flags).
  // Keycap sequences (1️⃣) and a few exotic combos are edge cases.
  {
    name: 'unicode_emoji',
    pattern:
      /\p{Extended_Pictographic}[\u{1F3FB}-\u{1F3FF}\uFE0F]*(?:\u200D\p{Extended_Pictographic}[\u{1F3FB}-\u{1F3FF}\uFE0F]*)*|\p{Regional_Indicator}{2}/gu,
  },
];

export interface MaskedMessage {
  /** Text with non-translatable spans replaced by sentinels. */
  masked: string;
  /** Re-insert the originals into a translated string. */
  restore: (translated: string) => string;
  /** Token ids that did NOT survive translation (should be empty). */
  findUnrestored: (translated: string) => number[];
  /** How many spans were masked. */
  tokenCount: number;
}

/**
 * Pure, provider-independent, and cheap. Call it ONCE per message — the
 * fan-out layer will reuse the single masked form across every target
 * language and only re-run restore() per viewer.
 */
export function maskNonTranslatable(
  raw: string,
  rules: MaskRule[] = DEFAULT_RULES,
): MaskedMessage {
  // Defensive: drop any pre-existing sentinels so user input can't collide.
  let work = raw.replace(/[\uE000\uE001]/g, '');
  const tokens: string[] = [];

  for (const rule of rules) {
    work = work.replace(rule.pattern, (match) => {
      let value = match;
      let tail = '';
      if (rule.trimTrailingPunct) {
        const t = value.match(/[.,!?;:)\]]+$/);
        if (t) {
          tail = t[0];
          value = value.slice(0, -tail.length);
        }
      }
      const id = tokens.length;
      tokens.push(value);
      return `${SENTINEL_OPEN}${id}${SENTINEL_CLOSE}${tail}`;
    });
  }

  const restore = (translated: string): string =>
    translated.replace(
      /\uE000(\d+)\uE001/g,
      (_, d: string) => tokens[Number(d)] ?? '',
    );

  const findUnrestored = (translated: string): number[] => {
    const seen = new Set<number>();
    for (const m of translated.matchAll(/\uE000(\d+)\uE001/g)) {
      seen.add(Number(m[1]));
    }
    return tokens.map((_, i) => i).filter((i) => !seen.has(i));
  };

  return { masked: work, restore, findUnrestored, tokenCount: tokens.length };
}

// ─────────────────────────────────────────────────────────────
// 2. Orchestration
// ─────────────────────────────────────────────────────────────

/** Injected so the layer stays testable and backend-agnostic (CLD3, fastText, franc…). */
export interface LanguageDetector {
  detect(text: string): Promise<LanguageCode> | LanguageCode;
}

export interface ChatTranslatorConfig {
  /** Required ONLY when the provider cannot auto-detect (e.g. TranslateGemma). */
  detector?: LanguageDetector;
  /** Applied only when the provider supports formality. Chat → informal default. */
  defaultFormality?: Formality;
  rules?: MaskRule[];
}

export interface ViewerTranslateOptions {
  /** Force a source language, skipping detection. */
  sourceLang?: LanguageCode;
  /** Prior messages for disambiguation — passed through, never translated. */
  context?: string;
  formality?: Formality;
  glossaryId?: string;
}

export interface ViewerTranslation {
  text: string;
  detectedSourceLang: LanguageCode;
  /** false = short-circuited (nothing to translate, or same language). */
  translated: boolean;
  /** Non-empty means the engine ate placeholders — inspect / fall back. */
  unrestoredTokens: number[];
}

/** Compare base subtags: "pt-BR" and "pt" count as the same language. */
function sameLang(a: LanguageCode, b: LanguageCode): boolean {
  return a.split('-')[0].toLowerCase() === b.split('-')[0].toLowerCase();
}

export function createChatTranslator(
  provider: TranslationProvider,
  config: ChatTranslatorConfig = {},
) {
  const caps = provider.capabilities();
  const rules = config.rules ?? DEFAULT_RULES;
  const defaultFormality: Formality = config.defaultFormality ?? 'prefer_less';

  async function resolveSource(
    detectInput: string,
    explicit?: LanguageCode,
  ): Promise<LanguageCode | null> {
    if (explicit) return explicit;
    if (caps.autoDetectSource) return null; // let the provider do it
    // Capability says it can't — this is the worked-example degradation path.
    if (!config.detector) {
      throw new TranslationError(
        `${provider.name} cannot auto-detect source and no detector is configured`,
        provider.name,
      );
    }
    return await config.detector.detect(detectInput);
  }

  /**
   * Translate one message for one viewer's target language.
   *
   * The fan-out layer (next) will call this per distinct target — but more
   * efficiently: it will mask once via the exported `mask` below, then loop
   * targets over the already-masked form rather than re-masking each time.
   */
  async function translateForViewer(
    raw: string,
    targetLang: LanguageCode,
    opts: ViewerTranslateOptions = {},
  ): Promise<ViewerTranslation> {
    const { masked, restore, findUnrestored } = maskNonTranslatable(raw, rules);

    // Short-circuit 1: nothing translatable (pure emoji / mention / code / url).
    const bare = masked.replace(/\uE000\d+\uE001/g, '').trim();
    if (bare === '') {
      return {
        text: raw,
        detectedSourceLang: opts.sourceLang ?? targetLang,
        translated: false,
        unrestoredTokens: [],
      };
    }

    const detectInput = masked.replace(/\uE000\d+\uE001/g, ' ');
    const sourceLang = await resolveSource(detectInput, opts.sourceLang);

    // Short-circuit 2: the viewer already speaks the source language.
    if (sourceLang && sameLang(sourceLang, targetLang)) {
      return {
        text: raw,
        detectedSourceLang: sourceLang,
        translated: false,
        unrestoredTokens: [],
      };
    }

    const translateOpts: TranslateOptions = {
      sourceLang,
      targetLang,
      formality: caps.formality ? (opts.formality ?? defaultFormality) : undefined,
      context: opts.context,
      glossaryId: opts.glossaryId,
    };

    const result = await provider.translate(masked, translateOpts);
    const text = restore(result.text);

    return {
      text,
      detectedSourceLang: result.detectedSourceLang,
      translated: true,
      unrestoredTokens: findUnrestored(result.text),
    };
  }

  return {
    capabilities: caps,
    translateForViewer,
    /** Exposed so the fan-out layer can mask once and reuse across viewers. */
    mask: (raw: string): MaskedMessage => maskNonTranslatable(raw, rules),
  };
}

/* ─────────────────────────────────────────────────────────────
 * Usage:
 *
 *   import { createProvider, configFromEnv } from './translation-provider';
 *
 *   const provider = createProvider(configFromEnv());
 *   const chat = createChatTranslator(provider, {
 *     // Only needed if provider.capabilities().autoDetectSource === false:
 *     detector: { detect: (t) => franc(t) },   // your detector of choice
 *     defaultFormality: 'prefer_less',          // casual room
 *   });
 *
 *   const out = await chat.translateForViewer(
 *     'hey @alice did you see https://x.com/foo? lgtm 👍 `npm run build`',
 *     'de',
 *     { context: previousMessages.join('\n') },
 *   );
 *   // out.text → German, with @alice / the URL / 👍 / the code span intact.
 *   // out.unrestoredTokens → [] on the DeepL deployment; watch it on the LLM one.
 * ───────────────────────────────────────────────────────────── */
