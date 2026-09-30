/**
 * resolve-source.ts
 *
 * Ingest-time source-language resolution. Runs ONCE, when a message is posted,
 * and the result is written onto the message row (see ResolvedSource). Every
 * later translation — live fan-out and history replay alike — reads the source
 * off the row, so detection never runs on the hot path and the cache + the
 * persistent store can't drift onto different source languages.
 *
 * Precedence (highest signal first):
 *   1. user-confirmed override  — a replay-widget correction is ground truth.
 *   2. confident detection      — the author wrote something other than their
 *                                 UI language, and the detector is sure.
 *   3. UI-language prior         — the deterministic, cold-start-free fallback.
 *
 * Two deliberate properties:
 *   - It NEVER returns null. The TranslateGemma path can't auto-detect, and a
 *     persisted row needs a concrete fact regardless of backend.
 *   - It's deterministic. The same masked input + same uiLang always resolves
 *     identically, so cache keys (which include the source) never fragment.
 *
 * Runtime: Node 18+. No npm dependencies (the detector is injected).
 */

import type { LanguageCode } from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. Confidence-bearing detector contract
// ─────────────────────────────────────────────────────────────

/**
 * Richer than chat-translation-layer's LanguageDetector (which returns a bare
 * code) because ingest NEEDS the confidence verdict to drive the gate and the
 * short-text guard below. The code-only detector used elsewhere is just the
 * degenerate "already resolved, trust it" case.
 */
export interface DetectionResult {
  /** Best guess, or null when the detector has no opinion at all. */
  lang: LanguageCode | null;
  /** The detector's own reliability verdict (e.g. ELD's isReliable()). */
  reliable: boolean;
  /** Optional top score in [0,1], when the detector exposes one. */
  score?: number;
}

export interface ConfidenceDetector {
  detect(text: string): Promise<DetectionResult> | DetectionResult;
}

/**
 * Adapt nitotm `eld` to ConfidenceDetector. Constrain to the room's plausible
 * languages once, at construction — improves both accuracy and latency. ELD
 * already emits ISO 639-1 codes, so no code mapping is needed.
 */
type EldLike = {
  detect(t: string): {
    language: string;
    isReliable(): boolean;
    getScores(): Record<string, number>;
  };
  setLanguageSubset(s: string[] | false): unknown;
};

export function eldDetector(eld: EldLike, subset?: LanguageCode[]): ConfidenceDetector {
  if (subset) eld.setLanguageSubset(subset);
  return {
    detect(text: string): DetectionResult {
      const r = eld.detect(text);
      const top = r.language ? r.getScores()[r.language] : undefined;
      return {
        lang: r.language || null,
        reliable: r.isReliable(),
        score: typeof top === 'number' ? top : undefined,
      };
    },
  };
}

// ─────────────────────────────────────────────────────────────
// 2. Result + options
// ─────────────────────────────────────────────────────────────

export type SourceProvenance = 'user_confirmed' | 'detected' | 'ui_fallback';

/**
 * What to do in the short-text loanword danger zone when a reliable detection
 * DISAGREES with uiLang. The single source of truth for this union — the eval's
 * policy sweep and the logger's config stamp both import it from here.
 *
 *   'prefer_prior'   → keep uiLang. Safest against loanword false positives
 *                      ("ok" / "ciao" / "cool" detected as the wrong language).
 *   'trust_detector' → take the detection. Better when users code-switch in
 *                      short bursts; costs you on loanwords.
 *
 * Either branch marks the row confident:false, so it stays revisitable.
 */
export type ShortTextPolicy = 'prefer_prior' | 'trust_detector';

export interface ResolvedSource {
  /** The source language to store, translate FROM, and key the cache on. Never null. */
  lang: LanguageCode;
  /** Where `lang` came from — drives UI affordances and re-detection eligibility. */
  provenance: SourceProvenance;
  /**
   * true  → settled (user-confirmed, or a clean confident detection).
   * false → we fell back or guarded; a replay correction or a stronger model
   *         later could improve it. PERSIST this so such rows stay re-visitable
   *         (e.g. re-detect the whole `confident:false` backlog after a model
   *         upgrade, or nudge the replay widget more prominently in the UI).
   */
  confident: boolean;
  /** What the detector actually said, even when unused — telemetry / future re-detect. */
  detected?: LanguageCode;
}

export interface ResolveSourceOptions {
  /** Author's chosen UI language — the fallback prior. Present from message #1. */
  uiLang: LanguageCode;
  /** User-confirmed correction (replay widget). Strongest possible signal. */
  override?: LanguageCode;
  /**
   * Below this many letters we're in the loanword danger zone: a detector can be
   * *confidently wrong* on "ok" / "ciao" / "cool". There a detection that AGREES
   * with uiLang is trusted; on disagreement `shortTextPolicy` decides. Counts
   * Unicode letters in the (already masked) detect input. Default 10.
   */
  shortTextLetters?: number;
  /**
   * Behaviour in the short-text zone when a reliable detection disagrees with
   * uiLang. Default `prefer_prior` (the conservative loanword-safe choice).
   * This is a data-driven knob — flip it once the eval's short-disagreement
   * ratio says code-switching outweighs loanword false positives.
   */
  shortTextPolicy?: ShortTextPolicy;
  /** Optional floor on the detector's top score, layered on top of its reliable flag. */
  minScore?: number;
}

// ─────────────────────────────────────────────────────────────
// 3. The resolver
// ─────────────────────────────────────────────────────────────

export const DEFAULT_SHORT_TEXT_LETTERS = 10;
export const DEFAULT_SHORT_TEXT_POLICY: ShortTextPolicy = 'prefer_prior';

/** Compare base subtags: "pt-BR" and "pt" count as the same language. */
function sameLang(a: LanguageCode, b: LanguageCode): boolean {
  return a.split('-')[0].toLowerCase() === b.split('-')[0].toLowerCase();
}

/**
 * Letters only — masked input has had mentions/URLs/emoji/code turned to spaces.
 * Exported because source-resolution-eval.ts reuses it to bin cases by length;
 * it is a pure Unicode-letter counter with no side effects.
 */
export function letterCount(s: string): number {
  return (s.match(/\p{L}/gu) ?? []).length;
}

/**
 * Resolve one freshly-posted message's source language at ingest.
 *
 * `detectInput` is the MASKED text with sentinels replaced by spaces — i.e. the
 * same string the orchestration layer already builds for detection, free of
 * @mentions / URLs / emoji that would skew an n-gram detector.
 */
export async function resolveSource(
  detectInput: string,
  detector: ConfidenceDetector,
  opts: ResolveSourceOptions,
): Promise<ResolvedSource> {
  const { uiLang, override } = opts;

  // 1. A user correction is ground truth. Don't even detect — just record it.
  if (override) {
    return { lang: override, provenance: 'user_confirmed', confident: true };
  }

  // 2. Detect.
  const det = await detector.detect(detectInput);
  const detected = det.lang ?? undefined;
  const passesScore = opts.minScore === undefined || (det.score ?? 0) >= opts.minScore;
  // 3. Nothing usable → UI prior, flagged revisitable. The guard also narrows
  //    det.lang to a concrete code for everything below.
  if (det.lang === null || !det.reliable || !passesScore) {
    return { lang: uiLang, provenance: 'ui_fallback', confident: false, detected };
  }
  const lang = det.lang;

  // 4. Enough letters that we trust the detector outright (incl. when it
  //    overrides the prior — the German-UI user writing English).
  const threshold = opts.shortTextLetters ?? DEFAULT_SHORT_TEXT_LETTERS;
  if (letterCount(detectInput) >= threshold) {
    return { lang, provenance: 'detected', confident: true, detected };
  }

  // 5. Short input, loanword danger zone. A detection that corroborates the
  //    prior is trusted outright. On DISAGREEMENT the policy decides — but
  //    either branch marks confident:false so the row stays revisitable.
  if (sameLang(lang, uiLang)) {
    return { lang, provenance: 'detected', confident: true, detected };
  }
  const policy = opts.shortTextPolicy ?? DEFAULT_SHORT_TEXT_POLICY;
  if (policy === 'trust_detector') {
    // Took the detector's answer, but we're in the danger zone — not settled.
    return { lang, provenance: 'detected', confident: false, detected };
  }
  // prefer_prior: keep uiLang, record what the detector said for telemetry.
  return { lang: uiLang, provenance: 'ui_fallback', confident: false, detected };
}
