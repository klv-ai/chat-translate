/**
 * source-resolution-eval.ts
 *
 * A measured stage for the ingest-time source resolver (resolve-source.ts),
 * SEPARATE from translation-eval.ts on purpose: that harness grades a provider's
 * translated OUTPUT (chrF / judge / placeholder survival); this one grades a
 * DECISION — "what source language did we resolve, and was it right?" — against
 * the gold labels already in sample-chat-eval.jsonl. Different ground truth,
 * different sweep, so it's a sibling stage rather than something bolted into
 * runEval.
 *
 * It answers the one question we deliberately left to data instead of guessing:
 * on SHORT chat messages where the detector disagrees with the author's UI
 * language, do we keep the prior (safe against "ok"/"ciao" loanword false
 * positives) or trust the detector (better when people code-switch in short
 * bursts)? That is resolve-source.ts's `shortTextPolicy`, and this measures both
 * settings on the real masking pipeline.
 *
 * Two outputs:
 *   1. Detector calibration — ASSUMPTION-FREE. Bins cases by letter count and
 *      reports how often the detector is reliable and correct at each length.
 *      This is the raw signal for choosing shortTextLetters; it needs no uiLang.
 *   2. Policy sweep — runs the REAL resolveSource across thresholds × policies,
 *      scores resolved-vs-gold, and breaks out the short-disagreement zone where
 *      the policy actually bites. That breakdown (detector-right vs prior-right)
 *      is the crux, and it's where the uiLang model matters — see below.
 *
 * Nothing here calls a translation provider, so it runs cheaply offline; it
 * exercises masking + the resolver only.
 *
 * Runtime: Node 18+. The detector is injected (e.g. ELD via eldDetector).
 */

import { maskNonTranslatable, DEFAULT_RULES } from './chat-translation-layer';
import {
  resolveSource,
  letterCount,
  type ConfidenceDetector,
  type DetectionResult,
  type ResolveSourceOptions,
} from './resolve-source';
import type { LanguageCode } from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. Cases + shared helpers
// ─────────────────────────────────────────────────────────────

/**
 * Gold-labelled case. Intentionally a subset of the sample-chat-eval.jsonl row
 * shape — we only need the text and its gold source language here.
 */
export interface SourceEvalCase {
  id: string;
  text: string;
  /** GOLD source language — the label we score against. */
  sourceLang: LanguageCode;
  /** Optional real UI language for this case; otherwise supplied by the uiLang model. */
  uiLang?: LanguageCode;
}

const PLACEHOLDER = /\uE000\d+\uE001/g;

/** Build the exact detectInput production feeds the resolver: masked, sentinels→spaces. */
function detectInputFor(text: string, rules: typeof DEFAULT_RULES = DEFAULT_RULES): string {
  const { masked } = maskNonTranslatable(text, rules);
  return masked.replace(PLACEHOLDER, ' ');
}

/** Base-subtag comparison, matching the resolver and the rest of the codebase. */
function sameLang(a: LanguageCode, b: LanguageCode): boolean {
  return a.split('-')[0].toLowerCase() === b.split('-')[0].toLowerCase();
}

function pct(num: number, den: number): string {
  return den === 0 ? '—' : ((num / den) * 100).toFixed(0);
}

/**
 * How to obtain a uiLang for a case that doesn't carry one.
 *  - NATIVE_UI (default) models the common "writes in their native language"
 *    assumption (uiLang = gold). Under it, every short disagreement is by
 *    construction a detector error the prior rescues — it isolates the
 *    loanword-protection side of the tradeoff.
 *  - linguaFrancaUi('en') pretends everyone's UI is English, so any message
 *    authored in another language reads as a code-switch relative to the prior —
 *    it surfaces the OTHER side, where trusting the detector wins.
 * Run both to bracket the truth; your real mix sits somewhere between.
 */
export type UiLangModel = (c: SourceEvalCase) => LanguageCode;
const NATIVE_UI: UiLangModel = (c) => c.uiLang ?? c.sourceLang;
export const linguaFrancaUi =
  (lingua: LanguageCode): UiLangModel =>
  (c) =>
    c.uiLang ?? lingua;

// ─────────────────────────────────────────────────────────────
// 2. Detector calibration (assumption-free)
// ─────────────────────────────────────────────────────────────

export interface CalibrationBin {
  /** Inclusive lower bound on letter count for this bin. */
  minLetters: number;
  /** Translatable cases that fell in this bin. */
  n: number;
  /** Of n, how many the detector flagged reliable. */
  reliable: number;
  /** Of the reliable ones, how many matched gold (the number that drives the threshold). */
  reliableCorrect: number;
  /** Of n, how many matched gold regardless of the reliable flag. */
  correct: number;
}

export interface CalibrationReport {
  bins: CalibrationBin[];
  /** Cases skipped because nothing was translatable (pure emoji/mention/url/code). */
  skippedUntranslatable: string[];
}

/**
 * Bin cases by letter count and report detector accuracy per bin. The column
 * that matters is correct|reliable% — set shortTextLetters at the length where
 * it stops being trustworthy.
 */
export async function calibrateDetector(
  cases: SourceEvalCase[],
  detector: ConfidenceDetector,
  binLowerBounds: number[] = [0, 5, 10, 20, 40],
  rules: typeof DEFAULT_RULES = DEFAULT_RULES,
): Promise<CalibrationReport> {
  const bounds = [...binLowerBounds].sort((a, b) => a - b);
  const bins: CalibrationBin[] = bounds.map((minLetters) => ({
    minLetters,
    n: 0,
    reliable: 0,
    reliableCorrect: 0,
    correct: 0,
  }));
  const skippedUntranslatable: string[] = [];

  for (const c of cases) {
    const di = detectInputFor(c.text, rules);
    const letters = letterCount(di);
    if (letters === 0) {
      skippedUntranslatable.push(c.id);
      continue;
    }

    let bi = 0;
    for (let i = 0; i < bounds.length; i++) if (letters >= bounds[i]) bi = i;
    const bin = bins[bi];

    const det = await detector.detect(di);
    const correct = det.lang != null && sameLang(det.lang, c.sourceLang);
    bin.n++;
    if (correct) bin.correct++;
    if (det.lang != null && det.reliable) {
      bin.reliable++;
      if (correct) bin.reliableCorrect++;
    }
  }

  return { bins, skippedUntranslatable };
}

export function formatCalibrationMarkdown(r: CalibrationReport): string {
  const cols = ['letters ≥', 'n', 'reliable%', 'correct|reliable%', 'correct%'];
  const line = (cells: string[]) => `| ${cells.join(' | ')} |`;
  const rows = r.bins
    .filter((b) => b.n > 0)
    .map((b) => [
      String(b.minLetters),
      String(b.n),
      pct(b.reliable, b.n),
      pct(b.reliableCorrect, b.reliable),
      pct(b.correct, b.n),
    ]);
  return [
    '### Detector calibration (assumption-free)',
    line(cols),
    line(cols.map(() => '---')),
    ...rows.map(line),
    '',
    `_skipped (nothing translatable): ${r.skippedUntranslatable.join(', ') || 'none'}_`,
    '_set `shortTextLetters` at the length where `correct|reliable%` falls off._',
  ].join('\n');
}

// ─────────────────────────────────────────────────────────────
// 3. Policy sweep (runs the real resolveSource)
// ─────────────────────────────────────────────────────────────

export type ShortTextPolicy = 'prefer_prior' | 'trust_detector';

export interface PolicyCellResult {
  shortTextLetters: number;
  policy: ShortTextPolicy;
  /** Translatable cases scored. */
  n: number;
  /** resolved.lang == gold. */
  correct: number;
  /** Cases that exercised the policy: short + reliable detection that DISAGREES with uiLang. */
  shortDisagree: number;
  /** Of shortDisagree, how many the DETECTOR had right (== gold). */
  shortDisagreeDetectorRight: number;
  /** Of shortDisagree, how many the PRIOR had right (uiLang == gold). */
  shortDisagreePriorRight: number;
}

export interface PolicyReport {
  cells: PolicyCellResult[];
  uiLangModelName: string;
  skippedUntranslatable: string[];
}

export interface SweepOptions {
  thresholds?: number[];
  policies?: ShortTextPolicy[];
  uiLangModel?: UiLangModel;
  uiLangModelName?: string;
  minScore?: number;
  rules?: typeof DEFAULT_RULES;
}

export async function sweepPolicies(
  cases: SourceEvalCase[],
  detector: ConfidenceDetector,
  opts: SweepOptions = {},
): Promise<PolicyReport> {
  const thresholds = opts.thresholds ?? [6, 8, 10, 12, 16];
  const policies = opts.policies ?? ['prefer_prior', 'trust_detector'];
  const uiLangModel = opts.uiLangModel ?? NATIVE_UI;
  const rules = opts.rules ?? DEFAULT_RULES;

  // Detection depends only on the masked input, so compute it once per case and
  // reuse across every (threshold × policy) cell. resolveSource re-detects
  // internally — that's fine and deterministic — but the breakdown reads this
  // precomputed verdict so it lines up exactly with what the resolver saw.
  type Prepared = { c: SourceEvalCase; di: string; det: DetectionResult; uiLang: LanguageCode };
  const prepared: Prepared[] = [];
  const skippedUntranslatable: string[] = [];
  for (const c of cases) {
    const di = detectInputFor(c.text, rules);
    if (letterCount(di) === 0) {
      skippedUntranslatable.push(c.id);
      continue;
    }
    const det = await detector.detect(di);
    prepared.push({ c, di, det, uiLang: uiLangModel(c) });
  }

  const cells: PolicyCellResult[] = [];
  for (const shortTextLetters of thresholds) {
    for (const policy of policies) {
      const cell: PolicyCellResult = {
        shortTextLetters,
        policy,
        n: 0,
        correct: 0,
        shortDisagree: 0,
        shortDisagreeDetectorRight: 0,
        shortDisagreePriorRight: 0,
      };

      for (const { c, di, det, uiLang } of prepared) {
        const resolveOpts: ResolveSourceOptions = {
          uiLang,
          shortTextLetters,
          minScore: opts.minScore,
          shortTextPolicy: policy,
        };
        const r = await resolveSource(di, detector, resolveOpts);
        cell.n++;
        if (sameLang(r.lang, c.sourceLang)) cell.correct++;

        // The policy-relevant zone, defined to match resolveSource's `usable`:
        // short + reliable (+ score floor) + detection disagrees with the prior.
        const reliable =
          det.lang != null &&
          det.reliable &&
          (opts.minScore === undefined || (det.score ?? 0) >= opts.minScore);
        const short = letterCount(di) < shortTextLetters;
        if (reliable && short && !sameLang(det.lang!, uiLang)) {
          cell.shortDisagree++;
          if (sameLang(det.lang!, c.sourceLang)) cell.shortDisagreeDetectorRight++;
          if (sameLang(uiLang, c.sourceLang)) cell.shortDisagreePriorRight++;
        }
      }

      cells.push(cell);
    }
  }

  return {
    cells,
    uiLangModelName: opts.uiLangModelName ?? 'native (uiLang = gold source)',
    skippedUntranslatable,
  };
}

export function formatPolicyMarkdown(r: PolicyReport): string {
  const cols = [
    'shortLetters',
    'policy',
    'n',
    'accuracy%',
    'short-disagree n',
    'det right',
    'prior right',
  ];
  const line = (cells: string[]) => `| ${cells.join(' | ')} |`;
  const rows = r.cells.map((c) => [
    String(c.shortTextLetters),
    c.policy,
    String(c.n),
    pct(c.correct, c.n),
    String(c.shortDisagree),
    String(c.shortDisagreeDetectorRight),
    String(c.shortDisagreePriorRight),
  ]);
  return [
    `### Short-text policy sweep — uiLang model: ${r.uiLangModelName}`,
    line(cols),
    line(cols.map(() => '---')),
    ...rows.map(line),
    '',
    '_In the short-disagree zone the policy is the only thing that moves accuracy._',
    '_`det right` : `prior right` is the loanword-vs-code-switch ratio — independent of_',
    '_how often that zone occurs in your traffic. Under a native uiLang model `prior right`_',
    '_is high by construction; run `linguaFrancaUi(...)` to surface the code-switch side._',
    `_skipped (nothing translatable): ${r.skippedUntranslatable.join(', ') || 'none'}_`,
  ].join('\n');
}

// ─────────────────────────────────────────────────────────────
// 4. Combined runner
// ─────────────────────────────────────────────────────────────

export interface SourceResolutionReport {
  calibration: CalibrationReport;
  policy: PolicyReport;
}

export async function runSourceResolutionEval(
  cases: SourceEvalCase[],
  detector: ConfidenceDetector,
  opts: SweepOptions & { binLowerBounds?: number[] } = {},
): Promise<SourceResolutionReport> {
  const calibration = await calibrateDetector(cases, detector, opts.binLowerBounds, opts.rules);
  const policy = await sweepPolicies(cases, detector, opts);
  return { calibration, policy };
}

export function formatReportMarkdown(r: SourceResolutionReport): string {
  return [formatCalibrationMarkdown(r.calibration), '', formatPolicyMarkdown(r.policy)].join('\n');
}

/* ─────────────────────────────────────────────────────────────
 * Usage:
 *
 *   import { eld } from 'eld/large';
 *   import { eldDetector } from './resolve-source';
 *   import {
 *     runSourceResolutionEval, formatReportMarkdown, linguaFrancaUi,
 *   } from './source-resolution-eval';
 *   import { readFileSync } from 'node:fs';
 *
 *   const cases = readFileSync('sample-chat-eval.jsonl', 'utf8')
 *     .trim().split('\n').map((l) => JSON.parse(l));
 *
 *   const EUROPEAN = ['en','de','fr','es','it','pt','nl'];
 *   const detector = eldDetector(eld, EUROPEAN);
 *
 *   // Native assumption — shows how often the prior rescues a wrong short detection:
 *   console.log(formatReportMarkdown(await runSourceResolutionEval(cases, detector)));
 *
 *   // Code-switch stress — shows where trusting the detector would win instead:
 *   console.log(formatReportMarkdown(await runSourceResolutionEval(cases, detector, {
 *     uiLangModel: linguaFrancaUi('en'),
 *     uiLangModelName: 'lingua franca = en',
 *   })));
 * ───────────────────────────────────────────────────────────── */
