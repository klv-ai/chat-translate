/**
 * translation-eval.ts
 *
 * An eval harness that runs several translation backends side by side on a
 * sample of REAL chat traffic, so the build-vs-buy (and which-quant) decision
 * rests on measured numbers rather than vibes.
 *
 * It is deliberately built around three honest tiers of signal:
 *
 *   1. EXACT, zero-reference, deterministic — always run:
 *        • placeholder survival: did the masked @mentions / URLs / emoji / code
 *          come back intact? This is the do-not-translate fidelity that breaks
 *          chat, and it needs no gold reference. It is also precisely where we
 *          expect Q4 → Q8 → DeepL to separate: NMT preserves sentinels almost
 *          perfectly, a raw LLM less so, and a more aggressively quantised LLM
 *          least of all.
 *        • error rate, empty-output rate, latency (p50/p95), throughput, cost.
 *
 *   2. chrF against a human reference — run when the dataset has one. Character
 *      n-gram F-score, good for short, morphologically rich European text.
 *      (From-scratch implementation; for publication-grade numbers use
 *      sacreBLEU. Treat these as relative, candidate-vs-candidate signal.)
 *
 *   3. LLM-as-judge — optional, injected. Scores adequacy / fluency / REGISTER
 *      (the tu/vous, du/Sie axis where DeepL's formality control and the LLM's
 *      best-effort prompt visibly diverge). Reference-free. See buildJudgePrompt.
 *
 * For the final call between your top two candidates, a small human eval still
 * beats all of the above — these metrics narrow the field, they don't crown a
 * winner.
 *
 * Methodology choices worth knowing:
 *   • We MASK then translate the masked form then restore — i.e. we measure the
 *     real production pipeline, which is what makes placeholder survival
 *     meaningful.
 *   • We call the provider DIRECTLY, bypassing the fan-out cache, so latency is
 *     cold provider latency, not a cache hit.
 *   • Candidates run sequentially (not interleaved) so two local backends
 *     sharing one GPU don't contend and skew each other's latency.
 *   • Supply a gold `sourceLang` per row: it keeps the comparison apples-to-
 *     apples (every candidate translates the same declared direction) and is
 *     required for the local backends, which cannot auto-detect.
 *
 * Runtime: Node 18+ (global fetch). No npm dependencies.
 */

import {
  DEFAULT_RULES,
  type LanguageDetector,
  maskNonTranslatable,
} from './chat-translation-layer';
import {
  BaseTranslationProvider,
  type Formality,
  type LanguageCode,
  type ProviderCapabilities,
  type TranslateOptions,
  TranslationError,
  type TranslationProvider,
} from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. A llama.cpp / GGUF backend (Q8, BF16, …) for the eval
// ─────────────────────────────────────────────────────────────

/**
 * Ollama only surfaces Q4_K_M of TranslateGemma. If you intend to ship the
 * self-hosted path you would most likely run a higher-fidelity GGUF (Q8_0, or
 * BF16) under llama.cpp's `llama-server`, which exposes an OpenAI-compatible
 * /v1/chat/completions endpoint. This adapter lets the eval test the quant you
 * would actually deploy. Promote it into translation-provider.ts if you ship.
 */
export interface LlamaCppConfig {
  /** llama-server host. Default http://localhost:8080 */
  host?: string;
  /** Served model name/alias (informational under OpenAI-compat). */
  model?: string;
  /** Metadata only, surfaced in the report label, e.g. "Q8_0", "BF16". */
  quantLabel?: string;
}

/** Shared with the Ollama provider's intent — align with the model card. */
function buildGemmaPrompt(text: string, o: TranslateOptions): string {
  const register =
    o.formality === 'more' || o.formality === 'prefer_more'
      ? ' Use a formal register.'
      : o.formality === 'less' || o.formality === 'prefer_less'
        ? ' Use an informal register.'
        : '';
  const ctx = o.context
    ? `\nConversation context (for disambiguation only, do not translate):\n${o.context}\n`
    : '';
  return (
    `Translate the following text from ${o.sourceLang} to ${o.targetLang}.` +
    `${register} Output only the translation, with no preamble or quotes.` +
    `${ctx}\n\n${text}`
  );
}

export class LlamaCppTranslateGemmaProvider extends BaseTranslationProvider {
  readonly name: string;
  private readonly host: string;
  private readonly model: string;

  constructor(config: LlamaCppConfig = {}) {
    super();
    this.host = config.host ?? 'http://localhost:8080';
    this.model = config.model ?? 'translategemma';
    this.name = `translategemma-gguf${config.quantLabel ? `-${config.quantLabel}` : ''}`;
  }

  capabilities(): ProviderCapabilities {
    // Same honest capabilities as the Ollama path — runtime/quant doesn't add
    // first-class features the layer can rely on.
    return {
      autoDetectSource: false,
      formality: false,
      glossaries: false,
      contextHint: true,
      nativeBatch: false,
    };
  }

  async translate(text: string, options: TranslateOptions) {
    if (!options.sourceLang) {
      throw new TranslationError(
        `${this.name} requires an explicit sourceLang (no auto-detect)`,
        this.name,
      );
    }
    const prompt = buildGemmaPrompt(text, options);
    let res: Response;
    try {
      res = await fetch(`${this.host}/v1/chat/completions`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.model,
          stream: false,
          temperature: 0,
          messages: [{ role: 'user', content: prompt }],
        }),
      });
    } catch (cause) {
      throw new TranslationError('llama-server request failed', this.name, cause, true);
    }
    if (!res.ok) {
      throw new TranslationError(
        `llama-server returned ${res.status}`,
        this.name,
        undefined,
        res.status >= 500,
      );
    }
    const json = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
    };
    const out = json.choices?.[0]?.message?.content?.trim();
    if (!out) throw new TranslationError('Empty translation from model', this.name);
    return { text: out, detectedSourceLang: options.sourceLang };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.host}/health`);
      return res.ok;
    } catch {
      return false;
    }
  }
}

// ─────────────────────────────────────────────────────────────
// 2. Eval types
// ─────────────────────────────────────────────────────────────

export interface EvalCase {
  id: string;
  text: string;
  /** Gold source language. Strongly recommended; required for local backends. */
  sourceLang?: LanguageCode;
  targetLang: LanguageCode;
  /** Human translation, for chrF. Optional. */
  reference?: string;
  /** Prior messages for disambiguation — passed through, never translated. */
  context?: string;
}

export interface Candidate {
  /** Short label for the report, e.g. "deepl", "tg-q4-ollama", "tg-q8-gguf". */
  label: string;
  provider: TranslationProvider;
  /** API $ per 1M source characters (DeepL etc.). Omit for self-hosted. */
  costPerMillionChars?: number;
  /** Per-candidate parallelism. Default caps.nativeBatch ? 8 : 1. */
  maxConcurrency?: number;
  notes?: string;
}

export interface JudgeScore {
  /** 1–5: does the translation preserve the source meaning? */
  adequacy: number;
  /** 1–5: is the target text natural and grammatical? */
  fluency: number;
  /** 1–5: is the register/formality appropriate for casual chat? */
  register: number;
  notes?: string;
}

export type QualityJudge = (args: {
  source: string;
  sourceLang?: LanguageCode;
  target: LanguageCode;
  candidateOutput: string;
  reference?: string;
}) => Promise<JudgeScore>;

export interface CaseResult {
  caseId: string;
  candidate: string;
  targetLang: LanguageCode;
  output?: string;
  error?: string;
  latencyMs: number;
  inputChars: number;
  maskedTokens: number;
  /** 0..1; 1 when there were no tokens to preserve. */
  placeholderSurvival: number;
  chrf?: number;
  judge?: JudgeScore;
}

export interface CandidateAggregate {
  label: string;
  notes?: string;
  cases: number;
  errors: number;
  errorRate: number;
  meanPlaceholderSurvival: number;
  /** Of cases that HAD tokens, the fraction that preserved every one. */
  perfectPlaceholderRate: number;
  meanChrf?: number;
  meanJudge?: { adequacy: number; fluency: number; register: number };
  latencyP50: number;
  latencyP95: number;
  totalInputChars: number;
  /** Wall-clock chars/sec for this candidate (capacity signal for local). */
  throughputCharsPerSec: number;
  /** Only for candidates with costPerMillionChars set. */
  estCostUsd?: number;
}

export interface EvalReport {
  candidates: CandidateAggregate[];
  perCase: CaseResult[];
  generatedAt: string;
}

export interface RunEvalOptions {
  formality?: Formality;
  /** Fills missing sourceLang when present; otherwise local backends error. */
  detector?: LanguageDetector;
  judge?: QualityJudge;
  rules?: Parameters<typeof maskNonTranslatable>[1];
}

// ─────────────────────────────────────────────────────────────
// 3. Metrics
// ─────────────────────────────────────────────────────────────

function charNgrams(s: string, n: number): Map<string, number> {
  const grams = new Map<string, number>();
  for (let i = 0; i + n <= s.length; i++) {
    const g = s.slice(i, i + n);
    grams.set(g, (grams.get(g) ?? 0) + 1);
  }
  return grams;
}

/** Approximate chrF (β=2, n≤6, spaces included). Relative signal, not sacreBLEU. */
export function chrF(hyp: string, ref: string, maxN = 6, beta = 2): number {
  if (!hyp || !ref) return 0;
  const b2 = beta * beta;
  let fSum = 0;
  let valid = 0;
  for (let n = 1; n <= maxN; n++) {
    const h = charNgrams(hyp, n);
    const r = charNgrams(ref, n);
    if (h.size === 0 || r.size === 0) continue;
    let match = 0;
    let hTotal = 0;
    let rTotal = 0;
    for (const [g, c] of h) {
      hTotal += c;
      match += Math.min(c, r.get(g) ?? 0);
    }
    for (const [, c] of r) rTotal += c;
    const p = hTotal ? match / hTotal : 0;
    const rec = rTotal ? match / rTotal : 0;
    valid++;
    if (p + rec > 0) fSum += ((1 + b2) * p * rec) / (b2 * p + rec);
  }
  return valid ? (fSum / valid) * 100 : 0;
}

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.floor((p / 100) * sorted.length));
  return sorted[idx];
}

async function runBounded(tasks: (() => Promise<void>)[], limit: number): Promise<void> {
  let i = 0;
  const lanes = Math.max(1, Math.min(limit, tasks.length));
  await Promise.all(
    Array.from({ length: lanes }, async () => {
      while (i < tasks.length) await tasks[i++]();
    }),
  );
}

// ─────────────────────────────────────────────────────────────
// 4. The runner
// ─────────────────────────────────────────────────────────────

/**
 * The judge prompt. Pipe the returned string to your frontier model (under the
 * same no-retain MSA) and parse the JSON it returns. Kept as a string so the
 * harness has zero model dependencies and you stay in control of the endpoint.
 */
export function buildJudgePrompt(args: {
  source: string;
  sourceLang?: LanguageCode;
  target: LanguageCode;
  candidateOutput: string;
}): string {
  return [
    'You are grading a chat-message translation. The original is casual instant-message text.',
    `Source${args.sourceLang ? ` (${args.sourceLang})` : ''}: ${args.source}`,
    `Translation (${args.target}): ${args.candidateOutput}`,
    '',
    'Score each axis 1–5 (5 best):',
    '- adequacy: is the source meaning preserved?',
    '- fluency: is the translation natural and grammatical for a native speaker?',
    '- register: is the formality right for a casual chat (not stiff, not sloppy)?',
    '',
    'Respond with ONLY a JSON object: {"adequacy":n,"fluency":n,"register":n,"notes":"…"}.',
  ].join('\n');
}

export async function runEval(
  cases: EvalCase[],
  candidates: Candidate[],
  options: RunEvalOptions = {},
): Promise<EvalReport> {
  const rules = options.rules ?? DEFAULT_RULES;
  const perCase: CaseResult[] = [];
  const aggregates: CandidateAggregate[] = [];

  // Candidates run one at a time so co-located GPU backends don't contend.
  for (const cand of candidates) {
    const caps = cand.provider.capabilities();
    const concurrency = cand.maxConcurrency ?? (caps.nativeBatch ? 8 : 1);
    const formality: Formality | undefined = caps.formality
      ? (options.formality ?? 'prefer_less')
      : undefined;

    const results: CaseResult[] = [];
    const candStart = Date.now();

    const tasks = cases.map((c) => async () => {
      const { masked, restore, findUnrestored, tokenCount } = maskNonTranslatable(c.text, rules);

      // Resolve source: gold label first, then detector, else null (auto).
      let sourceLang: LanguageCode | null = c.sourceLang ?? null;
      if (!sourceLang && options.detector) {
        sourceLang = await options.detector.detect(masked.replace(/\uE000\d+\uE001/g, ' '));
      }

      const opts: TranslateOptions = {
        sourceLang,
        targetLang: c.targetLang,
        formality,
        context: c.context,
      };

      const t0 = Date.now();
      let output: string | undefined;
      let error: string | undefined;
      let survival = 1;
      let chrf: number | undefined;
      let judge: JudgeScore | undefined;

      try {
        const res = await cand.provider.translate(masked, opts);
        const unrestored = findUnrestored(res.text);
        survival = tokenCount === 0 ? 1 : (tokenCount - unrestored.length) / tokenCount;
        output = restore(res.text);
        if (c.reference) chrf = chrF(output, c.reference);
        if (options.judge && output) {
          judge = await options.judge({
            source: c.text,
            sourceLang: sourceLang ?? undefined,
            target: c.targetLang,
            candidateOutput: output,
            reference: c.reference,
          });
        }
      } catch (e) {
        error = e instanceof Error ? e.message : String(e);
      }

      results.push({
        caseId: c.id,
        candidate: cand.label,
        targetLang: c.targetLang,
        output,
        error,
        latencyMs: Date.now() - t0,
        inputChars: c.text.length,
        maskedTokens: tokenCount,
        placeholderSurvival: survival,
        chrf,
        judge,
      });
    });

    await runBounded(tasks, concurrency);
    const wallSec = Math.max(0.001, (Date.now() - candStart) / 1000);

    // Aggregate.
    const ok = results.filter((r) => !r.error);
    const withTokens = ok.filter((r) => r.maskedTokens > 0);
    const chrfs = ok.map((r) => r.chrf).filter((x): x is number => x !== undefined);
    const judged = ok.map((r) => r.judge).filter((x): x is JudgeScore => x !== undefined);
    const latencies = results.map((r) => r.latencyMs);
    const totalInputChars = results.reduce((s, r) => s + r.inputChars, 0);
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

    aggregates.push({
      label: cand.label,
      notes: cand.notes,
      cases: results.length,
      errors: results.length - ok.length,
      errorRate: results.length ? (results.length - ok.length) / results.length : 0,
      meanPlaceholderSurvival: mean(ok.map((r) => r.placeholderSurvival)),
      perfectPlaceholderRate: withTokens.length
        ? withTokens.filter((r) => r.placeholderSurvival === 1).length / withTokens.length
        : 1,
      meanChrf: chrfs.length ? mean(chrfs) : undefined,
      meanJudge: judged.length
        ? {
            adequacy: mean(judged.map((j) => j.adequacy)),
            fluency: mean(judged.map((j) => j.fluency)),
            register: mean(judged.map((j) => j.register)),
          }
        : undefined,
      latencyP50: percentile(latencies, 50),
      latencyP95: percentile(latencies, 95),
      totalInputChars,
      throughputCharsPerSec: totalInputChars / wallSec,
      estCostUsd:
        cand.costPerMillionChars !== undefined
          ? (totalInputChars / 1_000_000) * cand.costPerMillionChars
          : undefined,
    });

    perCase.push(...results);
  }

  return { candidates: aggregates, perCase, generatedAt: new Date().toISOString() };
}

// ─────────────────────────────────────────────────────────────
// 5. Reporting
// ─────────────────────────────────────────────────────────────

export function formatReportMarkdown(report: EvalReport): string {
  const cols = [
    'candidate',
    'cases',
    'err%',
    'token-keep%',
    'perfect%',
    'chrF',
    'judge A/F/R',
    'p50 ms',
    'p95 ms',
    'chars/s',
    'est $',
  ];
  const rows = report.candidates.map((c) => [
    c.notes ? `${c.label} (${c.notes})` : c.label,
    String(c.cases),
    (c.errorRate * 100).toFixed(1),
    (c.meanPlaceholderSurvival * 100).toFixed(1),
    (c.perfectPlaceholderRate * 100).toFixed(1),
    c.meanChrf !== undefined ? c.meanChrf.toFixed(1) : '—',
    c.meanJudge
      ? `${c.meanJudge.adequacy.toFixed(1)}/${c.meanJudge.fluency.toFixed(1)}/${c.meanJudge.register.toFixed(1)}`
      : '—',
    c.latencyP50.toFixed(0),
    c.latencyP95.toFixed(0),
    c.throughputCharsPerSec.toFixed(0),
    c.estCostUsd !== undefined ? `$${c.estCostUsd.toFixed(4)}` : 'infra',
  ]);
  const line = (cells: string[]) => `| ${cells.join(' | ')} |`;
  return [
    line(cols),
    line(cols.map(() => '---')),
    ...rows.map(line),
    '',
    '_token-keep% = mean placeholder survival; perfect% = cases preserving every token;_',
    '_chrF is relative signal (from-scratch, not sacreBLEU); "infra" = self-hosted, amortise GPU separately._',
  ].join('\n');
}

/* ─────────────────────────────────────────────────────────────
 * Usage:
 *
 *   import { DeepLProvider, TranslateGemmaProvider } from './translation-provider';
 *   import {
 *     runEval, formatReportMarkdown, LlamaCppTranslateGemmaProvider,
 *   } from './translation-eval';
 *   import { readFileSync } from 'node:fs';
 *
 *   const cases = readFileSync('sample-chat-eval.jsonl', 'utf8')
 *     .trim().split('\n').map((l) => JSON.parse(l));
 *
 *   const report = await runEval(cases, [
 *     { label: 'deepl', provider: new DeepLProvider({ apiKey: process.env.DEEPL_API_KEY! }),
 *       costPerMillionChars: 25, notes: 'Pro' },
 *     { label: 'tg-q4', provider: new TranslateGemmaProvider({ model: 'translategemma:4b' }),
 *       notes: 'Q4_K_M via Ollama' },
 *     { label: 'tg-q8', provider: new LlamaCppTranslateGemmaProvider({ host: 'http://localhost:8080', quantLabel: 'Q8_0' }),
 *       notes: 'Q8_0 GGUF via llama.cpp' },
 *   ], {
 *     formality: 'prefer_less',
 *     // judge: async (a) => parseJson(await myFrontierModel(buildJudgePrompt(a))),
 *   });
 *
 *   console.log(formatReportMarkdown(report));
 *   // writeFileSync('eval-report.json', JSON.stringify(report, null, 2));
 * ───────────────────────────────────────────────────────────── */
