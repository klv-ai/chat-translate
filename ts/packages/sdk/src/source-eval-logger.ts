/**
 * source-eval-logger.ts
 *
 * Live-traffic capture for the source-resolution eval loop. Emits two
 * append-only event types into a dedicated Postgres store; the eval
 * (source-resolution-eval.ts) joins them by messageId into SourceEvalCase.
 *
 * Why two events, not one mutable row (see SOURCE-RESOLUTION-SUMMARY open Qs):
 *   - The prediction (resolution) and the gold label (correction) arrive at
 *     different times — a replay correction can land days later.
 *   - The prediction must be FROZEN. The serving message row gets overwritten
 *     to user_confirmed on correction, which is right for history but destroys
 *     what the resolver decided BEFORE the human overruled it — exactly what
 *     the eval grades. So capture lives in its own store with append-only
 *     semantics, never the mutable serving row.
 *
 * Three load-bearing properties:
 *   1. OFF THE HOT PATH, FAIL-OPEN. Recording is fire-and-forget into a
 *      bounded buffer. If the sink is down we drop-with-a-counter — we never
 *      block or fail a chat post to capture an eval sample. Losing samples is
 *      acceptable; losing a message is not.
 *   2. SAMPLE BY INTERESTINGNESS. Confident long-text detections are abundant
 *      and low-signal; short / not-confident / corrected rows are the whole
 *      point. Keep 100% of the latter, downsample the former, and store the
 *      per-row rate so the eval can reweight.
 *   3. CONFIG-STAMPED. Every decision records the thresholds/policy/detector
 *      that produced it. Without this, tuning a knob makes historical rows
 *      unreproducible and you can't tell a data change from a config change.
 *
 * Runtime: Node 18+. No npm deps in the core; PgEventSink expects a `pg` Pool
 * injected (kept at the edge so the buffer/logger stay dependency-free).
 */

import type {
  ResolvedSource,
  ResolveSourceOptions,
  ShortTextPolicy,
  SourceProvenance,
} from './resolve-source';
import { DEFAULT_SHORT_TEXT_LETTERS, DEFAULT_SHORT_TEXT_POLICY } from './resolve-source';
import type { LanguageCode } from './translation-provider';

// ─────────────────────────────────────────────────────────────
// 1. Event shapes
// ─────────────────────────────────────────────────────────────

/**
 * The exact configuration that produced a resolution — what makes a captured
 * decision reproducible after a knob moves. Build it with `configStampFrom`
 * (below) from the SAME options passed to resolveSource, so it records the
 * EFFECTIVE config rather than a hand-copied literal that can silently drift.
 */
export interface ResolverConfigStamp {
  shortTextLetters: number;
  shortTextPolicy: ShortTextPolicy;
  minScore?: number;
  /** Detector identity + version, e.g. "eld@1.4.0". */
  detectorId: string;
  /** ELD DB size if applicable: "xs" | "s" | "m" | "l". */
  detectorDb?: string;
}

/** The frozen prediction, captured at ingest. */
export interface SourceResolutionEvent {
  // — join keys —
  messageId: string;
  /** Message post time, epoch ms. Also the partition key. */
  at: number;

  // — inputs the resolver saw —
  /** Masked detectInput (sentinels → spaces): same string the resolver got. */
  maskedDetectInput: string;
  uiLang: LanguageCode;
  letterCount: number;

  // — outputs (mirror ResolvedSource) —
  lang: LanguageCode;
  provenance: SourceProvenance;
  confident: boolean;
  detected?: LanguageCode;
  score?: number;

  // — reproducibility + reweighting —
  config: ResolverConfigStamp;
  /** Keep-probability this row was sampled at, in [0,1]. 1 = always kept. */
  sampleRate: number;
}

/** The gold label, captured whenever a replay correction posts. */
export interface SourceCorrectionEvent {
  messageId: string;
  /** Correction time, epoch ms. */
  at: number;
  correctedLang: LanguageCode;
}

// ─────────────────────────────────────────────────────────────
// 2. The logger contract (what the ingest path calls)
// ─────────────────────────────────────────────────────────────

/**
 * Both methods are void and fire-and-forget: the caller NEVER awaits them.
 * That is the contract — it's how recording stays off the hot path.
 */
export interface SourceEvalLogger {
  recordResolution(event: SourceResolutionEvent): void;
  recordCorrection(event: SourceCorrectionEvent): void;
  /** Snapshot for observability — same instinct as RoomStats. */
  stats(): EvalLoggerStats;
  /** Flush + stop timers. Call on graceful shutdown so the last batch lands. */
  close(): Promise<void>;
}

export interface EvalLoggerStats {
  bufferedResolutions: number;
  bufferedCorrections: number;
  /** Sampled OUT before buffering (by design — not a problem). */
  sampledOut: number;
  /** Dropped because the buffer was full (capacity pressure — watch this). */
  droppedFull: number;
  /** Dropped because a flush failed (sink down — watch this). */
  droppedFlushError: number;
  flushes: number;
}

// ─────────────────────────────────────────────────────────────
// 3. The sink contract (what talks to storage)
// ─────────────────────────────────────────────────────────────

/**
 * Injectable so the buffer stays storage-agnostic — Postgres below, but a
 * queue (Kafka/SQS) → durable store satisfies it too for multi-instance.
 * Both writes are batch; the sink should upsert/ignore-on-conflict so a
 * retried flush is idempotent on (messageId, at).
 */
export interface EventSink {
  writeResolutions(rows: SourceResolutionEvent[]): Promise<void>;
  writeCorrections(rows: SourceCorrectionEvent[]): Promise<void>;
}

// ─────────────────────────────────────────────────────────────
// 4. Sampler — decides keep-probability for a resolution
// ─────────────────────────────────────────────────────────────

/** Given the resolution (sans sampleRate), return keep-probability in [0,1]. */
export type ResolutionSampler = (e: Omit<SourceResolutionEvent, 'sampleRate'>) => number;

/**
 * Default policy: keep everything interesting, thin only the easy bulk.
 *   - not confident         → keep all (the revisitable backlog)
 *   - short-text zone       → keep all (where the resolver is hardest tested)
 *   - confident long detect  → 5% (abundant, low signal)
 *   - everything else        → keep all (overrides, etc. — rare, label-bearing)
 * Corrections are NEVER sampled — they're the gold labels, always recorded.
 */
export function defaultResolutionSampler(easyKeepRate = 0.05): ResolutionSampler {
  return (e) => {
    const isEasyBulk =
      e.confident && e.provenance === 'detected' && e.letterCount >= e.config.shortTextLetters;
    return isEasyBulk ? easyKeepRate : 1;
  };
}

// ─────────────────────────────────────────────────────────────
// 5. Buffering logger — the actual fire-and-forget machinery
// ─────────────────────────────────────────────────────────────

export interface BufferingLoggerOptions {
  sink: EventSink;
  /** Keep-probability policy. Default: defaultResolutionSampler(). */
  sampler?: ResolutionSampler;
  /** Flush when either buffer reaches this many rows. Default 500. */
  flushAt?: number;
  /** Periodic flush, ms. Default 5_000. */
  flushIntervalMs?: number;
  /** Hard cap per buffer; rows past this are dropped-with-a-counter. Default 50_000. */
  maxBuffer?: number;
  /** Optional [0,1) source for sampling; injectable for deterministic tests. */
  random?: () => number;
}

export class BufferingEvalLogger implements SourceEvalLogger {
  private res: SourceResolutionEvent[] = [];
  private corr: SourceCorrectionEvent[] = [];
  private readonly sink: EventSink;
  private readonly sampler: ResolutionSampler;
  private readonly flushAt: number;
  private readonly maxBuffer: number;
  private readonly random: () => number;
  private readonly timer: ReturnType<typeof setInterval>;
  private flushing = false;
  private s: EvalLoggerStats = {
    bufferedResolutions: 0,
    bufferedCorrections: 0,
    sampledOut: 0,
    droppedFull: 0,
    droppedFlushError: 0,
    flushes: 0,
  };

  constructor(opts: BufferingLoggerOptions) {
    this.sink = opts.sink;
    this.sampler = opts.sampler ?? defaultResolutionSampler();
    this.flushAt = opts.flushAt ?? 500;
    this.maxBuffer = opts.maxBuffer ?? 50_000;
    this.random = opts.random ?? Math.random;
    this.timer = setInterval(() => void this.flush(), opts.flushIntervalMs ?? 5_000);
    // Don't keep the process alive just for the flush timer.
    (this.timer as { unref?: () => void }).unref?.();
  }

  recordResolution(event: SourceResolutionEvent): void {
    // Sample first — the dice decide, and we stamp the rate we rolled at so
    // the eval can reweight thinned classes back up.
    const rate = this.sampler(event);
    if (rate < 1 && this.random() >= rate) {
      this.s.sampledOut++;
      return;
    }
    if (this.res.length >= this.maxBuffer) {
      this.s.droppedFull++;
      return;
    }
    this.res.push({ ...event, sampleRate: rate });
    if (this.res.length >= this.flushAt) void this.flush();
  }

  recordCorrection(event: SourceCorrectionEvent): void {
    // Never sampled — gold labels are always recorded.
    if (this.corr.length >= this.maxBuffer) {
      this.s.droppedFull++;
      return;
    }
    this.corr.push(event);
    if (this.corr.length >= this.flushAt) void this.flush();
  }

  stats(): EvalLoggerStats {
    return {
      ...this.s,
      bufferedResolutions: this.res.length,
      bufferedCorrections: this.corr.length,
    };
  }

  async close(): Promise<void> {
    clearInterval(this.timer);
    await this.flush();
  }

  /**
   * Drain both buffers into the sink. Overlapping flushes are guarded. On sink
   * error we DROP the batch and count it (fail-open): re-queueing risks
   * unbounded growth and we've decided sample loss is acceptable. If you later
   * need at-least-once, bound a retry queue here — but never let it back-
   * pressure recordResolution / recordCorrection.
   */
  private async flush(): Promise<void> {
    if (this.flushing) return;
    if (this.res.length === 0 && this.corr.length === 0) return;
    this.flushing = true;
    const res = this.res;
    this.res = [];
    const corr = this.corr;
    this.corr = [];
    try {
      if (res.length) await this.sink.writeResolutions(res);
      if (corr.length) await this.sink.writeCorrections(corr);
      this.s.flushes++;
    } catch {
      this.s.droppedFlushError += res.length + corr.length;
    } finally {
      this.flushing = false;
    }
  }
}

/** A no-op logger for the DeepL deployment or local dev — capture turned off. */
export const NoopEvalLogger: SourceEvalLogger = {
  recordResolution() {},
  recordCorrection() {},
  stats: () => ({
    bufferedResolutions: 0,
    bufferedCorrections: 0,
    sampledOut: 0,
    droppedFull: 0,
    droppedFlushError: 0,
    flushes: 0,
  }),
  close: async () => {},
};

// ─────────────────────────────────────────────────────────────
// 6. Ergonomic builders — turn resolver inputs/output into events
// ─────────────────────────────────────────────────────────────

/**
 * Build the config stamp from the SAME options object passed to resolveSource,
 * applying resolve-source's own exported defaults. This is the anti-drift move:
 * the stamp records the EFFECTIVE config, and if a default changes in
 * resolve-source.ts the stamp follows automatically — no hand-copied literal to
 * forget. Detector identity isn't in the resolver options (it's the injected
 * detector instance), so it's passed alongside.
 */
export function configStampFrom(
  opts: Pick<ResolveSourceOptions, 'shortTextLetters' | 'shortTextPolicy' | 'minScore'>,
  detector: { id: string; db?: string },
): ResolverConfigStamp {
  return {
    shortTextLetters: opts.shortTextLetters ?? DEFAULT_SHORT_TEXT_LETTERS,
    shortTextPolicy: opts.shortTextPolicy ?? DEFAULT_SHORT_TEXT_POLICY,
    minScore: opts.minScore,
    detectorId: detector.id,
    detectorDb: detector.db,
  };
}

/**
 * Build the resolution event from what the ingest path already has in hand:
 * the resolver's output plus the inputs it saw. `sampleRate` is filled by the
 * logger, so it's omitted here.
 */
export function resolutionEventFrom(args: {
  messageId: string;
  at?: number;
  maskedDetectInput: string;
  uiLang: LanguageCode;
  letterCount: number;
  resolved: ResolvedSource;
  config: ResolverConfigStamp;
  score?: number;
}): Omit<SourceResolutionEvent, 'sampleRate'> {
  return {
    messageId: args.messageId,
    at: args.at ?? Date.now(),
    maskedDetectInput: args.maskedDetectInput,
    uiLang: args.uiLang,
    letterCount: args.letterCount,
    lang: args.resolved.lang,
    provenance: args.resolved.provenance,
    confident: args.resolved.confident,
    detected: args.resolved.detected,
    score: args.score,
    config: args.config,
  };
}

// ─────────────────────────────────────────────────────────────
// 7. Postgres sink (sketch) + DDL
// ─────────────────────────────────────────────────────────────

/**
 * Own schema, two append-only fact tables, range-partitioned by month so
 * retention/archival is a partition drop and the eval just reads a window.
 * Late corrections attach to OLD message ids in OLD partitions — that's fine
 * here (the table is the truth); it only matters for the Cube refresh story.
 *
 *   create schema if not exists source_eval;
 *
 *   create table if not exists source_eval.resolution (
 *     message_id           text        not null,
 *     at                   timestamptz not null,
 *     masked_detect_input  text        not null,
 *     ui_lang              text        not null,
 *     letter_count         int         not null,
 *     lang                 text        not null,
 *     provenance           text        not null,
 *     confident            boolean     not null,
 *     detected             text,
 *     score                real,
 *     cfg_short_text_letters int       not null,
 *     cfg_short_text_policy  text      not null,
 *     cfg_min_score          real,
 *     cfg_detector_id        text      not null,
 *     cfg_detector_db        text,
 *     sample_rate          real        not null,
 *     primary key (message_id, at)
 *   ) partition by range (at);
 *
 *   create table if not exists source_eval.correction (
 *     message_id     text        not null,
 *     at             timestamptz not null,
 *     corrected_lang text        not null,
 *     primary key (message_id, at)
 *   ) partition by range (at);
 *
 *   create index on source_eval.resolution (message_id);
 *   create index on source_eval.correction (message_id);
 *   -- monthly partitions: pg_partman, or pre-create per month.
 */

type PoolLike = {
  query(text: string, values?: unknown[]): Promise<unknown>;
};

export class PgEventSink implements EventSink {
  constructor(
    private readonly pool: PoolLike,
    private readonly schema = 'source_eval',
  ) {}

  async writeResolutions(rows: SourceResolutionEvent[]): Promise<void> {
    if (!rows.length) return;
    const cols = 15;
    const values: unknown[] = [];
    const tuples = rows.map((r, i) => {
      const b = i * cols;
      values.push(
        r.messageId,
        new Date(r.at),
        r.maskedDetectInput,
        r.uiLang,
        r.letterCount,
        r.lang,
        r.provenance,
        r.confident,
        r.detected ?? null,
        r.score ?? null,
        r.config.shortTextLetters,
        r.config.shortTextPolicy,
        r.config.minScore ?? null,
        r.config.detectorId,
        r.sampleRate,
      );
      return `(${Array.from({ length: cols }, (_, k) => `$${b + k + 1}`).join(',')})`;
    });
    await this.pool.query(
      `insert into ${this.schema}.resolution
       (message_id, at, masked_detect_input, ui_lang, letter_count,
        lang, provenance, confident, detected, score,
        cfg_short_text_letters, cfg_short_text_policy, cfg_min_score,
        cfg_detector_id, sample_rate)
       values ${tuples.join(',')}
       on conflict (message_id, at) do nothing`,
      values,
    );
  }

  async writeCorrections(rows: SourceCorrectionEvent[]): Promise<void> {
    if (!rows.length) return;
    const values: unknown[] = [];
    const tuples = rows.map((r, i) => {
      const b = i * 3;
      values.push(r.messageId, new Date(r.at), r.correctedLang);
      return `($${b + 1},$${b + 2},$${b + 3})`;
    });
    await this.pool.query(
      `insert into ${this.schema}.correction (message_id, at, corrected_lang)
       values ${tuples.join(',')}
       on conflict (message_id, at) do nothing`,
      values,
    );
  }
}

/* ─────────────────────────────────────────────────────────────
 * Wiring at ingest (illustrative):
 *
 *   const logger = new BufferingEvalLogger({ sink: new PgEventSink(pgPool) });
 *   const detectorId = { id: 'eld@1.4.0', db: 'm' };
 *
 *   // Post handler, AFTER resolveSource (keep resolve-source.ts pure). Define
 *   // the resolver options ONCE and feed them to both calls, so the stamp can
 *   // never describe a different config than the one that actually ran:
 *   const resolverOpts = { uiLang, shortTextPolicy: 'prefer_prior' as const };
 *   const resolved = await resolveSource(detectInput, detector, resolverOpts);
 *   logger.recordResolution(resolutionEventFrom({
 *     messageId, maskedDetectInput: detectInput, uiLang,
 *     letterCount: (detectInput.match(/\p{L}/gu) ?? []).length,
 *     resolved,
 *     config: configStampFrom(resolverOpts, detectorId),
 *   }));                                  // fire-and-forget; not awaited
 *
 *   // Replay-widget correction handler:
 *   logger.recordCorrection({ messageId, at: Date.now(), correctedLang });
 * ───────────────────────────────────────────────────────────── */
