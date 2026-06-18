# Babel-Fish Chat Translation — Eval Capture & Reporting

The third companion to `TRANSLATION-SERVICE-SUMMARY.md` (provider boundary, masking,
fan-out, cache, build-vs-buy) and `SOURCE-RESOLUTION-SUMMARY.md` (ingest-time source
resolution). This one covers the layer that turns live traffic into the labelled
dataset the source-resolution eval needs — the **"Logging / labelled-collection
process"** left open at the end of that document. It also records two things settled in
the same thread: the `shortTextPolicy` knob finally landing in the resolver, and the
decision about a reporting/aggregation layer on top of the captured data.

## Context — why a capture layer at all

The source-resolution eval grades a **past prediction** against a **later label**. Two
facts about that shape the whole design:

- **The label arrives late and out-of-band.** A replay correction can land days after
  ingest, so a record has to be joinable by `messageId` across two events separated in
  time.
- **The prediction must be frozen.** This is the trap hiding in the serving path: the
  replay flow overwrites the message row to `user_confirmed`, clobbering the original
  resolved `lang` / `provenance`. That is *correct for history* (you want the corrected
  value rendered), but it destroys exactly what the eval grades — what the resolver
  decided **before** the human overruled it.

Serving (mutable, latest-wins) and eval (prediction frozen, label appended) want
**opposite update semantics on the same fact**. That conflict is the entire reason the
two are split apart.

## Store decision: pure log vs. table

**Neither in the naive form — the answer is a dedicated, append-only events store you
own, in its own Postgres schema.**

- **Not an ephemeral observability log.** Those are built for short-retention debugging
  and sampling, not for slowly accumulating a labelled dataset whose labels trickle in
  over weeks. The disqualifier is the in-walls constraint: the masked `detectInput`
  still carries the message's actual words, so shipping it to a SaaS log vendor quietly
  breaks the "nothing leaves our walls" promise the TranslateGemma deployment exists to
  make. Whatever captures this must be able to live entirely in-walls.
- **Not the serving message row.** Beyond the clobbering problem above, in a busy
  install the messages table gets messy fast if it carries eval bookkeeping. A dedicated
  namespace keeps the two concerns from contaminating each other.
- **→ Log-*shaped* emission** (fire-and-forget, append-only, off the hot path) **into a
  queryable store you control** (`source_eval` schema, two fact tables).

## The two-event design

Two append-only event types, joined at read time into the eval's `SourceEvalCase`:

- **`SourceResolutionEvent`** — the **frozen prediction**, captured at ingest. Carries
  everything needed to interpret the decision *and* re-run the resolver: the join keys
  (`messageId`, `at`), the inputs the resolver saw (`maskedDetectInput`, `uiLang`,
  `letterCount`), the outputs (`lang`, `provenance`, `confident`, `detected`, `score`),
  the **config stamp**, and the **`sampleRate`** it was kept at.
- **`SourceCorrectionEvent`** — the **gold label**, captured whenever a replay
  correction posts. Just `messageId`, `at`, `correctedLang`.

Prediction lives in one event untouched; label lives in the other. Both survive,
because neither overwrites the other.

## Three load-bearing properties

1. **Off the hot path, fail-open.** Recording is fire-and-forget into a bounded
   in-memory buffer with batched async flush. The caller **never awaits** it. If the
   sink is down we **drop-with-a-counter** — we never block or fail a chat post to
   capture an eval sample. Losing samples is acceptable; losing a message is not.
   `BufferingEvalLogger` guards overlapping flushes, drops-and-counts on sink error
   (re-queueing risks unbounded growth), and `unref()`s its flush timer so it never
   keeps the process alive.
2. **Sample by interestingness, not uniformly.** Confident long-text detections are
   abundant and low-signal; short / not-confident / corrected rows are the whole point.
   `defaultResolutionSampler` keeps **100%** of the interesting cases and thins only
   confident long-text `detected` rows to ~5%. This bounds volume *and* raises dataset
   quality together. The per-row keep-rate is stamped so the eval can reweight thinned
   classes back up. **Corrections are never sampled** — they're the gold labels.
3. **Config-stamped.** Every resolution records the thresholds / policy / detector that
   produced it (`ResolverConfigStamp`). Without this, tuning a knob makes historical
   rows unreproducible and you can't tell a data change from a config change.

## The config stamp & the anti-drift discipline

The stamp is only useful if it records what *actually ran*. The failure mode is a
hand-copied config literal at the call site that silently drifts from the resolver's
real behaviour. Two moves close that gap:

- **`shortTextPolicy` now exists in the resolver.** It was previously *doc-only* —
  `SOURCE-RESOLUTION-SUMMARY.md` described it as a knob while `resolve-source.ts`
  hardcoded `prefer_prior` in step 5. It is now a real `ResolveSourceOptions` field
  (default `prefer_prior`), with `ShortTextPolicy` exported as the single source of
  truth for the union — imported by both the eval's policy sweep and the stamp. On a
  short-text disagreement, `prefer_prior` keeps `uiLang` (`ui_fallback`) and
  `trust_detector` takes the detection (`detected`); **both mark `confident: false`.**
- **`configStampFrom(opts, detector)`** builds the stamp from the *same* options object
  passed to `resolveSource`, applying `resolve-source`'s **exported defaults**
  (`DEFAULT_SHORT_TEXT_LETTERS`, `DEFAULT_SHORT_TEXT_POLICY`). If a default changes in
  the resolver, the stamp follows automatically. The ingest call site defines the
  options once and feeds both calls, so the stamp cannot describe a different config
  than the one that ran.

One subtlety worth holding onto: `trust_detector` yields `provenance: 'detected'` with
`confident: false`. That pairing is intentional — `confident` is the revisit signal,
not `provenance` — and it means the default sampler keeps **100%** of those rows (its
easy-bulk branch requires `confident`), which is exactly the short-disagreement traffic
the policy sweep needs to read.

## Reporting / aggregation (Cube)

A semantic-layer aggregator (e.g. Cube) is a **good fit for the reporting read-path**,
with a sharp line around where it sits.

- **It sits on the read side, on the two fact tables.** It does not touch the
  fire-and-forget capture path. Pre-aggregations are purpose-built for the dashboards we
  want — correction rate by language by month, provenance distribution, the
  `confident: false` backlog over time — which are exactly time-partitioned rollups over
  an append-only fact table. Defining "correction rate" once in the semantic layer keeps
  the metric from drifting across tiles. Self-hostable, so it respects the in-walls
  constraint (verify its pre-agg cache store stays in-walls too).
- **Keep it strictly separate from the eval.** Cube answers *"what's happening in prod
  right now"*; `source-resolution-eval.ts` answers *"would tuning this knob help"* by
  grading decisions against gold labels with the policy sweep. A pretty dashboard is
  seductive and must not become the basis for a tuning call it can't actually support.
- **The headline metric is skewed — label it honestly.** Corrections measure
  *wrong-and-noticed*, not error rate; silence isn't a clean positive. A tile labelled
  "accuracy" off correction counts would actively mislead. Call it **correction rate**,
  and keep the real accuracy estimate coming from hand-labelled uncorrected rows via the
  eval.
- **Mind the late-correction / partition-refresh gotcha.** This is the same out-of-band
  label that forced the two-event split, resurfacing in pre-aggregations: a correction
  that lands days later attaches to an *old* partition, and a by-month rollup that treats
  past months as sealed will silently undercount it. The refresh key must revisit
  recent-enough partitions (e.g. key on `max(correction.at)`).
- **Don't stand it up before volume warrants.** Same instinct as
  `InMemoryCache → Redis`: for a small single install, a couple of SQL views over the
  two tables may cover early dashboards. The Cube payoff shows up with real volume and
  many tiles — which lines up with the still-unresolved volume question in the main
  summary.

## Storage model

- **Own schema, two append-only fact tables**, range-partitioned by month so
  retention/archival is a partition drop and the eval just reads a window.
- **Idempotent writes** keyed on `(message_id, at)` (`on conflict do nothing`), so a
  retried flush after a partial failure is safe.
- **`sample_rate` is persisted** per row so thinned classes can be reweighted at eval
  time.
- The `EventSink` is injectable: `PgEventSink` is the Postgres implementation, but a
  queue (Kafka/SQS) → durable store satisfies the same contract for multi-instance
  deployments. `NoopEvalLogger` turns capture off (DeepL deployment, local dev).

## Key design decisions & rationale (condensed)

- **Two events, not one mutable row.** Late, out-of-band labels plus the requirement to
  freeze the prediction make a single overwritable row impossible. Capture is its own
  append-only store, never the serving row.
- **Dedicated schema over the messages table.** Keeps eval bookkeeping out of a hot
  serving table that gets messy under load, and keeps the in-walls boundary clean.
- **Fail-open, off the hot path.** Sample loss is acceptable; a blocked or failed chat
  post is not. Recording can never back-pressure ingest.
- **Sample by interestingness + stamp the rate.** Thins the abundant easy bulk, keeps
  every hard case, and stays statistically honest via per-row reweighting.
- **Config-stamped, drift-proof.** The stamp is derived from the same options the
  resolver ran with, using the resolver's own exported defaults — so it records the
  effective config and follows default changes automatically.
- **Aggregator is read-only and downstream.** Reporting sits on the fact tables; it
  neither replaces the eval nor touches the capture path.

## New files

| File | Layer |
|---|---|
| `source-eval-logger.ts` | Live-traffic capture: `SourceResolutionEvent` / `SourceCorrectionEvent`, `SourceEvalLogger` + `BufferingEvalLogger` (fire-and-forget, bounded, fail-open), `defaultResolutionSampler`, `configStampFrom` / `resolutionEventFrom`, `EventSink` + `PgEventSink` + DDL, `NoopEvalLogger` |

(Companion change, not a new file: `resolve-source.ts` gained the `shortTextPolicy`
knob, the exported `ShortTextPolicy` type, and exported defaults.)

## Open questions & next steps

- **Set `shortTextLetters` and decide the policy from real data.** The logger graduates
  the eval from the ten-row toy sample to live room traffic; once there's enough
  labelled short traffic, set the threshold from the calibration bins and pick
  `prefer_prior` vs `trust_detector` from the short-disagreement ratio.
- **Honest accuracy needs uncorrected rows too.** Corrections are skewed
  (wrong-and-noticed). A real calibration number wants a few hundred *hand-labelled*
  in-domain short messages, ideally with the per-row `uiLang` — the corrected rows are
  free gold; the uncorrected sample is what you pull for manual labelling.
- **Multi-instance sink.** Swap `PgEventSink` for a queue → durable store when running
  more than one node, kept in-walls on the self-hosted deployment.
- **Cube refresh strategy.** Implement the late-correction partition refresh before
  trusting any by-month correction-rate rollup.
- **DeepL-path capture.** `NoopEvalLogger` is the default there, but capturing on DeepL
  too would enable cross-deployment comparison of resolution quality — worth it if the
  policy call needs to hold across both backends.
- **Leave the inline resolvers alone.** `chat-translation-layer.ts` and
  `chat-fanout-cache.ts` each carry their own inline `resolveSource` safety net (the
  live-fan-out path that reads the persisted source off the row via the `explicit`
  branch). Those are intentionally separate from the ingest resolver and do **not** take
  the `shortTextPolicy` knob — don't thread it through them.
