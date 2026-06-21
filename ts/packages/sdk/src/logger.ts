/**
 * `@chat-translate/sdk/logger` — live-traffic source-resolution capture.
 *
 * Fire-and-forget, fail-open capture of `(maskedDetectInput, resolved source,
 * provenance, confident, ...)` at ingest plus later replay corrections, sampled
 * by interestingness and config-stamped for reproducibility. Ships a buffering
 * logger, a no-op logger, and a Postgres `EventSink` (the `pg` Pool is injected
 * via `PoolLike` — `pg` is an optional peer dependency, never imported here).
 */
export * from './source-eval-logger';
