/**
 * @chat-translate/sdk — main entry.
 *
 * The live, provider-agnostic translation surface: the provider abstraction
 * (DeepL / self-hosted TranslateGemma), do-not-translate masking, the chat
 * orchestration layer, the per-room fan-out + cache, and ingest-time source
 * resolution. This is everything the serving path needs.
 *
 * Heavier / optional tooling lives behind subpaths so the core install stays
 * dependency-free:
 *   - `@chat-translate/sdk/eval`        translation-quality eval harness
 *   - `@chat-translate/sdk/eval/source` source-resolution eval (calibration + policy sweep)
 *   - `@chat-translate/sdk/logger`      live-traffic capture logger + Postgres sink
 *   - `@chat-translate/sdk/detector`    ELD-backed LanguageDetector adapters (needs `eld`)
 */

export * from './chat-fanout-cache';
export * from './chat-translation-layer';
export * from './detector';
export * from './resolve-source';
export * from './translation-provider';
