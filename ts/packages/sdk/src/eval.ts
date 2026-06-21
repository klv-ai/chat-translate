/**
 * `@chat-translate/sdk/eval` — translation-quality eval harness.
 *
 * Ranks candidate providers on real chat traffic: placeholder survival, error
 * rate, latency, throughput, cost, optional chrF against a reference, and an
 * optional LLM-as-judge (adequacy / fluency / register). Offline tooling — kept
 * off the serving path and behind its own subpath.
 *
 * Note: `formatReportMarkdown` here is distinct from the same-named export in
 * `@chat-translate/sdk/eval/source`; the two never meet in one module.
 */
export * from './translation-eval';
