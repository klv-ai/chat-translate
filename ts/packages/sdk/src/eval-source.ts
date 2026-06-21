/**
 * `@chat-translate/sdk/eval/source` — source-resolution eval.
 *
 * Grades a DECISION (resolved source vs gold label), never calling a provider.
 * Two outputs: assumption-free detector calibration (binned by letter count)
 * and a policy sweep over `shortTextLetters` × `{prefer_prior, trust_detector}`
 * with the short-disagreement breakdown that the loanword-vs-code-switch call
 * hinges on.
 *
 * Note: `formatReportMarkdown` here is distinct from the same-named export in
 * `@chat-translate/sdk/eval`; the subpath split is what keeps them apart.
 */
export * from './source-resolution-eval';
