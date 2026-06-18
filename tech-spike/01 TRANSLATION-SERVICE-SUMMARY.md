# Babel-Fish Chat Translation — Design Summary

A reference for future work. Captures the context, the build-vs-buy reasoning, the
architecture we landed on, the rationale behind each layer, and what's still open.

Followed by two refinement documents:
- 02 SOURCE-RESOLUTION-SUMMARY.md
- 03 EVAL-CAPTURE-SUMMARY.md

## Context

- **Product:** a "babel-fish" service bolted onto a customer's chat / message board.
  Everyone writes in their own language; the system auto-translates each message into
  the languages other people in the room are reading.
- **Languages:** almost always European.
- **Brand:** "private intelligence." The product already uses frontier models under a
  **no-retain / no-train MSA** — so the committed privacy bar is *contractual
  non-retention*, not *nothing-leaves-our-infrastructure*.

## The build-vs-buy decision

We compared three tiers: dedicated MT APIs (DeepL, Google, Microsoft, Amazon),
frontier LLMs as translators (GPT/Claude/Gemini), and self-hosted open models
(TranslateGemma). For live chat fan-out, LLMs are too slow and cost-unpredictable on
the hot path; the real contest is **DeepL vs self-hosted TranslateGemma**.

**Decision: support both, switched per deployment (static config), not per-room or
per-message.** Rationale:

- **DeepL Pro is the strong default** for European real-time chat: quality leader on
  European pairs, low latency, native formality control (tu/vous, du/Sie), glossaries,
  a `context` parameter, and batch. On **Pro**, text is deleted after translation, never
  used for training, processed on EU servers — which **satisfies the same no-retain bar
  already accepted elsewhere in the product.** Privacy, on the customer's own terms, does
  not force self-hosting.
- **Self-hosted TranslateGemma wins** when the driving need is (a) **brand purity** — the
  stronger "nothing leaves our walls" claim for a security review, (b) **predictable cost
  at high volume**, or (c) **offline/edge**. 55 languages; also multimodal (can translate
  text inside images).
- A **per-deployment switch** gives the hedge, a cost escape hatch, and no vendor lock-in,
  while keeping all the expensive engineering provider-agnostic and written once.

**Unresolved input to this decision:** expected **message volume** and **average number of
distinct languages co-viewed per room**. Those two numbers set the cost crossover between
DeepL's per-character bill and an amortized GPU. (See Open Questions.)

## Architecture (four layers, low → high)

The guiding principle: **the provider boundary sits as low as possible** — it knows how to
translate a string and nothing about chat. Everything chat-specific lives above it, written
once, and works regardless of backend.

1. **`translation-provider.ts` — provider abstraction.**
   `TranslationProvider` interface with a `capabilities()` descriptor; `DeepLProvider` and
   `TranslateGemmaProvider` (Ollama); `createProvider` / `configFromEnv` is the
   per-deployment switch. `BaseTranslationProvider` gives every backend a correct sequential
   `translateBatch` for free; DeepL overrides it with its native 50-text batch.

2. **`chat-translation-layer.ts` — masking + orchestration.**
   `maskNonTranslatable` replaces do-not-translate spans (@mentions, emails, URLs,
   #channels, `code`, emoji) with opaque sentinels, translates around them, and restores
   them verbatim. `createChatTranslator` does capability-driven source resolution and cheap
   short-circuits.

3. **`chat-fanout-cache.ts` — fan-out + cache.**
   `createRoomTranslator` masks once and translates that single masked form into each
   distinct *viewed* language, restoring per target. `TranslationCache` interface +
   `InMemoryTranslationCache` (LRU+TTL); returns `RoomStats` for observability.

4. **`translation-eval.ts` + `sample-chat-eval.jsonl` — eval harness.**
   `runEval` ranks a *list* of candidates on real chat traffic. Includes
   `LlamaCppTranslateGemmaProvider` so you can test the Q8 GGUF you'd actually ship.

## Key design decisions & rationale

- **`capabilities()` drives graceful degradation.** A flag is `true` only if the chat layer
  can *depend* on it; best-effort approximations stay `false` so nothing builds on sand. The
  worked example is `autoDetectSource`: DeepL can, TranslateGemma can't — so the layer runs
  its own detector before calling the local backend, and both behave identically upstream.
- **Mask once, restore per target.** The expensive provider-bound work fans out; the cheap
  string work repeats. Each target's `restore()` re-inserts that message's own mentions/URLs.
- **Cache on the masked text, not the raw text.** Two messages differing only by which user
  they @-mention or which URL they link mask to the *same* form, so they share a cache entry.
  This is the whole reason masking happens before caching. The cache stores the provider's
  output **before restore** (still carrying sentinels), keyed on
  `(masked, source, target, formality, glossaryId)`.
- **Configure a detector even on DeepL.** It enables the same-language short-circuit (skip a
  paid call when a viewer already reads the source language) and gives concrete cache keys
  instead of `"auto"`. Without one, DeepL still works but pays a call per target.
- **Concurrency defaults off a capability.** `caps.nativeBatch ? 8 : 1` — parallel for managed
  DeepL, serialized for the single-GPU self-hosted path (which can OOM under parallel load).
- **Sentinel robustness is provider-dependent, and we made it visible.** Private-Use-Area
  tokens survive NMT (DeepL) almost perfectly; a raw LLM can occasionally drop one.
  `findUnrestored` / `unrestoredTokens` surfaces any token the engine ate so the LLM
  deployment can fall back. (DeepL's native `ignore_tags` is stronger but provider-specific,
  so the portable layer doesn't depend on it.)
- **Prefer `prefer_*` formality variants.** Plain `more`/`less` make DeepL error on languages
  that don't grammaticalise formality; `prefer_*` silently no-ops — friendlier for a mixed room.

## Eval methodology (so results aren't misleading)

- **Three honest tiers of signal:** (1) *exact, reference-free* placeholder survival, error
  rate, latency, throughput, cost; (2) *chrF* when a human reference exists (relative signal —
  from-scratch impl, not sacreBLEU); (3) *LLM-as-judge* for adequacy / fluency / **register**
  (injected via `buildJudgePrompt`, wired to your own frontier model under the MSA).
- **Placeholder survival is the headline chat metric** and the one expected to separate the
  quants: DeepL ≈ 100%, Q8 high, Q4 visibly lower.
- **Quant methodology trap (important):** Ollama only surfaces **Q4_K_M**. If you ship
  self-hosted you'd run a higher-fidelity **GGUF (Q8_0, possibly BF16) under llama.cpp** —
  most likely **Q8**. Evaluating Q4 and concluding "local is worse" would be a methodology
  error. Test the quant you'd actually deploy.
- **Guards:** mask-then-restore (measures the real pipeline), call the provider directly
  (bypass the cache → cold latency), run candidates sequentially (co-located GPU backends
  don't contend and skew each other's latency).

## Files

| File | Layer |
|---|---|
| `translation-provider.ts` | Provider interface, capabilities, DeepL + Ollama adapters, per-deployment switch |
| `chat-translation-layer.ts` | Do-not-translate masking + capability-driven orchestration |
| `chat-fanout-cache.ts` | Per-room fan-out, cache contract + in-memory LRU+TTL |
| `translation-eval.ts` | Eval harness, llama.cpp/GGUF adapter, chrF, judge prompt, report |
| `sample-chat-eval.jsonl` | Tiny European-pair sample exercising masking/register |

## Open questions & next steps

- **Cost model (not yet built):** takes message volume + avg distinct languages per room +
  cache hit rate, plots DeepL's per-character bill against amortized GPU cost for the Q8
  self-hosted path. This is what turns the build-vs-buy line into actual numbers.
- **Final quality call:** run the top two candidates through **sacreBLEU** and a **small human
  eval**; the in-harness chrF only narrows the field.
- **If self-hosting:** ship **Q8 GGUF via llama.cpp**, not Q4 via Ollama; align the Gemma
  translation prompt (`buildPrompt` / `buildGemmaPrompt`) with the published model card.
- **UX (not yet designed):** replace original vs show both; how to signal "auto-translated"
  so people extend grace when it's slightly off.
- **Scale-out:** swap `InMemoryTranslationCache` for Redis on multi-instance deployments
  (hash the flat key); consider batching multiple new messages bound for the same target.
- **Glossary / brand terms:** product names you never want translated — DeepL glossaries on
  that deployment; masking handles the rest provider-agnostically.
