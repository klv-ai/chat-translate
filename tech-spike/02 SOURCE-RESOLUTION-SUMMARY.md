# Babel-Fish Chat Translation — Source-Language Resolution

A second companion to `TRANSLATION-SERVICE-SUMMARY.md`. That doc covers the provider
boundary, masking, fan-out, cache, and the build-vs-buy call. This one covers the
piece that sits at message **ingest**: deciding what language a message was
written in, once, and writing that fact onto the row so nothing downstream has to
guess. Captures the reasoning, the decisions, the knobs left to data, and what's
still open.

## Context — why a source-language resolver

- **The local backend forces the issue.** `TranslateGemmaProvider.translate`
  throws on a missing `sourceLang` — its `capabilities().autoDetectSource` is
  `false` because the raw model can't do it as a feature the layer can depend on.
  So on the self-hosted path a detector isn't an optimization, it's load-bearing:
  **every code path must end in a concrete language code; null is not allowed.**
- **DeepL doesn't need it but benefits.** DeepL can auto-detect (pass `null`), so
  there a detector only buys the same-language short-circuit and concrete cache
  keys instead of `"auto"`. Still worth running — see the main summary.
- **We're persisting source as a fact anyway.** The wider app stores each message
  as written, plus each translation against its origin message id and target
  language, for latency-free history. A persisted row needs a concrete source
  regardless of backend — which lines up exactly with the "never null"
  requirement above and pushes detection to ingest (below).

## The detector choice

Hard constraint: on the TranslateGemma deployment the detector must stay
**in-process and local** — a cloud detect API would quietly undo the "nothing
leaves our walls" claim that deployment exists to make. Chat is also the
adversarial case: messages are short, and we detect on the **masked** text
(sentinels → spaces), which strips @mentions/URLs/emoji/code and can shrink an
already-short message further.

Candidates weighed against those constraints:

- **franc** — pure JS, no deps, but the wrong fit. Tuned for documents, returns
  `und` under ~10 characters, and emits ISO 639-3 (`eng`/`fra`), needing a
  mapping layer to reach the 2-letter codes the rest of the system uses.
- **ELD (`eld`, nitotm/efficient-language-detector-js)** — **chosen.** Vanilla
  JS, no deps, Apache-2.0, Node 16+. Purpose-built for short sentences with
  accuracy near the heavy detectors. Fits the code almost exactly: synchronous
  `detect()` returning `{ language, getScores(), isReliable() }` in **ISO 639-1**
  (no code mapping), `setLanguageSubset([...])` to constrain to the European set
  (better accuracy + latency), `isReliable()` as the confidence gate, and several
  DB sizes (≈37 MB XS → 138 MB L RAM) to trade footprint for accuracy.
- **fastText `lid.176`** — the accuracy ceiling and still local, but heavier ops
  (native/WASM bindings + a model file). The **upgrade path** if a measured eval
  shows ELD's large DB isn't enough — not the starting point.

Adapter: `eldDetector(eld, subset?)` wraps ELD as a `ConfidenceDetector`,
constraining to the room's plausible languages once at construction.

## Where detection runs: once, at ingest

Detection moved off the translation hot path. The flow:

- **Post →** mask, build `detectInput`, call `resolveSource`, write the
  **resolved** source onto the message row.
- **Live fan-out →** reads the source off the row and passes it as
  `opts.sourceLang`, so `createRoomTranslator`'s own `resolveSource` takes its
  `explicit` branch and never detects. Its detector is now only a safety net.
- **History replay →** reads the persisted translation rows: zero MT, zero
  detection.

`resolveSource` (in `resolve-source.ts`) precedence, highest signal first:

1. **user-confirmed override** (replay-widget correction) — ground truth;
   `provenance: 'user_confirmed'`, `confident: true`. Don't even detect.
2. **detection unusable** (null / not reliable / below optional `minScore`) →
   UI-language prior; `'ui_fallback'`, `confident: false`.
3. **detection + enough letters** (`>= shortTextLetters`) → trust it outright,
   including when it overrides the prior (the German-UI user writing English);
   `'detected'`, `confident: true`.
4. **detection + short + agrees with prior** → trust it; `'detected'`,
   `confident: true`.
5. **detection + short + disagrees with prior** → the ambiguous zone; see the
   policy knob below. Either way `confident: false`.

Two deliberate properties: it **never returns null**, and it's **deterministic**
(same masked input + same uiLang always resolves identically), so the source in
the cache key and the source on the row can't drift apart.

## Storage model

- **Resolved source is the source of truth — not the raw UI code.** The row may
  carry the author's *declared* UI language, but the field fed to the provider
  and used for the same-language skip must be the **resolved** source. Storing
  and translating from the raw UI code would make a German-UI user's English
  message render as German for everyone, forever, in history.
- **Columns worth keeping:** resolved `lang`, declared `uiLang`, `provenance`,
  `confident`, and `detected` (what the detector said even when unused — the
  telemetry that tells you how often the prior is overriding a distrusted short
  detection, i.e. the signal for tuning the threshold).
- **Store the masked form as its own artifact.** Cheap insurance: a late-arriving
  target language translates from the stored masked string without re-masking,
  and it pins exactly what was sent even if `DEFAULT_RULES` changes later, so
  historical messages don't silently re-mask differently.
- **Two stores, two grains, tied by the resolved source.** The cache holds
  provider output **before restore** (still carrying sentinels), keyed on the
  masked form, so messages differing only by mention/URL share an entry. The
  persistent table holds the **final restored** text, keyed on (message id,
  target). The resolved source is the same value in both, so they can't diverge.

## The short-text guard and the policy knob

Below `shortTextLetters` (default 10) we're in the loanword danger zone, where a
detector can be **confidently wrong** on "ok" / "ciao" / "cool". There, the
behaviour on a reliable-but-disagreeing detection is governed by `shortTextPolicy`:

- **`prefer_prior`** (default) — keep the UI language. Safest against loanword
  false positives.
- **`trust_detector`** — take the detection. Better when users code-switch in
  short bursts; costs you on loanwords.

This is a **knob, not a hardcoded choice**, precisely because the right answer
depends on traffic we haven't measured. Either branch marks `confident: false`,
keeping the row revisitable.

## Replay-widget feedback loop

A UI affordance lets a reader correct a wrong language. The widget posts
`{ messageId, correctedLang }`; we call `resolveSource` with
`override: correctedLang` (no re-detection — it's ground truth), update the row to
`user_confirmed / confident: true`, and re-translate the **stored masked form**
into each already-stored target, overwriting those persistent rows. Old cache
entries were keyed on the old source and simply age out; bust them explicitly if
you want the correction to feel instant for other live viewers. Net effect:
corrections become durable ground-truth labels for free.

## Measuring the decision (`source-resolution-eval.ts`)

A **sibling** stage to `translation-eval.ts`, not bolted into `runEval`: that
harness grades a provider's translated *output*; this one grades a *decision*
(resolved source vs gold label) and never calls a translation provider. It reuses
the real masking to build the same `detectInput`, and runs the real
`resolveSource`. Two outputs:

- **Detector calibration (assumption-free).** Bins cases by letter count;
  reports reliable% and correct-when-reliable% per bin. The `correct|reliable%`
  column is the raw signal for setting `shortTextLetters` — put the threshold
  just above the length where it falls off. No `uiLang` needed.
- **Policy sweep.** Runs `shortTextLetters` × `{prefer_prior, trust_detector}`
  and reports accuracy plus the **short-disagreement breakdown**. The headline is
  not overall accuracy (that moves with traffic mix) but the `det right : prior
  right` ratio *within the short-disagreement zone* — the loanword-vs-code-switch
  question answered directly, independent of how often the zone occurs.
- **The `uiLang` model brackets the truth.** Under the native assumption
  (`uiLang = gold`) there are no code-switches by construction, so you only see
  the loanword-protection side; `linguaFrancaUi('en')` reframes non-English
  messages as code-switches to surface the other side. Run both; reality sits
  between.

**Honesty caveats.** The bundled sample is ten rows and several short-circuit as
untranslatable, so first numbers are a wiring check, not a verdict. A real
threshold/policy call wants a few hundred labelled in-domain short messages, ideally
with the real per-row `uiLang` rather than a modelled one. Detection is
deterministic, so re-runs are stable.

## New files

| File | Layer |
|---|---|
| `resolve-source.ts` | Ingest-time source resolver: `ConfidenceDetector` + `eldDetector` adapter, `ResolvedSource` (provenance / confident / detected), precedence ladder, short-text guard + `shortTextPolicy` |
| `source-resolution-eval.ts` | Measured stage: detector calibration + policy sweep, `linguaFrancaUi` stress model, markdown report |

## Key design decisions & rationale (condensed)

- **Detect once at ingest, persist the result.** Keeps detection off the hot path
  and stops the cache and the persistent store from drifting onto different
  sources.
- **UI language is the fallback prior, detection overrides it when confident.**
  The UI language beats an "author's last language" prior because it has no cold
  start, is deterministic, and is free from message #1.
- **Confidence is a persisted flag, not just a runtime decision.**
  `confident: false` rows are a re-visitable backlog — re-detect after a model
  upgrade, or nudge the replay widget more prominently.
- **The policy is data-driven, not assumed.** `shortTextPolicy` defaults to the
  safe `prefer_prior`; the eval exists to justify changing it.

## Open questions & next steps

- **Logging / labelled-collection process (separate thread).** A live-traffic
  logger emitting `(masked detectInput, resolved source, provenance, confident,
  later replay correction)` in the `SourceEvalCase` shape, so the eval graduates
  from the toy sample to real room data and replay corrections become gold labels.
- **Set `shortTextLetters` from calibration** on that real data, not the default.
- **Decide `prefer_prior` vs `trust_detector`** from the short-disagreement ratio
  once there's enough labelled short traffic to read it.
- **DeepL-path policy.** There, detection can stay on with a `null` fallback
  (auto-detect available) instead of a default — slightly different wiring from
  the local path, same resolver.
- **Re-detect backlog.** After shipping a stronger model (or moving ELD DB size),
  sweep the `confident: false` rows and re-resolve.
