/**
 * translation-provider.ts
 *
 * Provider abstraction for the babel-fish chat translation service.
 *
 * Design intent:
 *   - The provider boundary sits as LOW as possible: it knows how to translate
 *     a string, and nothing about chat. Do-not-translate masking (@mentions,
 *     URLs, emoji, code), caching, and fan-out-to-viewed-languages all live
 *     ABOVE this interface, written once and reused regardless of backend.
 *   - capabilities() is the contract that lets the chat layer degrade
 *     gracefully instead of assuming the two backends are interchangeable.
 *   - The switch is per-deployment (static config), not per-room/per-message.
 *     See createProvider() / configFromEnv() at the bottom.
 *
 * Runtime: Node 18+ (uses global fetch). No npm dependencies.
 */

// ─────────────────────────────────────────────────────────────
// 1. Core types
// ─────────────────────────────────────────────────────────────

/** BCP-47-ish language code, e.g. "de", "fr", "pt-BR", "en-GB". */
export type LanguageCode = string;

/**
 * Register / politeness control. Maps directly onto DeepL's formality
 * parameter. Only meaningful for languages that grammaticalise it
 * (de, fr, it, es, nl, pl, pt, ru, ja, ...).
 *
 * Tip for the chat layer: prefer the "prefer_*" variants. Plain "more"/"less"
 * make DeepL *error* on languages that don't support formality, whereas
 * "prefer_*" silently no-ops — much friendlier for a mixed-language room.
 */
export type Formality = 'default' | 'more' | 'less' | 'prefer_more' | 'prefer_less';

export interface TranslateOptions {
  /** Omit / null = let the provider auto-detect — IF capabilities allow it. */
  sourceLang?: LanguageCode | null;
  targetLang: LanguageCode;
  formality?: Formality;
  /** Provider-side glossary identifier, if the backend supports one. */
  glossaryId?: string;
  /**
   * Surrounding text (e.g. the previous few chat messages) used to
   * disambiguate the message WITHOUT being translated. This is the single
   * biggest quality lever for short chat messages ("yeah, same here").
   */
  context?: string;
}

export interface TranslateResult {
  /** The translated text. */
  text: string;
  /** What the provider believed the source language was. */
  detectedSourceLang: LanguageCode;
}

/**
 * Honest, relyable-by-the-caller capabilities. A flag is `true` only if the
 * chat layer can DEPEND on it. Best-effort approximations (e.g. coaxing a raw
 * model toward a register via prompt) stay `false` so nothing builds on sand.
 */
export interface ProviderCapabilities {
  /** Can infer the source language when sourceLang is omitted. */
  autoDetectSource: boolean;
  /** First-class formality/register control the chat layer can rely on. */
  formality: boolean;
  /** Server-side, grammatically-adapted glossaries. */
  glossaries: boolean;
  /** Accepts a non-translated context hint to disambiguate short messages. */
  contextHint: boolean;
  /** True native multi-text batch in a single round-trip. */
  nativeBatch: boolean;
  /** Max texts per batch call (only meaningful when nativeBatch is true). */
  maxBatchSize?: number;
}

// ─────────────────────────────────────────────────────────────
// 2. Errors
// ─────────────────────────────────────────────────────────────

export class TranslationError extends Error {
  constructor(
    message: string,
    readonly provider: string,
    readonly cause?: unknown,
    /** Hint to the layer above: is a retry worth attempting? */
    readonly retryable: boolean = false,
  ) {
    super(message);
    this.name = 'TranslationError';
  }
}

// ─────────────────────────────────────────────────────────────
// 3. The interface + shared base
// ─────────────────────────────────────────────────────────────

export interface TranslationProvider {
  readonly name: string;
  capabilities(): ProviderCapabilities;
  translate(text: string, options: TranslateOptions): Promise<TranslateResult>;
  translateBatch(texts: string[], options: TranslateOptions): Promise<TranslateResult[]>;
  /** Cheap liveness probe — matters far more for the self-hosted path. */
  healthCheck(): Promise<boolean>;
}

/**
 * Shared scaffolding. The important default here is translateBatch: providers
 * without a native batch endpoint (Ollama) get a correct, if slower,
 * sequential implementation for free. Providers that have one (DeepL) override
 * it. Sequential (not Promise.all) is deliberate for the self-hosted path: a
 * single GPU does not benefit from being hammered in parallel and may OOM.
 */
export abstract class BaseTranslationProvider implements TranslationProvider {
  abstract readonly name: string;
  abstract capabilities(): ProviderCapabilities;
  abstract translate(text: string, options: TranslateOptions): Promise<TranslateResult>;
  abstract healthCheck(): Promise<boolean>;

  async translateBatch(texts: string[], options: TranslateOptions): Promise<TranslateResult[]> {
    const out: TranslateResult[] = [];
    for (const t of texts) out.push(await this.translate(t, options));
    return out;
  }
}

// ─────────────────────────────────────────────────────────────
// 4. DeepL adapter (managed pass-through)
// ─────────────────────────────────────────────────────────────

export interface DeepLConfig {
  apiKey: string;
  /** Override the host if needed; auto-selected from the key by default. */
  apiUrl?: string;
}

export class DeepLProvider extends BaseTranslationProvider {
  readonly name = 'deepl';
  private readonly endpoint: string;

  constructor(private readonly config: DeepLConfig) {
    super();
    // Free keys end in ":fx" and must hit the free host.
    const isFree = config.apiKey.endsWith(':fx');
    this.endpoint =
      config.apiUrl ?? (isFree ? 'https://api-free.deepl.com/v2' : 'https://api.deepl.com/v2');
  }

  capabilities(): ProviderCapabilities {
    return {
      autoDetectSource: true,
      formality: true,
      glossaries: true,
      contextHint: true, // DeepL's `context` param — no extra infra needed
      nativeBatch: true,
      maxBatchSize: 50,
    };
  }

  async translate(text: string, options: TranslateOptions): Promise<TranslateResult> {
    const [r] = await this.translateBatch([text], options);
    return r;
  }

  // Real batch: one round-trip translates many texts to a single target.
  async translateBatch(texts: string[], options: TranslateOptions): Promise<TranslateResult[]> {
    const body: Record<string, unknown> = {
      text: texts,
      target_lang: options.targetLang,
    };
    // Note: a glossary requires an explicit source_lang (can't auto-detect).
    if (options.sourceLang) body.source_lang = options.sourceLang;
    if (options.formality && options.formality !== 'default') {
      body.formality = options.formality;
    }
    if (options.glossaryId) body.glossary_id = options.glossaryId;
    if (options.context) body.context = options.context;

    let res: Response;
    try {
      res = await fetch(`${this.endpoint}/translate`, {
        method: 'POST',
        headers: {
          Authorization: `DeepL-Auth-Key ${this.config.apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });
    } catch (cause) {
      throw new TranslationError('DeepL request failed', this.name, cause, true);
    }

    if (!res.ok) {
      const retryable = res.status === 429 || res.status >= 500;
      throw new TranslationError(
        `DeepL returned ${res.status}`,
        this.name,
        await safeText(res),
        retryable,
      );
    }

    const json = (await res.json()) as {
      translations: { detected_source_language: string; text: string }[];
    };
    return json.translations.map((t) => ({
      text: t.text,
      detectedSourceLang: t.detected_source_language.toLowerCase(),
    }));
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.endpoint}/usage`, {
        headers: { Authorization: `DeepL-Auth-Key ${this.config.apiKey}` },
      });
      return res.ok;
    } catch {
      return false;
    }
  }
}

// ─────────────────────────────────────────────────────────────
// 5. TranslateGemma adapter (self-hosted, via Ollama)
// ─────────────────────────────────────────────────────────────

export interface TranslateGemmaConfig {
  /** Ollama host. Default: http://localhost:11434 */
  host?: string;
  /** e.g. "translategemma:4b" | "translategemma:12b" | "translategemma:27b" */
  model?: string;
}

/**
 * ISO 639-1 -> English language name. The instruction-tuned TranslateGemma
 * prompt names the languages in prose ("a professional English (en) to German
 * (de) translator"), so a code alone will not do. Region subtags are ignored;
 * an unmapped code falls back to itself, which still yields a usable prompt.
 *
 * Kept in sync with `LANGUAGE_NAMES` in the Python package.
 */
export const LANGUAGE_NAMES: Record<string, string> = {
  en: 'English',
  fr: 'French',
  de: 'German',
  es: 'Spanish',
  it: 'Italian',
  pt: 'Portuguese',
  nl: 'Dutch',
  pl: 'Polish',
  ru: 'Russian',
  uk: 'Ukrainian',
  sv: 'Swedish',
  da: 'Danish',
  no: 'Norwegian',
  nb: 'Norwegian',
  fi: 'Finnish',
  cs: 'Czech',
  sk: 'Slovak',
  sl: 'Slovenian',
  el: 'Greek',
  hu: 'Hungarian',
  ro: 'Romanian',
  bg: 'Bulgarian',
  hr: 'Croatian',
  sr: 'Serbian',
  et: 'Estonian',
  lv: 'Latvian',
  lt: 'Lithuanian',
  ga: 'Irish',
  mt: 'Maltese',
  is: 'Icelandic',
  tr: 'Turkish',
  ca: 'Catalan',
  ja: 'Japanese',
  zh: 'Chinese',
  ko: 'Korean',
  ar: 'Arabic',
  hi: 'Hindi',
};

/** English name for a language code, region subtag ignored. */
export function languageName(code: string): string {
  return LANGUAGE_NAMES[code.split('-')[0]?.toLowerCase() ?? ''] ?? code;
}

/**
 * The instruction-tuned TranslateGemma prompt, verbatim from the published
 * model card. Used by backends whose chat template does NOT build the
 * translation instruction itself — Ollama's stock Gemma template just wraps the
 * user message, so the full instruction has to be the message.
 *
 * There is deliberately no context or register slot: the model card format has
 * neither, which is why `TranslateGemmaProvider` reports `contextHint: false`
 * and `formality: false`. Inventing slots here would make the chat layer build
 * on sand.
 */
export function buildInstructionTunedPrompt(text: string, o: TranslateOptions): string {
  const src = o.sourceLang ?? '';
  const sourceName = languageName(src);
  const targetName = languageName(o.targetLang);
  return (
    `You are a professional ${sourceName} (${src}) to ${targetName} (${o.targetLang}) ` +
    `translator. Your goal is to accurately convey the meaning and nuances of the ` +
    `original ${sourceName} text while adhering to ${targetName} grammar, vocabulary, and ` +
    `cultural sensitivities.\n` +
    `Produce only the ${targetName} translation, without any additional explanations or ` +
    `commentary. Please translate the following ${sourceName} text into ${targetName}:\n\n\n${text}`
  );
}

export class TranslateGemmaProvider extends BaseTranslationProvider {
  readonly name = 'translategemma';
  private readonly host: string;
  private readonly model: string;

  constructor(config: TranslateGemmaConfig = {}) {
    super();
    this.host = config.host ?? 'http://localhost:11434';
    this.model = config.model ?? 'translategemma:4b';
  }

  capabilities(): ProviderCapabilities {
    return {
      // The raw model cannot do these as features the layer can RELY on,
      // so they stay false. The chat layer reads this and compensates — e.g.
      // since autoDetectSource is false, it runs its own language detector
      // before calling translate() on this backend.
      autoDetectSource: false,
      formality: false, // best-effort prompt hint only; not dependable
      glossaries: false, // no grammatical adaptation
      contextHint: false, // the model-card prompt has no context slot
      nativeBatch: false, // Ollama is one-shot; base class loops
    };
  }

  async translate(text: string, options: TranslateOptions): Promise<TranslateResult> {
    if (!options.sourceLang) {
      // Fail loudly rather than guess silently. capabilities() already told
      // the caller this backend can't auto-detect, so a missing sourceLang
      // is a bug in the layer above — it should have detected first.
      throw new TranslationError(
        'translategemma requires an explicit sourceLang (no auto-detect)',
        this.name,
      );
    }

    const prompt = buildInstructionTunedPrompt(text, options);
    let res: Response;
    try {
      res = await fetch(`${this.host}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.model,
          stream: false,
          messages: [{ role: 'user', content: prompt }],
          options: { temperature: 0 }, // fidelity over flair
        }),
      });
    } catch (cause) {
      throw new TranslationError('Ollama request failed', this.name, cause, true);
    }

    if (!res.ok) {
      throw new TranslationError(
        `Ollama returned ${res.status}`,
        this.name,
        await safeText(res),
        res.status >= 500,
      );
    }

    const json = (await res.json()) as { message?: { content?: string } };
    const out = json.message?.content?.trim();
    if (!out) {
      throw new TranslationError('Empty translation from model', this.name);
    }
    return { text: out, detectedSourceLang: options.sourceLang };
  }

  async healthCheck(): Promise<boolean> {
    try {
      const res = await fetch(`${this.host}/api/tags`);
      if (!res.ok) return false;
      const { models } = (await res.json()) as { models: { name: string }[] };
      return models.some((m) => m.name === this.model);
    } catch {
      return false;
    }
  }
}

// ─────────────────────────────────────────────────────────────
// 6. The per-deployment switch
// ─────────────────────────────────────────────────────────────

export type ProviderName = 'deepl' | 'translategemma';

export interface DeploymentConfig {
  provider: ProviderName;
  deepl?: DeepLConfig;
  translategemma?: TranslateGemmaConfig;
}

/**
 * Chosen ONCE at startup. One customer's deployment runs DeepL (best European
 * quality, zero infra); the privacy-maximalist who reads "private intelligence"
 * literally runs the local model (nothing leaves their walls). Nothing
 * downstream knows or cares which one it got — that's the whole point.
 */
export function createProvider(config: DeploymentConfig): TranslationProvider {
  switch (config.provider) {
    case 'deepl':
      if (!config.deepl?.apiKey) {
        throw new Error('deepl provider selected but no apiKey configured');
      }
      return new DeepLProvider(config.deepl);
    case 'translategemma':
      return new TranslateGemmaProvider(config.translategemma ?? {});
    default:
      throw new Error(`unknown translation provider: ${(config as { provider: string }).provider}`);
  }
}

/** Convenience: build config from environment for a 12-factor deployment. */
export function configFromEnv(
  env: Record<string, string | undefined> = process.env,
): DeploymentConfig {
  const provider = (env.TRANSLATION_PROVIDER ?? 'deepl') as ProviderName;
  return {
    provider,
    deepl: { apiKey: env.DEEPL_API_KEY ?? '' },
    translategemma: {
      host: env.OLLAMA_HOST,
      model: env.OLLAMA_MODEL,
    },
  };
}

// ─────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────

async function safeText(res: Response): Promise<string> {
  try {
    return await res.text();
  } catch {
    return '';
  }
}

/* ─────────────────────────────────────────────────────────────
 * Sketch of how the chat layer ABOVE this boundary uses it. This is
 * illustrative — the real masking/cache/fan-out code is a separate layer.
 *
 *   const provider = createProvider(configFromEnv());
 *
 *   async function translateForViewer(raw: string, target: LanguageCode) {
 *     const caps = provider.capabilities();
 *
 *     // 1. Mask things that must NOT be translated (mentions, URLs, code…).
 *     const { masked, restore } = maskNonTranslatable(raw);
 *
 *     // 2. Capability-driven degradation: detect source ourselves when the
 *     //    backend can't, so both providers behave identically upstream.
 *     const sourceLang = caps.autoDetectSource ? null : await detectLang(masked);
 *
 *     // 3. Translate (cache around this call — identical messages recur).
 *     const { text } = await provider.translate(masked, {
 *       sourceLang,
 *       targetLang: target,
 *       formality: caps.formality ? 'prefer_less' : undefined,
 *       context: recentMessages.join('\n'), // both backends accept this
 *     });
 *
 *     // 4. Re-insert the masked tokens.
 *     return restore(text);
 *   }
 * ───────────────────────────────────────────────────────────── */
