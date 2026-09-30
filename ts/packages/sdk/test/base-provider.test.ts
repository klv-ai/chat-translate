import { describe, expect, it } from 'vitest';
import {
  BaseTranslationProvider,
  type ProviderCapabilities,
  type TranslateOptions,
  type TranslateResult,
} from '../src/translation-provider';

class StubProvider extends BaseTranslationProvider {
  readonly name = 'stub';
  calls: string[] = [];
  private inFlight = 0;
  maxInFlight = 0;

  capabilities(): ProviderCapabilities {
    return {
      autoDetectSource: true,
      formality: false,
      glossaries: false,
      contextHint: false,
      nativeBatch: false,
    };
  }

  async translate(text: string, _options: TranslateOptions): Promise<TranslateResult> {
    this.inFlight++;
    this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
    await new Promise((r) => setTimeout(r, 1));
    this.calls.push(text);
    this.inFlight--;
    return { text: text.toUpperCase(), detectedSourceLang: 'en' };
  }

  async healthCheck(): Promise<boolean> {
    return true;
  }
}

describe('BaseTranslationProvider.translateBatch', () => {
  it('translates sequentially and preserves order', async () => {
    const p = new StubProvider();
    const res = await p.translateBatch(['a', 'b', 'c'], { targetLang: 'de' });

    expect(res.map((r) => r.text)).toEqual(['A', 'B', 'C']);
    expect(p.calls).toEqual(['a', 'b', 'c']);
    expect(p.maxInFlight).toBe(1); // never hammered in parallel (single-GPU safe)
  });
});
