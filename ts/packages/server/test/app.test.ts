import {
  createChatTranslator,
  createRoomTranslator,
  type TranslationProvider,
} from '@chat-translate/sdk';
import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/app';
import type { Translators } from '../src/provider';

// A deterministic, network-free provider that auto-detects and echoes the
// target language back, so the routes + masking + serialization are exercised
// without DeepL or Ollama.
const fakeProvider: TranslationProvider = {
  name: 'fake',
  capabilities: () => ({
    autoDetectSource: true,
    formality: true,
    glossaries: false,
    contextHint: true,
    nativeBatch: true,
    maxBatchSize: 50,
  }),
  async translate(text, opts) {
    return { text: `[${opts.targetLang}] ${text}`, detectedSourceLang: opts.sourceLang ?? 'en' };
  },
  async translateBatch(texts, opts) {
    return Promise.all(texts.map((t) => this.translate(t, opts)));
  },
  async healthCheck() {
    return true;
  },
};

function fakeTranslators(): Translators {
  return {
    provider: fakeProvider,
    chat: createChatTranslator(fakeProvider),
    room: createRoomTranslator(fakeProvider),
  };
}

function app() {
  return buildApp({ translators: fakeTranslators(), logLevel: 'silent' });
}

describe('server routes', () => {
  it('GET /capabilities reports the provider capabilities', async () => {
    const a = app();
    const res = await a.inject({ method: 'GET', url: '/capabilities' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toMatchObject({
      provider: 'fake',
      autoDetectSource: true,
      nativeBatch: true,
    });
    await a.close();
  });

  it('GET /health reflects provider health', async () => {
    const a = app();
    const res = await a.inject({ method: 'GET', url: '/health' });
    expect(res.statusCode).toBe(200);
    expect(res.json()).toEqual({ status: 'ok', provider: 'fake' });
    await a.close();
  });

  it('POST /translate preserves masked spans through translation', async () => {
    const a = app();
    const res = await a.inject({
      method: 'POST',
      url: '/translate',
      payload: { text: 'hey @alice 👍', targetLang: 'de' },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.translated).toBe(true);
    expect(body.text).toContain('@alice');
    expect(body.text).toContain('👍');
    expect(body.unrestoredTokens).toEqual([]);
    await a.close();
  });

  it('POST /translate/room serializes the byLanguage Map into an object', async () => {
    const a = app();
    const res = await a.inject({
      method: 'POST',
      url: '/translate/room',
      payload: { text: 'hey @alice ship it', viewedLanguages: ['de', 'fr', 'de'] },
    });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(Object.keys(body.byLanguage).sort()).toEqual(['de', 'fr']);
    expect(body.byLanguage.de.text).toContain('@alice');
    expect(body.stats.targets).toBe(2); // distinct languages
    await a.close();
  });

  it('rejects an invalid body with 400', async () => {
    const a = app();
    const res = await a.inject({
      method: 'POST',
      url: '/translate',
      payload: { targetLang: 'de' }, // missing required `text`
    });
    expect(res.statusCode).toBe(400);
    await a.close();
  });
});
