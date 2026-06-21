import { afterEach, describe, expect, it, vi } from 'vitest';
import { DeepLProvider } from '../src/translation-provider';

afterEach(() => vi.unstubAllGlobals());

const okResponse = (detected: string, text: string) =>
  new Response(JSON.stringify({ translations: [{ detected_source_language: detected, text }] }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });

describe('DeepLProvider', () => {
  it('uses the free host for :fx keys and sends a well-formed request', async () => {
    const fetchMock = vi.fn(async (url: unknown, init: unknown) => {
      expect(url).toBe('https://api-free.deepl.com/v2/translate');
      const opts = init as { body: string; headers: Record<string, string> };
      const body = JSON.parse(opts.body);
      expect(body.text).toEqual(['hallo']);
      expect(body.target_lang).toBe('en');
      expect(body.source_lang).toBe('de');
      expect(opts.headers.Authorization).toBe('DeepL-Auth-Key key-123:fx');
      return okResponse('DE', 'hello');
    });
    vi.stubGlobal('fetch', fetchMock);

    const p = new DeepLProvider({ apiKey: 'key-123:fx' });
    const r = await p.translate('hallo', { sourceLang: 'de', targetLang: 'en' });

    expect(r).toEqual({ text: 'hello', detectedSourceLang: 'de' }); // detected lang lowercased
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it('omits source_lang for paid keys when auto-detecting', async () => {
    const fetchMock = vi.fn(async (url: unknown, init: unknown) => {
      expect(url).toBe('https://api.deepl.com/v2/translate');
      const body = JSON.parse((init as { body: string }).body);
      expect('source_lang' in body).toBe(false);
      return okResponse('FR', 'hello');
    });
    vi.stubGlobal('fetch', fetchMock);

    const p = new DeepLProvider({ apiKey: 'paid-key' });
    const r = await p.translate('bonjour', { targetLang: 'en' });
    expect(r.detectedSourceLang).toBe('fr');
  });

  it('maps a 429 to a retryable TranslationError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response('rate limited', { status: 429 })),
    );
    const p = new DeepLProvider({ apiKey: 'paid-key' });
    await expect(p.translate('x', { targetLang: 'de' })).rejects.toMatchObject({
      name: 'TranslationError',
      provider: 'deepl',
      retryable: true,
    });
  });
});
