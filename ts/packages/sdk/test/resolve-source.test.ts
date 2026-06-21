import { describe, expect, it } from 'vitest';
import {
  type ConfidenceDetector,
  type DetectionResult,
  resolveSource,
} from '../src/resolve-source';

const stub = (r: DetectionResult): ConfidenceDetector => ({ detect: () => r });

describe('resolveSource precedence', () => {
  it('treats a user override as ground truth without detecting', async () => {
    const detector: ConfidenceDetector = {
      detect: () => {
        throw new Error('should not be called when an override is present');
      },
    };
    const out = await resolveSource('whatever', detector, { uiLang: 'en', override: 'fr' });
    expect(out).toEqual({ lang: 'fr', provenance: 'user_confirmed', confident: true });
  });

  it('trusts a long reliable detection outright, overriding the prior', async () => {
    const out = await resolveSource(
      'this is clearly written in english here',
      stub({ lang: 'en', reliable: true }),
      { uiLang: 'de' },
    );
    expect(out).toMatchObject({ lang: 'en', provenance: 'detected', confident: true });
  });

  it('falls back to the UI prior when detection is unreliable', async () => {
    const out = await resolveSource('xx', stub({ lang: 'en', reliable: false }), { uiLang: 'de' });
    expect(out).toMatchObject({
      lang: 'de',
      provenance: 'ui_fallback',
      confident: false,
      detected: 'en',
    });
  });

  it('short + agrees with prior → trusted', async () => {
    const out = await resolveSource('ok', stub({ lang: 'en', reliable: true }), { uiLang: 'en' });
    expect(out).toMatchObject({ lang: 'en', provenance: 'detected', confident: true });
  });

  it('short + disagrees + prefer_prior → keeps the UI prior, not confident', async () => {
    const out = await resolveSource('ok', stub({ lang: 'en', reliable: true }), {
      uiLang: 'it',
      shortTextPolicy: 'prefer_prior',
    });
    expect(out).toMatchObject({
      lang: 'it',
      provenance: 'ui_fallback',
      confident: false,
      detected: 'en',
    });
  });

  it('short + disagrees + trust_detector → takes the detection, not confident', async () => {
    const out = await resolveSource('ok', stub({ lang: 'en', reliable: true }), {
      uiLang: 'it',
      shortTextPolicy: 'trust_detector',
    });
    expect(out).toMatchObject({ lang: 'en', provenance: 'detected', confident: false });
  });
});
