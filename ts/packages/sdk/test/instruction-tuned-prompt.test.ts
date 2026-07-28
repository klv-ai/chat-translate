import { describe, expect, it } from 'vitest';
import {
  buildInstructionTunedPrompt,
  languageName,
  TranslateGemmaProvider,
} from '../src/translation-provider';

/**
 * These lock the TS prompt to the published TranslateGemma model card — the
 * same string `build_instruction_tuned_prompt` produces in the Python package.
 *
 * This file exists because the two implementations HAD drifted: TS shipped a
 * generic "Translate the following text from X to Y" placeholder, which changes
 * what the model does. If you change the prompt, change it in both packages and
 * update this expectation deliberately.
 */
describe('buildInstructionTunedPrompt', () => {
  it('matches the model card format verbatim', () => {
    const prompt = buildInstructionTunedPrompt('Delete folder', {
      sourceLang: 'en',
      targetLang: 'es',
    });

    expect(prompt).toBe(
      'You are a professional English (en) to Spanish (es) translator. ' +
        'Your goal is to accurately convey the meaning and nuances of the original ' +
        'English text while adhering to Spanish grammar, vocabulary, and cultural ' +
        'sensitivities.\n' +
        'Produce only the Spanish translation, without any additional explanations ' +
        'or commentary. Please translate the following English text into Spanish:\n\n\nDelete folder',
    );
  });

  it('names languages in prose, ignoring the region subtag', () => {
    expect(languageName('pt-BR')).toBe('Portuguese');
    expect(languageName('DE')).toBe('German');
    // An unmapped code still yields a usable prompt rather than throwing.
    expect(languageName('tet')).toBe('tet');
  });

  it('has no context or register slot, matching what capabilities() claims', () => {
    const withExtras = buildInstructionTunedPrompt('hi', {
      sourceLang: 'en',
      targetLang: 'fr',
      formality: 'prefer_more',
      context: 'a prior message',
    });

    // The model card format has neither slot. Reporting them as unsupported and
    // then silently dropping them is the honest pairing.
    expect(withExtras).not.toContain('a prior message');
    expect(withExtras).not.toContain('formal register');

    const caps = new TranslateGemmaProvider().capabilities();
    expect(caps.contextHint).toBe(false);
    expect(caps.formality).toBe(false);
  });
});
