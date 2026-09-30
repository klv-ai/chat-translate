import { describe, expect, it } from 'vitest';
import {
  buildInstructionTunedPrompt,
  languageName,
  TranslateGemmaProvider,
} from '../src/translation-provider';

/**
 * Pins the TS prompt to the published TranslateGemma model card — the same
 * string `build_instruction_tuned_prompt` produces in the Python package. A
 * prompt change must be made in both packages.
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
    expect(languageName('tet')).toBe('Tetum');
    // An unmapped code falls back to itself.
    expect(languageName('xx')).toBe('xx');
  });

  it('has no context or register slot, matching what capabilities() claims', () => {
    const withExtras = buildInstructionTunedPrompt('hi', {
      sourceLang: 'en',
      targetLang: 'fr',
      formality: 'prefer_more',
      context: 'a prior message',
    });

    expect(withExtras).not.toContain('a prior message');
    expect(withExtras).not.toContain('formal register');

    const caps = new TranslateGemmaProvider().capabilities();
    expect(caps.contextHint).toBe(false);
    expect(caps.formality).toBe(false);
  });
});
