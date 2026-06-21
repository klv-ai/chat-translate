import { describe, expect, it } from 'vitest';
import { maskNonTranslatable } from '../src/chat-translation-layer';

describe('maskNonTranslatable', () => {
  it('round-trips do-not-translate spans verbatim', () => {
    const raw = 'hey @alice see https://x.com/pr/42 lgtm 👍 `npm run build` :tada:';
    const m = maskNonTranslatable(raw);

    // The translatable surface no longer contains the protected spans.
    expect(m.masked).not.toContain('@alice');
    expect(m.masked).not.toContain('https://');
    expect(m.masked).not.toContain('👍');
    expect(m.tokenCount).toBe(5); // mention, url, emoji, code span, shortcode

    // Restoring the (untranslated) masked form reproduces the original exactly.
    expect(m.restore(m.masked)).toBe(raw);
    expect(m.findUnrestored(m.masked)).toEqual([]);
  });

  it('reports tokens the engine dropped', () => {
    const m = maskNonTranslatable('ping @bob now');
    expect(m.tokenCount).toBe(1);

    // A translation that ate the sentinel: token 0 never reappears.
    const eaten = 'translated text without the placeholder';
    expect(m.findUnrestored(eaten)).toEqual([0]);
  });

  it('keeps trailing sentence punctuation outside the URL token', () => {
    const raw = 'see https://x.com/foo.';
    const m = maskNonTranslatable(raw);
    expect(m.restore(m.masked)).toBe(raw);
    expect(m.masked.endsWith('.')).toBe(true); // the period was trimmed out of the span
  });
});
