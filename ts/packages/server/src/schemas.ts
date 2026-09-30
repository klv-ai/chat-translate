import type { Formality } from '@chat-translate/sdk';

/** POST /translate request body. */
export interface TranslateBody {
  text: string;
  targetLang: string;
  sourceLang?: string;
  formality?: Formality;
  context?: string;
  glossaryId?: string;
}

/** POST /translate/room request body. */
export interface RoomBody {
  text: string;
  viewedLanguages: string[];
  sourceLang?: string;
  formality?: Formality;
  context?: string;
  glossaryId?: string;
}

const FORMALITY = ['default', 'more', 'less', 'prefer_more', 'prefer_less'];

// Shared response fragment — mirrors ViewerTranslation. Used directly for
// /translate and as the value schema of the /translate/room language map, so
// Fastify's fast-json-stringify validates and trims both.
const viewerTranslation = {
  type: 'object',
  required: ['text', 'detectedSourceLang', 'translated', 'unrestoredTokens'],
  additionalProperties: false,
  properties: {
    text: { type: 'string' },
    detectedSourceLang: { type: 'string' },
    translated: { type: 'boolean' },
    unrestoredTokens: { type: 'array', items: { type: 'integer' } },
  },
};

const sharedBodyProps = {
  sourceLang: { type: 'string' },
  formality: { type: 'string', enum: FORMALITY },
  context: { type: 'string', maxLength: 8000 },
  glossaryId: { type: 'string' },
};

export const translateSchema = {
  body: {
    type: 'object',
    required: ['text', 'targetLang'],
    additionalProperties: false,
    properties: {
      text: { type: 'string', minLength: 1, maxLength: 8000 },
      targetLang: { type: 'string', minLength: 1 },
      ...sharedBodyProps,
    },
  },
  response: { 200: viewerTranslation },
};

export const roomSchema = {
  body: {
    type: 'object',
    required: ['text', 'viewedLanguages'],
    additionalProperties: false,
    properties: {
      text: { type: 'string', minLength: 1, maxLength: 8000 },
      viewedLanguages: {
        type: 'array',
        items: { type: 'string', minLength: 1 },
        minItems: 1,
      },
      ...sharedBodyProps,
    },
  },
  response: {
    200: {
      type: 'object',
      required: ['detectedSourceLang', 'stats', 'byLanguage'],
      properties: {
        detectedSourceLang: { type: 'string' },
        stats: {
          type: 'object',
          properties: {
            targets: { type: 'integer' },
            cacheHits: { type: 'integer' },
            translated: { type: 'integer' },
            shortCircuited: { type: 'integer' },
          },
        },
        byLanguage: { type: 'object', additionalProperties: viewerTranslation },
      },
    },
  },
};

export const capabilitiesSchema = {
  response: {
    200: {
      type: 'object',
      properties: {
        provider: { type: 'string' },
        autoDetectSource: { type: 'boolean' },
        formality: { type: 'boolean' },
        glossaries: { type: 'boolean' },
        contextHint: { type: 'boolean' },
        nativeBatch: { type: 'boolean' },
        maxBatchSize: { type: 'integer' },
      },
    },
  },
};
