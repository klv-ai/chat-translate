import type { FastifyInstance } from 'fastify';
import type { Translators } from './provider';
import {
  capabilitiesSchema,
  type RoomBody,
  roomSchema,
  type TranslateBody,
  translateSchema,
} from './schemas';
import { serializeRoomTranslation } from './serialize';

/** Translation-only HTTP surface over the SDK's chat/room translators. */
export function registerRoutes(app: FastifyInstance, t: Translators): void {
  app.get('/health', async (_req, reply) => {
    const ok = await t.provider.healthCheck();
    reply.code(ok ? 200 : 503);
    return { status: ok ? 'ok' : 'degraded', provider: t.provider.name };
  });

  app.get('/capabilities', { schema: capabilitiesSchema }, async () => ({
    provider: t.provider.name,
    ...t.room.capabilities,
  }));

  app.post<{ Body: TranslateBody }>('/translate', { schema: translateSchema }, async (req) => {
    const { text, targetLang, sourceLang, formality, context, glossaryId } = req.body;
    return t.chat.translateForViewer(text, targetLang, {
      sourceLang,
      formality,
      context,
      glossaryId,
    });
  });

  app.post<{ Body: RoomBody }>('/translate/room', { schema: roomSchema }, async (req) => {
    const { text, viewedLanguages, sourceLang, formality, context, glossaryId } = req.body;
    const out = await t.room.translateForRoom(text, viewedLanguages, {
      sourceLang,
      formality,
      context,
      glossaryId,
    });
    return serializeRoomTranslation(out);
  });
}
