import { TranslationError } from '@chat-translate/sdk';
import Fastify, { type FastifyInstance } from 'fastify';
import type { Translators } from './provider';
import { registerRoutes } from './routes';

export interface BuildAppOptions {
  translators: Translators;
  logLevel?: string;
}

/**
 * Build a configured Fastify instance. Kept separate from process bootstrap so
 * tests can drive it with `app.inject()` and a fake provider.
 */
export function buildApp(opts: BuildAppOptions): FastifyInstance {
  const app = Fastify({ logger: { level: opts.logLevel ?? 'info' } });

  app.setErrorHandler((err, req, reply) => {
    // Provider failures: surface a clear upstream status + retry hint.
    if (err instanceof TranslationError) {
      reply.code(err.retryable ? 503 : 502).send({
        error: 'translation_failed',
        provider: err.provider,
        message: err.message,
        retryable: err.retryable,
      });
      return;
    }
    // Schema validation (and other client errors) carry a 4xx statusCode.
    const statusCode =
      err && typeof err === 'object' && 'statusCode' in err && typeof err.statusCode === 'number'
        ? err.statusCode
        : undefined;
    const message = err instanceof Error ? err.message : 'bad request';
    if (statusCode !== undefined && statusCode >= 400 && statusCode < 500) {
      reply.code(statusCode).send({ error: 'bad_request', message });
      return;
    }
    req.log.error(err);
    reply.code(500).send({ error: 'internal_error', message: 'internal server error' });
  });

  registerRoutes(app, opts.translators);
  return app;
}
