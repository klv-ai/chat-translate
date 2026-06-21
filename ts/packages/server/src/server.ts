import { buildApp } from './app';
import { loadConfig } from './config';
import { buildTranslators } from './provider';

async function main(): Promise<void> {
  const cfg = loadConfig();
  const translators = buildTranslators(cfg);
  const app = buildApp({ translators, logLevel: cfg.logLevel });

  const shutdown = (signal: string): void => {
    app.log.info({ signal }, 'shutting down');
    app
      .close()
      .then(() => process.exit(0))
      .catch(() => process.exit(1));
  };
  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  try {
    await app.listen({ host: cfg.host, port: cfg.port });
  } catch (err) {
    app.log.error(err);
    process.exit(1);
  }
}

void main();
