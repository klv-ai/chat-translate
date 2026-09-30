import { configFromEnv, type DeploymentConfig, type LanguageCode } from '@chat-translate/sdk';

/** Everything the server needs to boot, derived once from the environment. */
export interface ServerConfig {
  /** Which translation backend + its settings (deepl | translategemma). */
  deployment: DeploymentConfig;
  host: string;
  port: number;
  /** Prior/fallback language used by the detector when it has no opinion. */
  defaultUiLang: LanguageCode;
  /** Constrain the detector to the room's plausible languages (ISO 639-1). */
  detectorSubset?: LanguageCode[];
  logLevel: string;
}

function parseList(value: string | undefined): LanguageCode[] | undefined {
  if (!value) return undefined;
  const items = value
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean);
  return items.length > 0 ? items : undefined;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): ServerConfig {
  return {
    deployment: configFromEnv(env),
    host: env.HOST ?? '0.0.0.0',
    port: Number.parseInt(env.PORT ?? '8080', 10),
    defaultUiLang: env.DEFAULT_UI_LANG ?? 'en',
    detectorSubset: parseList(env.DETECTOR_SUBSET),
    logLevel: env.LOG_LEVEL ?? 'info',
  };
}
