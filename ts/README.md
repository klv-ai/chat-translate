# chat-translate (TypeScript)

A provider-agnostic **chat translation SDK** plus a **thin Fastify HTTP server**, in a pnpm-workspace monorepo. Everyone in a room writes in their own language; the system auto-translates to every language people are reading — preserving @mentions, URLs, emoji, and code.

- **`@chat-translate/sdk`** — the library. Zero runtime dependencies in core. Installable into any TypeScript project (ships `.d.ts`). DeepL and self-hosted TranslateGemma (Ollama) backends behind one interface; do-not-translate masking; per-room fan-out + cache; ingest source resolution; optional eval + capture tooling behind subpaths.
- **`@chat-translate/server`** — a thin HTTP API over the SDK, packaged as a Docker container. Picks the backend from the environment at boot.

The architecture and design rationale live in [`../claude-docs`](../claude-docs).

## Layout

```
ts/
├── packages/
│   ├── sdk/        @chat-translate/sdk      (library)
│   └── server/     @chat-translate/server   (Fastify app + Dockerfile)
├── docker-compose.yml        # server (+ optional ollama profile)
└── tsconfig.base.json, biome.json, pnpm-workspace.yaml
```

## Prerequisites

- Node.js **20+**
- pnpm 9 (via `corepack enable`, or `corepack pnpm ...`)

## Develop

```sh
pnpm install
pnpm typecheck     # tsc --noEmit, both packages
pnpm test          # vitest
pnpm build         # tsup → dist (ESM + .d.ts for the SDK)
pnpm lint          # biome
pnpm dev           # run the server with tsx watch
```

## Using the SDK in another project

```sh
# from your app — local install of the built package
pnpm add /path/to/ts/packages/sdk    # or a `pnpm pack` tarball
pnpm add eld                         # optional: only if you use a real detector
```

```ts
import { createProvider, configFromEnv, createRoomTranslator, InMemoryTranslationCache } from '@chat-translate/sdk';
import { eldLanguageDetector } from '@chat-translate/sdk/detector';
import { eld } from 'eld/large';

const provider = createProvider(configFromEnv());
const detector = eldLanguageDetector(eld, { subset: ['en', 'de', 'fr'], fallback: 'en' });
const room = createRoomTranslator(provider, { detector, cache: new InMemoryTranslationCache() });

const out = await room.translateForRoom('hey @alice ship it 🚀', ['de', 'fr', 'en']);
out.byLanguage.get('de')?.text;   // German, with @alice + 🚀 intact
out.detectedSourceLang;           // 'en'
```

Subpath entries (kept out of the core so it stays dependency-free):

| Import | Purpose |
| --- | --- |
| `@chat-translate/sdk` | provider, masking, chat/room translators, source resolution |
| `@chat-translate/sdk/detector` | ELD-backed `LanguageDetector` adapters (needs `eld`) |
| `@chat-translate/sdk/eval` | translation-quality eval harness |
| `@chat-translate/sdk/eval/source` | source-resolution calibration + policy sweep |
| `@chat-translate/sdk/logger` | live-traffic capture logger + Postgres sink (needs `pg`) |

## Running the server

```sh
cp .env.example .env     # set TRANSLATION_PROVIDER + provider settings
pnpm --filter @chat-translate/server build
node packages/server/dist/server.js
```

### HTTP API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `POST` | `/translate` | `{ text, targetLang, sourceLang?, formality?, context?, glossaryId? }` | `ViewerTranslation` |
| `POST` | `/translate/room` | `{ text, viewedLanguages[], ... }` | `{ detectedSourceLang, stats, byLanguage }` |
| `GET` | `/health` | – | `{ status, provider }` (503 when degraded) |
| `GET` | `/capabilities` | – | `{ provider, ...ProviderCapabilities }` |

```sh
curl -s localhost:8080/capabilities
curl -s -X POST localhost:8080/translate/room -H 'content-type: application/json' \
  -d '{"text":"hey @alice ship it 🚀","viewedLanguages":["de","fr","en"]}'
# → byLanguage is an OBJECT keyed by language code (the SDK's Map, serialized)
```

The eval harness and capture logger stay SDK-only and are **not** exposed over HTTP.

### Configuration (env)

| Var | Default | Notes |
| --- | --- | --- |
| `TRANSLATION_PROVIDER` | `deepl` | `deepl` \| `translategemma` |
| `DEEPL_API_KEY` | – | required for DeepL; `:fx` suffix → free host |
| `OLLAMA_HOST` | `http://localhost:11434` | TranslateGemma backend |
| `TRANSLATEGEMMA_MODEL` | `translategemma:4b` | |
| `HOST` / `PORT` | `0.0.0.0` / `8080` | |
| `DEFAULT_UI_LANG` | `en` | detector fallback when ELD has no opinion |
| `DETECTOR_SUBSET` | – | ISO 639-1, comma-separated; constrains the detector |
| `LOG_LEVEL` | `info` | Fastify/pino level |

The server always wires an ELD detector: it is **load-bearing** for TranslateGemma (which can't auto-detect a source) and an optimization for DeepL (same-language short-circuit + concrete cache keys).

## Docker

```sh
# DeepL deployment (server only) — set DEEPL_API_KEY in .env first
docker compose up --build

# Self-hosted TranslateGemma — also starts Ollama
#   set TRANSLATION_PROVIDER=translategemma and OLLAMA_HOST=http://ollama:11434 in .env
docker compose --profile selfhosted up --build
docker compose exec ollama ollama pull translategemma:4b   # one-time
```

The image is multi-stage (`packages/server/Dockerfile`), runs as a non-root user, and has a `/health` healthcheck.

## License

Code is MIT (see `package.json`). The self-hosted backend uses Google's Gemma model, governed by the [Gemma Terms of Use](../GEMMA-LICENSE.md) — users must agree to its terms.
