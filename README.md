# chat-translate

Provider-agnostic translation for short text: chat messages, message boards and UI strings. Everyone in a room writes in their own language, and each message is translated into every language people are reading. @mentions, URLs, emoji and code pass through untouched.

The same design is implemented twice, as independent packages:

| Package | What it is | Docs |
| --- | --- | --- |
| **TypeScript** — `@chat-translate/sdk` + `@chat-translate/server` | An SDK with no runtime dependencies, plus a thin Fastify HTTP server packaged as a Docker image | [ts/README.md](ts/README.md) |
| **Python** — `chat-translate` | A pip-installable SDK with a bundled offline language detector, plus a `chat-translate` CLI for ICU message catalogs | [python/README.md](python/README.md) |

## Features

- **One provider interface, chosen per deployment from the environment:**
  - [DeepL](https://www.deepl.com/pro-api) (managed)
  - Google's TranslateGemma, self-hosted via [Ollama](https://ollama.com)
  - A local GGUF file via `llama-cpp-python` (Python only)
- **Do-not-translate masking:** @mentions, URLs, emoji and code survive translation.
- **Per-room fan-out with a cache:** a message is translated once per target language, not once per reader. Readers of the source language are skipped.
- **Source-language resolution at ingest:** each message gets a concrete source language, which is required for backends that can't auto-detect.
- **Capability flags:** each provider declares what it supports (auto-detect, formality, context), so callers degrade gracefully rather than assuming backends are interchangeable.
- **ICU MessageFormat catalogs (Python):** placeholders and plural structure are preserved and every result is verified, with optional LLM review for meaning and plural-form expansion.

## Quick start

```sh
# TypeScript HTTP server in Docker (DeepL)
cd ts && cp .env.example .env    # set DEEPL_API_KEY
docker compose up --build

# Python SDK
pip install ./python
```

See each package's README for SDK usage, configuration and self-hosting.

## Choosing a backend

- **DeepL** is the simplest option. The paid plan offers unlimited characters; a free development tier is capped at 500,000 characters per month. Text is sent to DeepL's API.
- **TranslateGemma**, self-hosted via Ollama or a GGUF file, keeps text on your own infrastructure. It can't detect the source language, so a language detector is required; both SDKs include one.

## Repository layout

```
ts/            TypeScript SDK + HTTP server (pnpm workspace, Dockerfile, docker-compose)
python/        Python SDK + CLI
model_files/   Local GGUF weights (git-ignored)
```

## License

The code is released under the [MIT License](LICENSE).

The self-hosted backends run Google's Gemma models, which are governed by the [Gemma Terms of Use](GEMMA-LICENSE.md). You must accept those terms to use the models.
