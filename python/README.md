# chat-translate (Python)

A standalone, pip-installable **chat translation SDK**. It is the Python counterpart of the TypeScript [`@chat-translate/sdk`](../ts); see the [repository README](../README.md) for an overview. Everyone in a room writes in their own language; the system auto-translates to every language people are reading, preserving @mentions, URLs, emoji, and code.

Provider-agnostic behind one interface:
- **DeepL** (managed, via `httpx`)
- **Local GGUF** (self-hosted, via `llama-cpp-python`, reading a model from [`../model_files`](../model_files))
- **Ollama** (self-hosted, via the Ollama HTTP API — `httpx`, no extra dependency)

Unlike the TS SDK (which injected its detector), this package **bundles** the [ELD](https://pypi.org/project/eld/) language detector and `regex`, so a single install gives a working, offline-capable SDK.

## Install

```sh
pip install ./python              # core SDK: eld + regex + httpx
pip install './python[local]'     # + llama-cpp-python (native build) for the GGUF backend
```

`llama-cpp-python` is an optional extra: the calling app usually owns a `llama_cpp.Llama` instance, which you **inject** into the local provider. Install the extra only if you want the SDK to build the model itself.

## Quick start

```python
from chat_translate import (
    create_room_translator, create_provider, config_from_env,
    InMemoryTranslationCache, eld_language_detector,
)
from eld import LanguageDetector

provider = create_provider(config_from_env())          # deepl | local | ollama, from env
detector = eld_language_detector(LanguageDetector(), subset=["en", "de", "fr"], fallback="en")
room = create_room_translator(provider, detector=detector, cache=InMemoryTranslationCache())

out = room.translate_for_room("hey @alice ship it 🚀", ["de", "fr", "en"])
out.by_language["de"].text      # German, with @alice + 🚀 intact
out.detected_source_lang        # "en"
out.stats                       # RoomStats(targets=3, translated=2, short_circuited=1, ...)
```

### Local GGUF (inject the runtime)

```python
from llama_cpp import Llama
from chat_translate import LocalInstructionTunedProvider, create_room_translator, eld_language_detector
from eld import LanguageDetector

llama = Llama(model_path="model_files/translategemma-4b-it.gguf")  # owned by your app
provider = LocalInstructionTunedProvider(llama=llama)               # injected — no [local] extra needed
detector = eld_language_detector(LanguageDetector(), fallback="en")
room = create_room_translator(provider, detector=detector)       # detector is load-bearing here
```

The local backend can't auto-detect a source language, so a detector is **required** for it (and a useful optimization for DeepL).

`LocalInstructionTunedProvider` is for instruction-tuned TranslateGemma GGUFs, whose chat template builds the translation prompt from structured content; it is what `TRANSLATION_PROVIDER=local` selects. `LocalGGUFProvider` sends a plain instruction prompt, for GGUFs whose template just wraps a user message.

## Configuration (env)

Copy [`.env.example`](.env.example) to `.env` for a starting point. The SDK reads `os.environ` only; the CLI also loads `./.env`.

| Var | Default | Notes |
| --- | --- | --- |
| `TRANSLATION_PROVIDER` | `deepl` | `deepl` \| `local` \| `ollama` |
| `DEEPL_API_KEY` | – | required for DeepL; `:fx` suffix → free host |
| `MODEL_PATH` | – | path to a `.gguf` (under `model_files/`) for the local provider |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon for the `ollama` provider |
| `OLLAMA_MODEL` | `translategemma:4b` | model tag the Ollama provider requests |
| `OLLAMA_TIMEOUT_SECONDS` | `60` | per-request timeout |
| `OLLAMA_KEEP_ALIVE` | `5m` | how long Ollama keeps the model loaded after a request |
| `OLLAMA_MAX_OUTPUT_TOKENS` | – | cap on generated tokens (unset = server default) |

### Completion provider (catalog review and plural expansion)

`review_catalog` and `expand_message` take a `CompletionProvider`: a general instruct model, separate from the translation provider. Build one with `create_completion_provider(completion_config_from_env())`.

| Var | Default | Notes |
| --- | --- | --- |
| `COMPLETION_PROVIDER` | `ollama` | `ollama` |
| `COMPLETION_MODEL` | – | **required**; model tag of a general instruct model |
| `COMPLETION_HOST` | `OLLAMA_HOST`, else `http://localhost:11434` | |
| `COMPLETION_TIMEOUT_SECONDS` | `120` | per-request timeout |
| `COMPLETION_KEEP_ALIVE` | `5m` | |

## UI message catalogs

`translate_catalog` translates a natural-key catalog (each key is the source string) through any `TranslationProvider`. ICU arguments and plural skeletons are preserved, each result is checked by `verify`, and entries that fail are reported rather than returned. `review_catalog` and `expand_message` use a `CompletionProvider` to flag translations with the wrong meaning and to add plural categories the source language lacks.

```python
from chat_translate import (
    completion_config_from_env, config_from_env, create_completion_provider,
    create_provider, review_catalog, translate_catalog,
)

provider = create_provider(config_from_env())
result = translate_catalog(
    [
        "Delete folder",
        "Imported {count, plural, one {# file} other {# files}}"
    ],
    provider,
    "es-ES",
    protected=["Acme Notes"]
)
result.translated   # {key: translation} for entries that passed verification
result.failures     # EntryResult list with .error

llm = create_completion_provider(completion_config_from_env())
review = review_catalog(
    result.translated.items(),
    "es-ES",
    llm,
    context="Acme Notes is a note-taking app."
)
review.flagged
```

## CLI

The package installs a `chat-translate` command (also `python -m chat_translate.cli`) for working with catalog JSON files. It is a thin wrapper over the functions above and lives in `chat_translate.cli`; the SDK does not import it. Providers are built from the environment variables above, loaded from `./.env` if present (or `--env-file PATH`; variables already set take precedence).

```sh
# Translate en.json into es-ES.json (resumes: keys already in the output are skipped)
chat-translate catalog en.json es-ES [--out es-ES.json] [--protect terms.json] [--limit N] [--redo]

# Flag translations with the wrong meaning (needs COMPLETION_MODEL)
chat-translate review es-ES.json es-ES [--context product.txt] [--out flagged.json]

# Add plural categories the target language needs (edits in place; needs COMPLETION_MODEL)
chat-translate plurals ru-RU.json ru-RU [--dry-run]
```

`--protect` takes a JSON array of terms to keep verbatim, or `{"terms": [...]}`. `catalog` writes only verified translations and checkpoints to the output file as it goes; failed keys are listed and retried on the next run.

## Develop

```sh
cd python
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy src/chat_translate
pytest
```

The test suite mirrors the TS SDK's: masking round-trip, source-resolution precedence, cache LRU/TTL, sequential batch, the providers (DeepL via a mock transport, local GGUF via an injected fake llama), and room fan-out, plus ICU catalog handling and the CLI. CI needs no native build — the GGUF tests inject a fake llama.

## License

[MIT](../LICENSE). The local backend uses Google's Gemma model, governed by the [Gemma Terms of Use](../GEMMA-LICENSE.md).
