# chat-translate (Python)

A standalone, pip-installable **chat translation SDK** — the Python port of [`../ts`](../ts)'s `@chat-translate/sdk`. Everyone in a room writes in their own language; the system auto-translates to every language people are reading, preserving @mentions, URLs, emoji, and code.

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

`llama-cpp-python` is an optional extra: the calling app usually owns a `llama_cpp.Llama` instance, which you **inject** into `LocalGGUFProvider`. Install the extra only if you want the SDK to build the model itself.

## Quick start

```python
from chat_translate import (
    create_room_translator, create_provider, config_from_env,
    InMemoryTranslationCache, eld_language_detector,
)
from eld import LanguageDetector

provider = create_provider(config_from_env())          # deepl | local from env
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
from chat_translate import LocalGGUFProvider, create_room_translator, eld_language_detector
from eld import LanguageDetector

llama = Llama(model_path="model_files/translategemma-4b.gguf")   # owned by your app
provider = LocalGGUFProvider(llama=llama)                        # injected — no [local] extra needed
detector = eld_language_detector(LanguageDetector(), fallback="en")
room = create_room_translator(provider, detector=detector)       # detector is load-bearing here
```

The local backend can't auto-detect a source language, so a detector is **required** for it (and a useful optimization for DeepL).

## Configuration (env)

| Var | Default | Notes |
| --- | --- | --- |
| `TRANSLATION_PROVIDER` | `deepl` | `deepl` \| `local` \| `ollama` |
| `DEEPL_API_KEY` | – | required for DeepL; `:fx` suffix → free host |
| `MODEL_PATH` | – | path to a `.gguf` (under `model_files/`) for the local provider |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama daemon for the `ollama` provider |
| `OLLAMA_MODEL` | `translategemma:4b` | model tag the Ollama provider requests |

## Develop

```sh
cd python
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
ruff check . && ruff format --check .
mypy src/chat_translate
pytest
```

The test suite mirrors the TS SDK's: masking round-trip, source-resolution precedence, cache LRU/TTL, sequential batch, the providers (DeepL via a mock transport, local GGUF via an injected fake llama), and room fan-out. CI needs no native build — the GGUF tests inject a fake llama.

## License

MIT. The local backend uses Google's Gemma model, governed by the [Gemma Terms of Use](../GEMMA-LICENSE.md).
