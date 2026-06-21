"""Provider abstraction for the chat translation SDK.

The provider boundary sits as low as possible: it knows how to translate a
string and nothing about chat. Masking, caching, fan-out, and source resolution
all live above this interface, written once and reused regardless of backend.

`capabilities()` is the contract that lets the chat layer degrade gracefully
instead of assuming the two backends (DeepL, local GGUF) are interchangeable.
The switch is per-deployment (static config) — see `create_provider` /
`config_from_env`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast, get_args, runtime_checkable

import httpx

LanguageCode = str

# Register / politeness control, mapping onto DeepL's `formality` parameter.
# Prefer the ``prefer_*`` variants: plain ``more``/``less`` make DeepL error on
# languages that do not support formality, whereas ``prefer_*`` silently no-op.
Formality = Literal["default", "more", "less", "prefer_more", "prefer_less"]

ProviderName = Literal["deepl", "local"]


@dataclass(slots=True)
class TranslateOptions:
    target_lang: LanguageCode
    #: Omit / None = let the provider auto-detect — IF capabilities allow it.
    source_lang: LanguageCode | None = None
    formality: Formality | None = None
    glossary_id: str | None = None
    #: Surrounding text used to disambiguate, WITHOUT being translated.
    context: str | None = None


@dataclass(slots=True)
class TranslateResult:
    text: str
    detected_source_lang: LanguageCode


@dataclass(slots=True)
class ProviderCapabilities:
    """A flag is True only if the chat layer can DEPEND on it.

    Best-effort approximations (e.g. coaxing a raw model toward a register via
    prompt) stay False so nothing builds on sand.
    """

    auto_detect_source: bool
    formality: bool
    glossaries: bool
    context_hint: bool
    native_batch: bool
    max_batch_size: int | None = None


class TranslationError(Exception):
    def __init__(
        self,
        message: str,
        provider: str,
        *,
        cause: object | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.cause = cause
        self.retryable = retryable


class TranslationProvider(ABC):
    name: str

    @abstractmethod
    def capabilities(self) -> ProviderCapabilities: ...

    @abstractmethod
    def translate(self, text: str, options: TranslateOptions) -> TranslateResult: ...

    @abstractmethod
    def translate_batch(
        self, texts: list[str], options: TranslateOptions
    ) -> list[TranslateResult]: ...

    @abstractmethod
    def health_check(self) -> bool:
        """Cheap liveness probe — matters most for the self-hosted path."""


class BaseTranslationProvider(TranslationProvider):
    """Shared scaffolding. The important default is `translate_batch`: providers
    without a native batch endpoint get a correct, if slower, sequential
    implementation for free. Sequential (not parallel) is deliberate for the
    self-hosted path: a single GPU does not benefit from being hammered.
    """

    def translate_batch(self, texts: list[str], options: TranslateOptions) -> list[TranslateResult]:
        return [self.translate(t, options) for t in texts]


# ─────────────────────────────────────────────────────────────
# DeepL adapter (managed pass-through)
# ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DeepLConfig:
    api_key: str
    #: Override the host if needed; auto-selected from the key by default.
    api_url: str | None = None


class DeepLProvider(BaseTranslationProvider):
    name = "deepl"

    def __init__(self, config: DeepLConfig, *, client: httpx.Client | None = None) -> None:
        # Free keys end in ":fx" and must hit the free host.
        is_free = config.api_key.endswith(":fx")
        self._endpoint = config.api_url or (
            "https://api-free.deepl.com/v2" if is_free else "https://api.deepl.com/v2"
        )
        self._api_key = config.api_key
        self._client = client or httpx.Client(timeout=30.0)

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=True,
            formality=True,
            glossaries=True,
            context_hint=True,
            native_batch=True,
            max_batch_size=50,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        return self.translate_batch([text], options)[0]

    def translate_batch(self, texts: list[str], options: TranslateOptions) -> list[TranslateResult]:
        body: dict[str, Any] = {"text": texts, "target_lang": options.target_lang}
        # A glossary requires an explicit source_lang (can't auto-detect).
        if options.source_lang:
            body["source_lang"] = options.source_lang
        if options.formality and options.formality != "default":
            body["formality"] = options.formality
        if options.glossary_id:
            body["glossary_id"] = options.glossary_id
        if options.context:
            body["context"] = options.context

        try:
            res = self._client.post(
                f"{self._endpoint}/translate",
                json=body,
                headers={"Authorization": f"DeepL-Auth-Key {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise TranslationError(
                "DeepL request failed", self.name, cause=exc, retryable=True
            ) from exc

        if not res.is_success:
            retryable = res.status_code == 429 or res.status_code >= 500
            raise TranslationError(
                f"DeepL returned {res.status_code}", self.name, cause=res.text, retryable=retryable
            )

        data = res.json()
        return [
            TranslateResult(
                text=t["text"], detected_source_lang=str(t["detected_source_language"]).lower()
            )
            for t in data["translations"]
        ]

    def health_check(self) -> bool:
        try:
            res = self._client.get(
                f"{self._endpoint}/usage",
                headers={"Authorization": f"DeepL-Auth-Key {self._api_key}"},
            )
            return res.is_success
        except httpx.HTTPError:
            return False


# ─────────────────────────────────────────────────────────────
# Local GGUF adapter (self-hosted, via llama.cpp)
# ─────────────────────────────────────────────────────────────


@runtime_checkable
class LlamaLike(Protocol):
    """The slice of the llama_cpp.Llama API this provider relies on."""

    def create_chat_completion(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


class LocalGGUFProvider(BaseTranslationProvider):
    """TranslateGemma-style backend backed by a local GGUF via llama.cpp.

    The llama runtime is INJECTABLE: pass a ready ``llama`` instance (the usual
    case — the calling app owns it), or a ``model_path`` to have the provider
    build one lazily via the optional ``[local]`` extra.
    """

    name = "local"

    def __init__(
        self,
        *,
        llama: LlamaLike | None = None,
        model_path: str | None = None,
        temperature: float = 0.0,
        **llama_kwargs: Any,
    ) -> None:
        self._temperature = temperature
        if llama is not None:
            self._llama: LlamaLike = llama
        elif model_path is not None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise TranslationError(
                    "llama-cpp-python is not installed; install the '[local]' extra "
                    "or inject a llama instance",
                    self.name,
                    cause=exc,
                ) from exc
            self._llama = cast(LlamaLike, Llama(model_path=model_path, **llama_kwargs))
        else:
            raise TranslationError(
                "LocalGGUFProvider requires either an injected `llama` or a `model_path`",
                self.name,
            )

    def capabilities(self) -> ProviderCapabilities:
        # The raw model cannot do these as features the layer can RELY on, so
        # they stay False. The chat layer reads this and compensates — since
        # auto_detect_source is False, it runs its own detector first.
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=True,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        if not options.source_lang:
            # Fail loudly rather than guess: capabilities() already told the
            # caller this backend can't auto-detect.
            raise TranslationError(
                "local GGUF provider requires an explicit source_lang (no auto-detect)",
                self.name,
            )
        prompt = self._build_prompt(text, options)
        try:
            res = self._llama.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=self._temperature,
            )
            out = str(res["choices"][0]["message"]["content"] or "").strip()
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001 - wrap any llama/runtime failure uniformly
            raise TranslationError(
                "llama.cpp inference failed", self.name, cause=exc, retryable=True
            ) from exc
        if not out:
            raise TranslationError("Empty translation from model", self.name)
        return TranslateResult(text=out, detected_source_lang=options.source_lang)

    def _build_prompt(self, text: str, o: TranslateOptions) -> str:
        if o.formality in ("more", "prefer_more"):
            register = " Use a formal register."
        elif o.formality in ("less", "prefer_less"):
            register = " Use an informal register."
        else:
            register = ""
        ctx = (
            f"\nConversation context (for disambiguation only, do not translate):\n{o.context}\n"
            if o.context
            else ""
        )
        return (
            f"Translate the following text from {o.source_lang} to {o.target_lang}."
            f"{register} Output only the translation, with no preamble or quotes."
            f"{ctx}\n\n{text}"
        )

    def health_check(self) -> bool:
        return self._llama is not None


# ─────────────────────────────────────────────────────────────
# Per-deployment switch
# ─────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DeploymentConfig:
    provider: ProviderName
    deepl: DeepLConfig | None = None
    local_model_path: str | None = None


def create_provider(
    config: DeploymentConfig,
    *,
    llama: LlamaLike | None = None,
    client: httpx.Client | None = None,
) -> TranslationProvider:
    """Build the configured provider. `llama`/`client` allow the calling app to
    inject the runtime/HTTP client (the llama instance especially)."""
    if config.provider == "deepl":
        if config.deepl is None or not config.deepl.api_key:
            raise ValueError("deepl provider selected but no api_key configured")
        return DeepLProvider(config.deepl, client=client)
    if config.provider == "local":
        return LocalGGUFProvider(llama=llama, model_path=config.local_model_path)
    raise ValueError(f"unknown translation provider: {config.provider!r}")


def config_from_env(env: Mapping[str, str] | None = None) -> DeploymentConfig:
    """Build config from environment for a 12-factor deployment."""
    e: Mapping[str, str] = os.environ if env is None else env
    raw = e.get("TRANSLATION_PROVIDER", "deepl")
    if raw not in get_args(ProviderName):
        raise ValueError(f"unknown TRANSLATION_PROVIDER: {raw!r}")
    provider = cast(ProviderName, raw)
    return DeploymentConfig(
        provider=provider,
        deepl=DeepLConfig(api_key=e.get("DEEPL_API_KEY", "")),
        local_model_path=e.get("MODEL_PATH"),
    )
