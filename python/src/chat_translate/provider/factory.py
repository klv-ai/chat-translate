"""Per-deployment provider switch (chosen once at startup)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast, get_args

import httpx

from .base import CompletionProvider, TranslationProvider
from .deepl import DeepLConfig, DeepLProvider
from .local import LlamaLike, LocalInstructionTunedProvider
from .ollama import (
    OllamaCompletionConfig,
    OllamaCompletionProvider,
    OllamaConfig,
    OllamaProvider,
)

ProviderName = Literal["deepl", "local", "ollama"]
CompletionProviderName = Literal["ollama"]

_DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass(slots=True)
class DeploymentConfig:
    provider: ProviderName
    deepl: DeepLConfig | None = None
    local_model_path: str | None = None
    ollama: OllamaConfig | None = None


@dataclass(slots=True)
class CompletionConfig:
    provider: CompletionProviderName
    ollama: OllamaCompletionConfig | None = None


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
        # The SDK targets instruction-tuned TranslateGemma GGUFs, whose rich chat
        # template needs structured content (see LocalInstructionTunedProvider).
        return LocalInstructionTunedProvider(llama=llama, model_path=config.local_model_path)
    if config.provider == "ollama":
        return OllamaProvider(config.ollama, client=client)
    raise ValueError(f"unknown translation provider: {config.provider!r}")


def create_completion_provider(
    config: CompletionConfig, *, client: httpx.Client | None = None
) -> CompletionProvider:
    """Build the configured general instruct model (see ``CompletionProvider``)."""
    if config.provider == "ollama":
        if config.ollama is None:
            raise ValueError("ollama completion provider selected but not configured")
        return OllamaCompletionProvider(config.ollama, client=client)
    raise ValueError(f"unknown completion provider: {config.provider!r}")


def config_from_env(env: Mapping[str, str] | None = None) -> DeploymentConfig:
    """Build translation config from environment for a 12-factor deployment."""
    e: Mapping[str, str] = os.environ if env is None else env
    raw = e.get("TRANSLATION_PROVIDER", "deepl")
    if raw not in get_args(ProviderName):
        raise ValueError(f"unknown TRANSLATION_PROVIDER: {raw!r}")
    provider = cast(ProviderName, raw)
    defaults = OllamaConfig()
    return DeploymentConfig(
        provider=provider,
        deepl=DeepLConfig(api_key=e.get("DEEPL_API_KEY", "")),
        local_model_path=e.get("MODEL_PATH"),
        ollama=OllamaConfig(
            host=e.get("OLLAMA_HOST", defaults.host),
            model=e.get("OLLAMA_MODEL", defaults.model),
            timeout_seconds=_float(e, "OLLAMA_TIMEOUT_SECONDS", defaults.timeout_seconds),
            keep_alive=e.get("OLLAMA_KEEP_ALIVE", defaults.keep_alive),
            max_output_tokens=_optional_int(e, "OLLAMA_MAX_OUTPUT_TOKENS"),
        ),
    )


def completion_config_from_env(env: Mapping[str, str] | None = None) -> CompletionConfig:
    """Build completion config from environment. ``COMPLETION_MODEL`` is required."""
    e: Mapping[str, str] = os.environ if env is None else env
    raw = e.get("COMPLETION_PROVIDER", "ollama")
    if raw not in get_args(CompletionProviderName):
        raise ValueError(f"unknown COMPLETION_PROVIDER: {raw!r}")
    model = e.get("COMPLETION_MODEL")
    if not model:
        raise ValueError("COMPLETION_MODEL is not set")
    defaults = OllamaCompletionConfig(model=model)
    return CompletionConfig(
        provider=cast(CompletionProviderName, raw),
        ollama=OllamaCompletionConfig(
            model=model,
            host=e.get("COMPLETION_HOST") or e.get("OLLAMA_HOST", defaults.host),
            timeout_seconds=_float(e, "COMPLETION_TIMEOUT_SECONDS", defaults.timeout_seconds),
            keep_alive=e.get("COMPLETION_KEEP_ALIVE", defaults.keep_alive),
        ),
    )


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def _optional_int(env: Mapping[str, str], name: str) -> int | None:
    raw = env.get(name)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
