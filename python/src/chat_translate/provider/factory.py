"""Per-deployment provider switch (chosen once at startup)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast, get_args

import httpx

from .base import TranslationProvider
from .deepl import DeepLConfig, DeepLProvider
from .local import LlamaLike, LocalGGUFProvider
from .ollama import OllamaConfig, OllamaProvider

ProviderName = Literal["deepl", "local", "ollama"]


@dataclass(slots=True)
class DeploymentConfig:
    provider: ProviderName
    deepl: DeepLConfig | None = None
    local_model_path: str | None = None
    ollama: OllamaConfig | None = None


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
    if config.provider == "ollama":
        return OllamaProvider(config.ollama, client=client)
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
        ollama=OllamaConfig(
            host=e.get("OLLAMA_HOST", "http://localhost:11434"),
            model=e.get("OLLAMA_MODEL", "translategemma:4b"),
        ),
    )
