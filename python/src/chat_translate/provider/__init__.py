"""Provider package: shared contracts, the per-backend adapters, and the
per-deployment factory. Re-exported flat so ``from chat_translate.provider
import X`` (and the internal ``from .provider import X``) keep working."""

from __future__ import annotations

from .base import (
    BaseTranslationProvider,
    Formality,
    LanguageCode,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    TranslationProvider,
    build_translation_prompt,
)
from .deepl import DeepLConfig, DeepLProvider
from .factory import (
    DeploymentConfig,
    ProviderName,
    config_from_env,
    create_provider,
)
from .local import LlamaLike, LocalGGUFProvider
from .ollama import OllamaConfig, OllamaProvider

__all__ = [
    "BaseTranslationProvider",
    "DeepLConfig",
    "DeepLProvider",
    "DeploymentConfig",
    "Formality",
    "LanguageCode",
    "LlamaLike",
    "LocalGGUFProvider",
    "OllamaConfig",
    "OllamaProvider",
    "ProviderCapabilities",
    "ProviderName",
    "TranslateOptions",
    "TranslateResult",
    "TranslationError",
    "TranslationProvider",
    "build_translation_prompt",
    "config_from_env",
    "create_provider",
]
