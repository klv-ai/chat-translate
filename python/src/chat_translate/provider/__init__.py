"""Provider package: shared contracts, the per-backend adapters, and the
per-deployment factory. Re-exported flat so ``from chat_translate.provider
import X`` (and the internal ``from .provider import X``) keep working."""

from __future__ import annotations

from .base import (
    LANGUAGE_NAMES,
    BaseTranslationProvider,
    CompletionError,
    CompletionProvider,
    ContentKind,
    Formality,
    LanguageCode,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    TranslationProvider,
    build_instruction_tuned_prompt,
    build_translation_prompt,
    build_ui_label_prompt,
    language_name,
)
from .deepl import DeepLConfig, DeepLProvider
from .factory import (
    CompletionConfig,
    CompletionProviderName,
    DeploymentConfig,
    ProviderName,
    completion_config_from_env,
    config_from_env,
    create_completion_provider,
    create_provider,
)
from .local import LlamaLike, LocalGGUFProvider, LocalInstructionTunedProvider
from .ollama import (
    OllamaCompletionConfig,
    OllamaCompletionProvider,
    OllamaConfig,
    OllamaProvider,
)

__all__ = [
    "LANGUAGE_NAMES",
    "BaseTranslationProvider",
    "CompletionConfig",
    "CompletionError",
    "CompletionProvider",
    "CompletionProviderName",
    "ContentKind",
    "DeepLConfig",
    "DeepLProvider",
    "DeploymentConfig",
    "Formality",
    "LanguageCode",
    "LlamaLike",
    "LocalGGUFProvider",
    "LocalInstructionTunedProvider",
    "OllamaCompletionConfig",
    "OllamaCompletionProvider",
    "OllamaConfig",
    "OllamaProvider",
    "ProviderCapabilities",
    "ProviderName",
    "TranslateOptions",
    "TranslateResult",
    "TranslationError",
    "TranslationProvider",
    "build_instruction_tuned_prompt",
    "build_translation_prompt",
    "build_ui_label_prompt",
    "completion_config_from_env",
    "config_from_env",
    "create_completion_provider",
    "create_provider",
    "language_name",
]
