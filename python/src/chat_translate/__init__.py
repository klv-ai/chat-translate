"""chat_translate — provider-agnostic chat translation SDK.

DeepL / local-GGUF backends behind one interface, do-not-translate masking,
per-room fan-out + cache, and ingest-time source resolution.
"""

from __future__ import annotations

from .chat import (
    ChatTranslator,
    LanguageDetector,
    ViewerTranslateOptions,
    ViewerTranslation,
    create_chat_translator,
)
from .detector import eld_language_detector, to_language_detector
from .fanout import (
    CachedTranslation,
    InMemoryTranslationCache,
    RoomStats,
    RoomTranslation,
    RoomTranslator,
    TranslationCache,
    create_room_translator,
)
from .masking import (
    DEFAULT_RULES,
    MaskedMessage,
    MaskRule,
    mask_non_translatable,
)
from .provider import (
    BaseTranslationProvider,
    DeepLConfig,
    DeepLProvider,
    DeploymentConfig,
    Formality,
    LanguageCode,
    LlamaLike,
    LocalGGUFProvider,
    ProviderCapabilities,
    ProviderName,
    TranslateOptions,
    TranslateResult,
    TranslationError,
    TranslationProvider,
    config_from_env,
    create_provider,
)
from .resolve_source import (
    DEFAULT_SHORT_TEXT_LETTERS,
    DEFAULT_SHORT_TEXT_POLICY,
    ConfidenceDetector,
    DetectionResult,
    EldDetectorLike,
    ResolvedSource,
    ResolveSourceOptions,
    ShortTextPolicy,
    SourceProvenance,
    eld_detector,
    letter_count,
    resolve_source,
)

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_RULES",
    "DEFAULT_SHORT_TEXT_LETTERS",
    "DEFAULT_SHORT_TEXT_POLICY",
    "BaseTranslationProvider",
    "CachedTranslation",
    "ChatTranslator",
    "ConfidenceDetector",
    "DeepLConfig",
    "DeepLProvider",
    "DeploymentConfig",
    "DetectionResult",
    "EldDetectorLike",
    "Formality",
    "InMemoryTranslationCache",
    "LanguageCode",
    "LanguageDetector",
    "LlamaLike",
    "LocalGGUFProvider",
    "MaskRule",
    "MaskedMessage",
    "ProviderCapabilities",
    "ProviderName",
    "ResolveSourceOptions",
    "ResolvedSource",
    "RoomStats",
    "RoomTranslation",
    "RoomTranslator",
    "ShortTextPolicy",
    "SourceProvenance",
    "TranslateOptions",
    "TranslateResult",
    "TranslationCache",
    "TranslationError",
    "TranslationProvider",
    "ViewerTranslateOptions",
    "ViewerTranslation",
    "config_from_env",
    "create_chat_translator",
    "create_provider",
    "create_room_translator",
    "eld_detector",
    "eld_language_detector",
    "letter_count",
    "mask_non_translatable",
    "resolve_source",
    "to_language_detector",
]
