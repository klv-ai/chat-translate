"""Helpers shared by the CLI subcommands."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..provider.base import CompletionProvider, TranslationProvider
from ..provider.factory import (
    completion_config_from_env,
    config_from_env,
    create_completion_provider,
    create_provider,
)


def translation_provider_from_env() -> TranslationProvider:
    """Build the translation provider from the environment, exiting on bad config."""
    try:
        return create_provider(config_from_env())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def completion_provider_from_env() -> CompletionProvider:
    """Build the completion provider from the environment, exiting on bad config."""
    try:
        return create_completion_provider(completion_config_from_env())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None


def load_env_file(path: Path) -> None:
    """Set ``KEY=VALUE`` lines from *path* into ``os.environ``.

    Variables already set in the environment take precedence. Blank lines and
    ``#`` comments are skipped; surrounding quotes on a value are removed.
    """
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(name, value)


def read_catalog(path: Path) -> dict[str, str]:
    """Read a flat JSON object of strings."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def write_catalog(path: Path, catalog: dict[str, str]) -> None:
    """Write *catalog* as indented JSON via a temporary file and rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
