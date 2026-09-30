from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from chat_translate.cli import _common
from chat_translate.cli import catalog as catalog_cmd
from chat_translate.cli import plurals as plurals_cmd
from chat_translate.cli import review as review_cmd
from chat_translate.cli.__main__ import main
from chat_translate.provider.base import (
    BaseTranslationProvider,
    ProviderCapabilities,
    TranslateOptions,
    TranslateResult,
)


class UpperProvider(BaseTranslationProvider):
    name = "upper"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            auto_detect_source=False,
            formality=False,
            glossaries=False,
            context_hint=False,
            native_batch=False,
        )

    def translate(self, text: str, options: TranslateOptions) -> TranslateResult:
        self.seen.append(text)
        return TranslateResult(text=text.upper(), detected_source_lang="en")

    def health_check(self) -> bool:
        return True


class FakeLLM:
    name = "fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        return self.reply


def _write(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _read(path: Path) -> dict[str, str]:
    data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
    return data


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> UpperProvider:
    p = UpperProvider()
    monkeypatch.setattr(catalog_cmd, "translation_provider_from_env", lambda: p)
    return p


def test_catalog_translates_and_writes_in_source_order(
    tmp_path: Path, provider: UpperProvider
) -> None:
    src = tmp_path / "en.json"
    _write(src, {"Save": "Save", "Open {name}": "Open {name}"})

    assert main(["--env-file", str(tmp_path / "none"), "catalog", str(src), "es-ES"]) == 0

    out = _read(tmp_path / "es-ES.json")
    assert list(out) == ["Save", "Open {name}"]
    assert out["Open {name}"] == "OPEN {name}"


def test_catalog_resumes_from_existing_output(tmp_path: Path, provider: UpperProvider) -> None:
    src = tmp_path / "en.json"
    out = tmp_path / "es.json"
    _write(src, {"Save": "Save", "Open": "Open"})
    _write(out, {"Save": "Guardar"})

    main(["--env-file", str(tmp_path / "none"), "catalog", str(src), "es", "--out", str(out)])

    assert provider.seen == ["Open"]
    assert _read(out) == {"Save": "Guardar", "Open": "OPEN"}


def test_catalog_reads_protected_terms_file(tmp_path: Path, provider: UpperProvider) -> None:
    src = tmp_path / "en.json"
    terms = tmp_path / "terms.json"
    _write(src, {"Open Widgetbox": ""})
    _write(terms, {"terms": ["Widgetbox"]})

    main(["--env-file", str(tmp_path / "none"), "catalog", str(src), "es", "--protect", str(terms)])

    assert _read(tmp_path / "es.json") == {"Open Widgetbox": "OPEN Widgetbox"}


def test_review_uses_the_completion_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(review_cmd, "completion_provider_from_env", lambda: FakeLLM("WRONG: no"))
    cat = tmp_path / "es.json"
    flagged = tmp_path / "flagged.json"
    _write(cat, {"Resume": "Currículum", "Same": "Same"})

    main(["--env-file", str(tmp_path / "none"), "review", str(cat), "es", "--out", str(flagged)])

    assert list(json.loads(flagged.read_text(encoding="utf-8"))) == ["Resume"]


def test_plurals_edits_the_catalog_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        plurals_cmd, "completion_provider_from_env", lambda: FakeLLM("few: # a\nmany: # b")
    )
    key = "{count, plural, one {# file} other {# files}}"
    cat = tmp_path / "ru.json"
    _write(cat, {key: "{count, plural, one {# c} other {# d}}"})

    main(["--env-file", str(tmp_path / "none"), "plurals", str(cat), "ru-RU"])

    assert "few {# a}" in _read(cat)[key]


def test_missing_completion_model_exits_with_a_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPLETION_MODEL", raising=False)
    with pytest.raises(SystemExit, match="COMPLETION_MODEL"):
        _common.completion_provider_from_env()


def test_env_file_does_not_override_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = tmp_path / ".env"
    env.write_text('# comment\nCT_TEST_A="from file"\nexport CT_TEST_B=b\n', encoding="utf-8")
    monkeypatch.setenv("CT_TEST_A", "from env")
    monkeypatch.delenv("CT_TEST_B", raising=False)

    _common.load_env_file(env)

    assert os.environ["CT_TEST_A"] == "from env"
    assert os.environ["CT_TEST_B"] == "b"
    monkeypatch.delenv("CT_TEST_B")


def test_sdk_modules_do_not_depend_on_the_cli() -> None:
    root = Path(_common.__file__).resolve().parents[1]
    sdk = [*root.glob("*.py"), *(root / "provider").glob("*.py")]
    for path in sdk:
        imports = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert not any("argparse" in line or "cli" in line for line in imports), path
