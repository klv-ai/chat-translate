from __future__ import annotations

from chat_translate import DetectionResult, ResolveSourceOptions, resolve_source


class _Stub:
    def __init__(self, result: DetectionResult) -> None:
        self._r = result

    def detect(self, text: str) -> DetectionResult:
        return self._r


def stub(lang: str | None, reliable: bool, score: float | None = None) -> _Stub:
    return _Stub(DetectionResult(lang=lang, reliable=reliable, score=score))


def test_override_is_ground_truth_without_detecting() -> None:
    class _Boom:
        def detect(self, text: str) -> DetectionResult:
            raise AssertionError("should not detect when an override is present")

    out = resolve_source("whatever", _Boom(), ResolveSourceOptions(ui_lang="en", override="fr"))
    assert (out.lang, out.provenance, out.confident) == ("fr", "user_confirmed", True)


def test_long_reliable_detection_trusted_over_prior() -> None:
    out = resolve_source(
        "this is clearly written in english here",
        stub("en", reliable=True),
        ResolveSourceOptions(ui_lang="de"),
    )
    assert out.lang == "en"
    assert out.provenance == "detected"
    assert out.confident


def test_unreliable_falls_back_to_prior() -> None:
    out = resolve_source("xx", stub("en", reliable=False), ResolveSourceOptions(ui_lang="de"))
    assert out.lang == "de"
    assert out.provenance == "ui_fallback"
    assert not out.confident
    assert out.detected == "en"


def test_short_agreeing_with_prior_is_trusted() -> None:
    out = resolve_source("ok", stub("en", reliable=True), ResolveSourceOptions(ui_lang="en"))
    assert out.lang == "en"
    assert out.provenance == "detected"
    assert out.confident


def test_short_disagree_prefer_prior_keeps_ui_lang() -> None:
    out = resolve_source(
        "ok",
        stub("en", reliable=True),
        ResolveSourceOptions(ui_lang="it", short_text_policy="prefer_prior"),
    )
    assert out.lang == "it"
    assert out.provenance == "ui_fallback"
    assert not out.confident
    assert out.detected == "en"


def test_short_disagree_trust_detector_takes_detection() -> None:
    out = resolve_source(
        "ok",
        stub("en", reliable=True),
        ResolveSourceOptions(ui_lang="it", short_text_policy="trust_detector"),
    )
    assert out.lang == "en"
    assert out.provenance == "detected"
    assert not out.confident
