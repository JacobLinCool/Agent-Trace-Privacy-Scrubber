from __future__ import annotations

from collections import Counter

from trace_scrubber.redactors import (
    CachingPIIRedactor,
    RedactionConfig,
    TextRedactionResult,
)


class RecordingRedactor:
    """Inner redactor that records every batch it is asked to compute."""

    def __init__(self, batch_size: int = 7) -> None:
        self.batch_size = batch_size
        self.calls: list[list[str]] = []
        self.prepared = 0
        self.released = 0

    def prepare_model(self, config: RedactionConfig) -> None:
        self.prepared += 1

    def redact(self, text: str, config: RedactionConfig) -> TextRedactionResult:
        return self.redact_many([text], config)[0]

    def redact_many(
        self, texts: list[str], config: RedactionConfig
    ) -> list[TextRedactionResult]:
        self.calls.append(list(texts))
        return [
            TextRedactionResult(text=f"R:{t}", pii_counts=Counter({"x": 1}))
            for t in texts
        ]

    def release(self) -> None:
        self.released += 1


def test_caching_redactor_dedupes_within_call_and_maps_back() -> None:
    inner = RecordingRedactor()
    cache = CachingPIIRedactor(inner)
    cfg = RedactionConfig()

    out = cache.redact_many(["a", "b", "a", "b", "a"], cfg)

    assert [r.text for r in out] == ["R:a", "R:b", "R:a", "R:b", "R:a"]
    # The model only ever saw each distinct string once.
    assert inner.calls == [["a", "b"]]


def test_caching_redactor_caches_across_calls() -> None:
    inner = RecordingRedactor()
    cache = CachingPIIRedactor(inner)
    cfg = RedactionConfig()

    cache.redact_many(["a", "b"], cfg)
    out2 = cache.redact_many(["a", "c", "b"], cfg)

    assert [r.text for r in out2] == ["R:a", "R:c", "R:b"]
    # Second call only computes the new string.
    assert inner.calls == [["a", "b"], ["c"]]
    assert cache.cache_size == 3


def test_caching_redactor_exposes_inner_batch_size() -> None:
    assert CachingPIIRedactor(RecordingRedactor(batch_size=9)).batch_size == 9


def test_caching_redactor_skips_oversized_strings() -> None:
    inner = RecordingRedactor()
    cache = CachingPIIRedactor(inner, max_cache_chars=3)
    cfg = RedactionConfig()

    cache.redact_many(["abcd"], cfg)  # length 4 > cap -> not retained
    cache.redact_many(["abcd"], cfg)  # therefore recomputed
    assert inner.calls == [["abcd"], ["abcd"]]

    cache.redact_many(["xy", "xy"], cfg)  # short -> cached + deduped
    assert inner.calls[-1] == ["xy"]


def test_caching_redactor_release_clears_cache() -> None:
    inner = RecordingRedactor()
    cache = CachingPIIRedactor(inner)
    cfg = RedactionConfig()

    cache.redact_many(["a"], cfg)
    assert cache.cache_size == 1

    cache.release()
    assert cache.cache_size == 0
    assert inner.released == 1

    cache.redact_many(["a"], cfg)  # recomputed after release
    assert inner.calls == [["a"], ["a"]]


def test_caching_redactor_empty_input() -> None:
    inner = RecordingRedactor()
    assert CachingPIIRedactor(inner).redact_many([], RedactionConfig()) == []
    assert inner.calls == []
