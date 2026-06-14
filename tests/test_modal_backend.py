from __future__ import annotations

from typing import Any

from trace_scrubber.modal_backend import ModalPIIRedactor
from trace_scrubber.redactors import RedactionConfig


def test_modal_redactor_calls_deployed_batch_function() -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeFunction:
        def remote(self, texts: list[str], settings: dict[str, object]) -> list[dict[str, Any]]:
            calls.append((texts, settings))
            return [{"text": "hello <REDACTED:email>", "pii_counts": {"email": 1}}]

    def lookup(app_name: str, function_name: str) -> FakeFunction:
        assert app_name == "agent-trace-privacy-scrubber"
        assert function_name == "redact_text_batch"
        return FakeFunction()

    redactor = ModalPIIRedactor(function_lookup=lookup)
    config = RedactionConfig(
        model_name="OpenMed/privacy-filter-nemotron",
        mode="mask",
        chunk_size=4096,
        confidence_threshold=0.75,
        seed=7,
    )

    result = redactor.redact("hello alice@example.com", config)

    assert result.text == "hello <REDACTED:email>"
    assert result.pii_counts["email"] == 1
    assert calls == [
        (
            ["hello alice@example.com"],
            {
                "model_name": "OpenMed/privacy-filter-nemotron",
                "mode": "mask",
                "chunk_size": 4096,
                "confidence_threshold": 0.75,
                "seed": 7,
            },
        )
    ]
