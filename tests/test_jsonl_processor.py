from __future__ import annotations

from pathlib import Path

from trace_scrubber.jsonl_processor import process_jsonl_file
from trace_scrubber.jsonl_processor import process_jsonl_file_iter
from trace_scrubber.redactors import RedactionConfig, TextRedactionResult


def test_invalid_jsonl_line_is_sanitized_as_raw_text(tmp_path: Path) -> None:
    input_file = tmp_path / "trace.jsonl"
    output_file = tmp_path / "out" / "trace.jsonl"
    input_file.write_text(
        '{"ok": "sk-proj-FAKEFAKEFAKEFAKEFAKEFAKE"}\n'
        "not json with hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE\n",
        encoding="utf-8",
    )
    config = RedactionConfig(regex_enabled=True, model_enabled=False)

    report = process_jsonl_file(
        input_file, output_file, "trace.jsonl", config, total_lines=2
    )
    output = output_file.read_text(encoding="utf-8")

    assert report.lines_processed == 2
    assert report.invalid_json_lines == 1
    assert "sk-proj-FAKE" not in output
    assert "hf_FAKE" not in output
    assert "not json with" in output


def test_progress_can_emit_every_line_when_debounce_is_zero(tmp_path: Path) -> None:
    input_file = tmp_path / "trace.jsonl"
    output_file = tmp_path / "out" / "trace.jsonl"
    input_file.write_text('{"a":"1"}\n{"a":"2"}\n{"a":"3"}\n', encoding="utf-8")
    config = RedactionConfig(regex_enabled=True, model_enabled=False)

    events = list(
        process_jsonl_file_iter(
            input_file,
            output_file,
            "trace.jsonl",
            config,
            total_lines=3,
            progress_debounce_seconds=0,
        )
    )
    progress_lines = [
        event.progress.line_number for event in events if event.progress is not None
    ]

    assert progress_lines == [1, 2, 3]


def test_progress_debounce_still_emits_final_line(tmp_path: Path) -> None:
    input_file = tmp_path / "trace.jsonl"
    output_file = tmp_path / "out" / "trace.jsonl"
    input_file.write_text('{"a":"1"}\n{"a":"2"}\n{"a":"3"}\n', encoding="utf-8")
    config = RedactionConfig(regex_enabled=True, model_enabled=False)

    events = list(
        process_jsonl_file_iter(
            input_file,
            output_file,
            "trace.jsonl",
            config,
            total_lines=3,
            progress_debounce_seconds=999,
        )
    )
    progress_lines = [
        event.progress.line_number for event in events if event.progress is not None
    ]

    assert progress_lines == [1, 3]


def test_model_redaction_batches_across_jsonl_lines(tmp_path: Path) -> None:
    class FakeBatchRedactor:
        batch_size = 8

        def prepare_model(self, config: RedactionConfig) -> None:
            pass

        def redact(self, text: str, config: RedactionConfig) -> TextRedactionResult:
            return self.redact_many([text], config)[0]

        def redact_many(
            self, texts: list[str], config: RedactionConfig
        ) -> list[TextRedactionResult]:
            calls.append(texts)
            return [
                TextRedactionResult(text=f"<PII:{index}>")
                for index, _ in enumerate(texts)
            ]

        def release(self) -> None:
            pass

    calls: list[list[str]] = []
    input_file = tmp_path / "trace.jsonl"
    output_file = tmp_path / "out" / "trace.jsonl"
    input_file.write_text(
        '{"content":"alice@example.com"}\n'
        '{"content":"bob@example.com"}\n'
        '{"nested":["carol@example.com"]}\n',
        encoding="utf-8",
    )
    config = RedactionConfig(regex_enabled=False, model_enabled=True)

    process_jsonl_file(
        input_file,
        output_file,
        "trace.jsonl",
        config,
        total_lines=3,
        model_redactor=FakeBatchRedactor(),
    )

    assert calls == [["alice@example.com", "bob@example.com", "carol@example.com"]]
    assert output_file.read_text(encoding="utf-8").splitlines() == [
        '{"content":"<PII:0>"}',
        '{"content":"<PII:1>"}',
        '{"nested":["<PII:2>"]}',
    ]
