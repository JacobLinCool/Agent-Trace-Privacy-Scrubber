from __future__ import annotations

from pathlib import Path

from trace_scrubber.jsonl_processor import process_jsonl_file
from trace_scrubber.redactors import RedactionConfig


def test_invalid_jsonl_line_is_sanitized_as_raw_text(tmp_path: Path) -> None:
    input_file = tmp_path / "trace.jsonl"
    output_file = tmp_path / "out" / "trace.jsonl"
    input_file.write_text(
        '{"ok": "sk-proj-FAKEFAKEFAKEFAKEFAKEFAKE"}\n'
        'not json with hf_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE\n',
        encoding="utf-8",
    )
    config = RedactionConfig(regex_enabled=True, model_enabled=False)

    report = process_jsonl_file(input_file, output_file, "trace.jsonl", config, total_lines=2)
    output = output_file.read_text(encoding="utf-8")

    assert report.lines_processed == 2
    assert report.invalid_json_lines == 1
    assert "sk-proj-FAKE" not in output
    assert "hf_FAKE" not in output
    assert "not json with" in output
