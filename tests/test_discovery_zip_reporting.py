from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from trace_scrubber.discovery import discover_roots
from trace_scrubber.jsonl_processor import FileProcessReport
from trace_scrubber.redactors import RedactionConfig
from trace_scrubber.reporting import aggregate_file_reports
from trace_scrubber.zipper import build_zip_archive


def test_directory_discovery_finds_trace_files_and_guesses_agent(tmp_path: Path) -> None:
    root = tmp_path / ".codex" / "sessions"
    root.mkdir(parents=True)
    (root / "session.jsonl").write_text('{"message":"hello"}\n', encoding="utf-8")
    (root / "notes.txt").write_text("ignore me", encoding="utf-8")

    logs = discover_roots([root])

    assert len(logs) == 1
    assert logs[0].filename == "session.jsonl"
    assert logs[0].line_count == 1
    assert logs[0].agent_guess == "Codex"


def test_zip_output_preserves_relative_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    sanitized = workspace / "sanitized" / "nested"
    sanitized.mkdir(parents=True)
    (sanitized / "session.jsonl").write_text('{"safe":"ok"}\n', encoding="utf-8")
    report = FileProcessReport(
        input_relative_path="nested/session.jsonl",
        output_relative_path="nested/session.jsonl",
        lines_processed=1,
    )

    zip_path = build_zip_archive(workspace, [report], RedactionConfig(model_enabled=False))

    with ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        assert "nested/session.jsonl" in names
        assert "redaction_report.json" in names
        assert "README_FIRST.txt" in names


def test_report_aggregation_counts_totals() -> None:
    first = FileProcessReport(
        input_relative_path="a.jsonl",
        output_relative_path="a.jsonl",
        bytes_in=10,
        bytes_out=8,
        lines_processed=2,
        invalid_json_lines=1,
    )
    first.counts_by_regex_secret_rule.update({"openai_api_key": 2})
    first.counts_by_pii_label.update({"email": 1})

    second = FileProcessReport(
        input_relative_path="b.jsonl",
        output_relative_path="b.jsonl",
        bytes_in=20,
        bytes_out=15,
        lines_processed=3,
    )
    second.counts_by_regex_secret_rule.update({"openai_api_key": 1, "jwt": 1})

    aggregate = aggregate_file_reports([first, second])

    assert aggregate["files_processed"] == 2
    assert aggregate["bytes_in"] == 30
    assert aggregate["lines_processed"] == 5
    assert aggregate["invalid_json_lines"] == 1
    assert aggregate["counts_by_regex_secret_rule"]["openai_api_key"] == 3
    assert aggregate["counts_by_pii_label"]["email"] == 1
