"""Streaming JSONL trace processing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Iterator

from .redactors import OpenMedPIIRedactor, PIIRedactor, RedactionConfig, TextRedactionResult, sanitize_text


@dataclass
class FileProcessReport:
    input_relative_path: str
    output_relative_path: str
    bytes_in: int = 0
    bytes_out: int = 0
    lines_processed: int = 0
    invalid_json_lines: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    counts_by_pii_label: Counter[str] = field(default_factory=Counter)
    counts_by_regex_secret_rule: Counter[str] = field(default_factory=Counter)

    def to_dict(self) -> dict[str, object]:
        return {
            "input_relative_path": self.input_relative_path,
            "output_relative_path": self.output_relative_path,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "lines_processed": self.lines_processed,
            "invalid_json_lines": self.invalid_json_lines,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "duration_seconds": round(self.duration_seconds, 3),
            "counts_by_pii_label": dict(sorted(self.counts_by_pii_label.items())),
            "counts_by_regex_secret_rule": dict(sorted(self.counts_by_regex_secret_rule.items())),
        }


@dataclass
class ProcessProgress:
    relative_path: str
    line_number: int
    total_lines: int
    bytes_in: int
    counts_by_pii_label: Counter[str]
    counts_by_regex_secret_rule: Counter[str]


@dataclass
class ProcessEvent:
    kind: str
    progress: ProcessProgress | None = None
    report: FileProcessReport | None = None


def sanitize_json_value(
    value: object,
    config: RedactionConfig,
    model_redactor: PIIRedactor | None = None,
) -> tuple[object, TextRedactionResult]:
    """Recursively redact string values while preserving JSON structure."""

    if isinstance(value, str):
        result = sanitize_text(value, config, model_redactor=model_redactor)
        return result.text, result

    if isinstance(value, list):
        sanitized_items: list[object] = []
        aggregate = TextRedactionResult(text="")
        for item in value:
            sanitized, result = sanitize_json_value(item, config, model_redactor=model_redactor)
            sanitized_items.append(sanitized)
            aggregate.regex_counts.update(result.regex_counts)
            aggregate.pii_counts.update(result.pii_counts)
        return sanitized_items, aggregate

    if isinstance(value, dict):
        sanitized_object: dict[object, object] = {}
        aggregate = TextRedactionResult(text="")
        for key, item in value.items():
            sanitized, result = sanitize_json_value(item, config, model_redactor=model_redactor)
            sanitized_object[key] = sanitized
            aggregate.regex_counts.update(result.regex_counts)
            aggregate.pii_counts.update(result.pii_counts)
        return sanitized_object, aggregate

    return value, TextRedactionResult(text="")


def process_jsonl_file(
    input_path: str | Path,
    output_path: str | Path,
    relative_path: str,
    config: RedactionConfig,
    total_lines: int = 0,
    model_redactor: PIIRedactor | None = None,
) -> FileProcessReport:
    """Process a JSONL file and return its report."""

    report: FileProcessReport | None = None
    for event in process_jsonl_file_iter(
        input_path=input_path,
        output_path=output_path,
        relative_path=relative_path,
        config=config,
        total_lines=total_lines,
        model_redactor=model_redactor,
    ):
        if event.report is not None:
            report = event.report
    if report is None:
        raise RuntimeError("JSONL processing finished without a report")
    return report


def process_jsonl_file_iter(
    input_path: str | Path,
    output_path: str | Path,
    relative_path: str,
    config: RedactionConfig,
    total_lines: int = 0,
    model_redactor: PIIRedactor | None = None,
    progress_interval: int = 1,
    progress_debounce_seconds: float = 2.0,
) -> Iterator[ProcessEvent]:
    """Yield progress while processing one JSONL trace file line by line."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    model = model_redactor or OpenMedPIIRedactor()
    start_time = time.monotonic()
    report = FileProcessReport(
        input_relative_path=relative_path,
        output_relative_path=relative_path,
        bytes_in=input_file.stat().st_size,
    )
    last_progress_at = 0.0
    last_progress_line = 0

    with input_file.open("r", encoding="utf-8", errors="replace") as reader, output_file.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as writer:
        for line_number, line in enumerate(reader, start=1):
            raw_line = line.rstrip("\r\n")
            result = _sanitize_jsonl_line(raw_line, config, model)
            if result.invalid_json:
                report.invalid_json_lines += 1
                _append_invalid_json_warning(report, line_number)

            writer.write(result.output_line)
            writer.write("\n")

            report.lines_processed = line_number
            report.counts_by_regex_secret_rule.update(result.redaction.regex_counts)
            report.counts_by_pii_label.update(result.redaction.pii_counts)

            now = time.monotonic()
            should_emit = (
                line_number == 1
                or (
                    line_number % max(1, progress_interval) == 0
                    and now - last_progress_at >= progress_debounce_seconds
                )
            )
            if should_emit:
                last_progress_at = now
                last_progress_line = line_number
                yield ProcessEvent(
                    kind="progress",
                    progress=_build_progress(relative_path, total_lines, report),
                )

    report.bytes_out = output_file.stat().st_size
    report.duration_seconds = time.monotonic() - start_time
    if report.lines_processed and last_progress_line != report.lines_processed:
        yield ProcessEvent(
            kind="progress",
            progress=_build_progress(relative_path, total_lines, report),
        )
    yield ProcessEvent(kind="complete", report=report)


@dataclass
class _LineResult:
    output_line: str
    redaction: TextRedactionResult
    invalid_json: bool = False


def _build_progress(
    relative_path: str,
    total_lines: int,
    report: FileProcessReport,
) -> ProcessProgress:
    return ProcessProgress(
        relative_path=relative_path,
        line_number=report.lines_processed,
        total_lines=total_lines,
        bytes_in=report.bytes_in,
        counts_by_pii_label=report.counts_by_pii_label.copy(),
        counts_by_regex_secret_rule=report.counts_by_regex_secret_rule.copy(),
    )


def _sanitize_jsonl_line(
    raw_line: str,
    config: RedactionConfig,
    model_redactor: PIIRedactor,
) -> _LineResult:
    if config.preserve_json_structure:
        try:
            parsed = json.loads(raw_line)
        except json.JSONDecodeError:
            redaction = sanitize_text(raw_line, config, model_redactor=model_redactor)
            return _LineResult(output_line=redaction.text, redaction=redaction, invalid_json=True)

        sanitized, redaction = sanitize_json_value(parsed, config, model_redactor=model_redactor)
        output_line = json.dumps(sanitized, ensure_ascii=False, separators=(",", ":"))
        return _LineResult(output_line=output_line, redaction=redaction)

    redaction = sanitize_text(raw_line, config, model_redactor=model_redactor)
    return _LineResult(output_line=redaction.text, redaction=redaction)


def _append_invalid_json_warning(report: FileProcessReport, line_number: int) -> None:
    if len(report.warnings) < 100:
        report.warnings.append(f"Line {line_number} is invalid JSON; sanitized as raw text.")
    elif len(report.warnings) == 100:
        report.warnings.append("Additional invalid JSON line warnings omitted.")
