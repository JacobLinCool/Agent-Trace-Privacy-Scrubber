"""Streaming JSONL trace processing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Iterator, TextIO

from .redactors import (
    OpenMedPIIRedactor,
    PIIRedactor,
    RedactionConfig,
    TextRedactionResult,
    sanitize_texts,
)


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
            "counts_by_regex_secret_rule": dict(
                sorted(self.counts_by_regex_secret_rule.items())
            ),
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


@dataclass(frozen=True)
class _StringRef:
    index: int


@dataclass(frozen=True)
class _InputLine:
    line_number: int
    text: str


@dataclass
class _ProgressState:
    last_progress_at: float = 0.0
    last_progress_line: int = 0


def sanitize_json_value(
    value: object,
    config: RedactionConfig,
    model_redactor: PIIRedactor | None = None,
) -> tuple[object, TextRedactionResult]:
    """Recursively redact string values while preserving JSON structure."""

    return sanitize_json_values([value], config, model_redactor=model_redactor)[0]


def sanitize_json_values(
    values: list[object],
    config: RedactionConfig,
    model_redactor: PIIRedactor | None = None,
) -> list[tuple[object, TextRedactionResult]]:
    """Redact string leaves from multiple JSON values in one model batch."""

    strings: list[str] = []
    templates: list[object] = []
    refs_by_value: list[list[int]] = []
    for value in values:
        refs: list[int] = []
        templates.append(_collect_string_refs(value, strings, refs))
        refs_by_value.append(refs)

    redactions = sanitize_texts(strings, config, model_redactor=model_redactor)

    sanitized_results: list[tuple[object, TextRedactionResult]] = []
    for template, refs in zip(templates, refs_by_value):
        sanitized = _materialize_string_refs(template, redactions)
        aggregate = TextRedactionResult(
            text=sanitized if isinstance(sanitized, str) else ""
        )
        for ref in refs:
            aggregate.regex_counts.update(redactions[ref].regex_counts)
            aggregate.pii_counts.update(redactions[ref].pii_counts)
        sanitized_results.append((sanitized, aggregate))
    return sanitized_results


def _collect_string_refs(value: object, strings: list[str], refs: list[int]) -> object:
    if isinstance(value, str):
        index = len(strings)
        strings.append(value)
        refs.append(index)
        return _StringRef(index)

    if isinstance(value, list):
        return [_collect_string_refs(item, strings, refs) for item in value]

    if isinstance(value, dict):
        return {
            key: _collect_string_refs(item, strings, refs)
            for key, item in value.items()
        }

    return value


def _materialize_string_refs(
    template: object, redactions: list[TextRedactionResult]
) -> object:
    if isinstance(template, _StringRef):
        return redactions[template.index].text
    if isinstance(template, list):
        return [_materialize_string_refs(item, redactions) for item in template]
    if isinstance(template, dict):
        return {
            key: _materialize_string_refs(item, redactions)
            for key, item in template.items()
        }
    return template


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
    progress_state = _ProgressState()
    line_batch_size = _line_batch_size(model, config)

    with (
        input_file.open("r", encoding="utf-8", errors="replace") as reader,
        output_file.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as writer,
    ):
        line_batch: list[_InputLine] = []
        for line_number, line in enumerate(reader, start=1):
            line_batch.append(
                _InputLine(line_number=line_number, text=line.rstrip("\r\n"))
            )
            if len(line_batch) >= line_batch_size:
                yield from _process_line_batch(
                    line_batch,
                    writer,
                    config,
                    model,
                    relative_path,
                    total_lines,
                    report,
                    progress_interval,
                    progress_debounce_seconds,
                    progress_state,
                )
                line_batch.clear()

        if line_batch:
            yield from _process_line_batch(
                line_batch,
                writer,
                config,
                model,
                relative_path,
                total_lines,
                report,
                progress_interval,
                progress_debounce_seconds,
                progress_state,
            )

    report.bytes_out = output_file.stat().st_size
    report.duration_seconds = time.monotonic() - start_time
    if (
        report.lines_processed
        and progress_state.last_progress_line != report.lines_processed
    ):
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


def _process_line_batch(
    line_batch: list[_InputLine],
    writer: TextIO,
    config: RedactionConfig,
    model_redactor: PIIRedactor,
    relative_path: str,
    total_lines: int,
    report: FileProcessReport,
    progress_interval: int,
    progress_debounce_seconds: float,
    progress_state: _ProgressState,
) -> Iterator[ProcessEvent]:
    results = _sanitize_jsonl_lines(
        [line.text for line in line_batch], config, model_redactor
    )
    for input_line, result in zip(line_batch, results):
        if result.invalid_json:
            report.invalid_json_lines += 1
            _append_invalid_json_warning(report, input_line.line_number)

        writer.write(result.output_line)
        writer.write("\n")

        report.lines_processed = input_line.line_number
        report.counts_by_regex_secret_rule.update(result.redaction.regex_counts)
        report.counts_by_pii_label.update(result.redaction.pii_counts)

        now = time.monotonic()
        should_emit = input_line.line_number == 1 or (
            input_line.line_number % max(1, progress_interval) == 0
            and now - progress_state.last_progress_at >= progress_debounce_seconds
        )
        if should_emit:
            progress_state.last_progress_at = now
            progress_state.last_progress_line = input_line.line_number
            yield ProcessEvent(
                kind="progress",
                progress=_build_progress(relative_path, total_lines, report),
            )


def _line_batch_size(model_redactor: PIIRedactor, config: RedactionConfig) -> int:
    if not config.model_enabled:
        return 1
    try:
        return max(1, int(getattr(model_redactor, "batch_size", 1)))
    except (TypeError, ValueError):
        return 1


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
    return _sanitize_jsonl_lines([raw_line], config, model_redactor)[0]


def _sanitize_jsonl_lines(
    raw_lines: list[str],
    config: RedactionConfig,
    model_redactor: PIIRedactor,
) -> list[_LineResult]:
    values: list[object] = []
    output_as_json: list[bool] = []
    invalid_json: list[bool] = []

    for raw_line in raw_lines:
        if not config.preserve_json_structure:
            values.append(raw_line)
            output_as_json.append(False)
            invalid_json.append(False)
            continue

        try:
            parsed = json.loads(raw_line)
        except json.JSONDecodeError:
            values.append(raw_line)
            output_as_json.append(False)
            invalid_json.append(True)
            continue

        values.append(parsed)
        output_as_json.append(True)
        invalid_json.append(False)

    sanitized_values = sanitize_json_values(
        values, config, model_redactor=model_redactor
    )
    line_results: list[_LineResult] = []
    for (sanitized, redaction), should_dump_json, is_invalid in zip(
        sanitized_values,
        output_as_json,
        invalid_json,
    ):
        if should_dump_json:
            output_line = json.dumps(
                sanitized, ensure_ascii=False, separators=(",", ":")
            )
        else:
            output_line = (
                sanitized
                if isinstance(sanitized, str)
                else json.dumps(sanitized, ensure_ascii=False)
            )
        line_results.append(
            _LineResult(
                output_line=output_line, redaction=redaction, invalid_json=is_invalid
            )
        )
    return line_results


def _append_invalid_json_warning(report: FileProcessReport, line_number: int) -> None:
    if len(report.warnings) < 100:
        report.warnings.append(
            f"Line {line_number} is invalid JSON; sanitized as raw text."
        )
    elif len(report.warnings) == 100:
        report.warnings.append("Additional invalid JSON line warnings omitted.")
