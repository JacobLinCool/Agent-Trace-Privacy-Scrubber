"""Report generation and aggregation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Iterable

from . import APP_NAME, APP_VERSION
from .jsonl_processor import FileProcessReport
from .redactors import RedactionConfig

DISCLAIMER = (
    "Automated redaction is imperfect. Review sanitized traces manually before "
    "publishing or uploading them."
)


def aggregate_file_reports(reports: Iterable[FileProcessReport]) -> dict[str, object]:
    """Aggregate per-file redaction reports into package totals."""

    report_list = list(reports)
    pii_counts: Counter[str] = Counter()
    regex_counts: Counter[str] = Counter()
    errors = 0
    warnings = 0
    for report in report_list:
        pii_counts.update(report.counts_by_pii_label)
        regex_counts.update(report.counts_by_regex_secret_rule)
        errors += len(report.errors)
        warnings += len(report.warnings)

    return {
        "files_processed": len(report_list),
        "bytes_in": sum(report.bytes_in for report in report_list),
        "bytes_out": sum(report.bytes_out for report in report_list),
        "lines_processed": sum(report.lines_processed for report in report_list),
        "invalid_json_lines": sum(report.invalid_json_lines for report in report_list),
        "duration_seconds": round(
            sum(report.duration_seconds for report in report_list), 3
        ),
        "errors": errors,
        "warnings": warnings,
        "counts_by_pii_label": dict(sorted(pii_counts.items())),
        "counts_by_regex_secret_rule": dict(sorted(regex_counts.items())),
    }


def build_redaction_report(
    file_reports: list[FileProcessReport],
    config: RedactionConfig,
) -> dict[str, object]:
    """Build the JSON-serializable report written into each zip."""

    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_model": config.model_name,
        "redaction_mode": config.mode,
        "regex_sweep_enabled": config.regex_enabled,
        "model_pii_redaction_enabled": config.model_enabled,
        "preserve_json_structure": config.preserve_json_structure,
        "files_processed": len(file_reports),
        "files": [report.to_dict() for report in file_reports],
        "aggregate_totals": aggregate_file_reports(file_reports),
        "disclaimer": DISCLAIMER,
    }


def report_preview_rows(
    file_reports: Iterable[FileProcessReport],
) -> list[list[object]]:
    """Small table for the Gradio report preview."""

    rows: list[list[object]] = []
    for report in file_reports:
        rows.append(
            [
                report.input_relative_path,
                report.lines_processed,
                report.invalid_json_lines,
                sum(report.counts_by_regex_secret_rule.values()),
                sum(report.counts_by_pii_label.values()),
                len(report.errors),
                len(report.warnings),
                round(report.duration_seconds, 2),
            ]
        )
    return rows
