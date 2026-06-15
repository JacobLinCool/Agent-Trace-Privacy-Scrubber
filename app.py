from __future__ import annotations

import spaces
import html
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import gradio as gr

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trace_scrubber import APP_NAME, APP_VERSION
from trace_scrubber.discovery import (
    KNOWN_LOCAL_SOURCES,
    discover_custom_path,
    discover_known_source,
    discover_roots,
    discover_uploaded_files,
    rows_for_table,
    selected_relative_paths,
)
from trace_scrubber.jsonl_processor import FileProcessReport, process_jsonl_file_iter
from trace_scrubber.modal_backend import ModalPIIRedactor
from trace_scrubber.redactors import (
    ModelRedactionError,
    OpenMedPIIRedactor,
    PIIRedactor,
    RedactionConfig,
)
from trace_scrubber.reporting import report_preview_rows
from trace_scrubber.zipper import build_zip_archive, make_output_workspace

TABLE_HEADERS = [
    "selected",
    "agent_guess",
    "relative_path",
    "filename",
    "size_kb",
    "line_count",
    "status",
]

REPORT_HEADERS = [
    "file",
    "lines",
    "invalid_json",
    "regex_redactions",
    "pii_redactions",
    "errors",
    "warnings",
    "seconds",
]

MODEL_OPTIONS = [
    "OpenMed/privacy-filter-nemotron",
    "OpenMed/privacy-filter-nemotron-mlx",
]

CURRENT_RUNTIME_BACKEND = "Current app runtime (local machine or Space ZeroGPU)"
MODAL_CLOUD_BACKEND = "Modal cloud GPU (sends regex-sanitized text to Modal)"
COMPUTE_BACKEND_OPTIONS = [CURRENT_RUNTIME_BACKEND, MODAL_CLOUD_BACKEND]
MODAL_MODEL_OPTIONS = ["OpenMed/privacy-filter-nemotron"]

SOURCE_OPTIONS = [
    "Known local source",
    "Custom local path",
    "Upload files",
    "Upload directory",
    "Use sample logs",
]

APP_CSS = """
:root,
.gradio-container {
  --ats-bg: #f6f7f8;
  --ats-panel: #ffffff;
  --ats-panel-soft: #f9fafb;
  --ats-input: #ffffff;
  --ats-input-muted: #f3f4f6;
  --ats-border: #d9dee4;
  --ats-border-strong: #b9c2cd;
  --ats-text: #111827;
  --ats-muted: #5b6472;
  --ats-subtle: #7a8493;
  --ats-inverted-text: #ffffff;
  --ats-accent: #0f766e;
  --ats-accent-strong: #115e59;
  --ats-warn: #a16207;
  --ats-error: #b91c1c;
  --ats-success: #047857;
  --ats-shadow: 0 8px 24px rgba(17, 24, 39, 0.08);
}

body.dark,
body.dark .gradio-container,
.dark .gradio-container {
  --ats-bg: #181915;
  --ats-panel: #23241f;
  --ats-panel-soft: #2d2e28;
  --ats-input: #1f201c;
  --ats-input-muted: #171814;
  --ats-border: #4a4c43;
  --ats-border-strong: #686b5f;
  --ats-text: #f4f1e8;
  --ats-muted: #d2ccbd;
  --ats-subtle: #aaa394;
  --ats-inverted-text: #ffffff;
  --ats-accent: #2dd4bf;
  --ats-accent-strong: #99f6e4;
  --ats-warn: #facc15;
  --ats-error: #f87171;
  --ats-success: #34d399;
  --ats-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
  background: var(--ats-bg) !important;
  color-scheme: dark;
}

body:not(.dark) .gradio-container,
.gradio-container {
  color-scheme: light;
}

.gradio-container {
  background: var(--ats-bg) !important;
  color: var(--ats-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container,
.gradio-container .app-shell,
.gradio-container .workbench-grid,
.gradio-container .footer-note {
  background: var(--ats-bg) !important;
}

.gradio-container,
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label,
.gradio-container p,
.gradio-container span,
.gradio-container table,
.gradio-container th,
.gradio-container td {
  color: var(--ats-text) !important;
}

.app-shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 18px 18px 10px;
}

.app-topbar {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 20px;
  align-items: end;
  border-bottom: 1px solid var(--ats-border);
  padding-bottom: 16px;
}

.eyebrow {
  color: var(--ats-accent-strong);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.app-title {
  margin: 4px 0 4px;
  font-size: 30px;
  line-height: 1.08;
  letter-spacing: 0;
  color: var(--ats-text);
}

.app-subtitle {
  max-width: 780px;
  margin: 0;
  color: var(--ats-muted);
  font-size: 14px;
  line-height: 1.45;
}

.runtime-ledger {
  display: grid;
  grid-template-columns: repeat(2, minmax(130px, 1fr));
  gap: 8px;
  min-width: 330px;
}

.ledger-cell {
  border: 1px solid var(--ats-border);
  background: var(--ats-panel);
  padding: 10px 12px;
  border-radius: 8px;
}

.ledger-cell span {
  display: block;
  color: var(--ats-subtle);
  font-size: 11px;
  line-height: 1.2;
}

.ledger-cell strong {
  display: block;
  margin-top: 2px;
  color: var(--ats-text);
  font-size: 13px;
  line-height: 1.25;
}

.flow-rail {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 8px;
  margin-top: 12px;
}

.flow-step {
  border: 1px solid var(--ats-border);
  background: var(--ats-panel-soft);
  color: var(--ats-muted);
  padding: 8px 10px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 650;
}

.flow-step span {
  color: var(--ats-accent-strong);
  margin-right: 6px;
}

.workbench-grid {
  max-width: 1440px;
  margin: 0 auto;
  padding: 10px 18px 24px;
  gap: 16px !important;
}

.tool-panel {
  border: 1px solid var(--ats-border) !important;
  border-radius: 8px !important;
  background: var(--ats-panel) !important;
  color: var(--ats-text) !important;
  box-shadow: var(--ats-shadow);
  padding: 14px !important;
}

.tool-panel,
.tool-panel > *,
.tool-panel .block,
.tool-panel .form,
.tool-panel .tabs,
.tool-panel .tabitem,
.tool-panel .tab-nav,
.tool-panel [data-testid="block-info"] {
  background-color: var(--ats-panel) !important;
  border-color: var(--ats-border) !important;
  color: var(--ats-text) !important;
}

.tool-panel + .tool-panel {
  margin-top: 12px !important;
}

.section-heading {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.section-heading h2 {
  margin: 0;
  color: var(--ats-text);
  font-size: 15px;
  line-height: 1.25;
  letter-spacing: 0;
}

.section-heading span {
  color: var(--ats-subtle);
  font-size: 12px;
  font-weight: 650;
}

.status-panel,
.run-status {
  border: 1px solid var(--ats-border);
  background: var(--ats-panel-soft);
  border-radius: 8px;
  padding: 12px;
}

.status-panel strong,
.run-status strong {
  color: var(--ats-text);
}

.status-panel p,
.run-status p {
  margin: 4px 0 0;
  color: var(--ats-muted);
  font-size: 13px;
  line-height: 1.4;
}

.status-panel.success,
.run-status.success { border-color: rgba(4, 120, 87, 0.35); }
.status-panel.warning,
.run-status.warning { border-color: rgba(161, 98, 7, 0.35); }
.status-panel.error,
.run-status.error { border-color: rgba(185, 28, 28, 0.35); }
.run-status.active { border-color: rgba(15, 118, 110, 0.42); }

.metric-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin-top: 10px;
}

.metric {
  border: 1px solid var(--ats-border);
  background: var(--ats-input);
  border-radius: 7px;
  padding: 8px 9px;
}

.metric span {
  display: block;
  color: var(--ats-subtle);
  font-size: 11px;
  line-height: 1.2;
}

.metric strong {
  display: block;
  margin-top: 2px;
  font-size: 15px;
  line-height: 1.2;
}

.run-status__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  border: 1px solid var(--ats-border-strong);
  border-radius: 999px;
  color: var(--ats-accent-strong);
  background: var(--ats-input);
  padding: 3px 8px;
  font-size: 11px;
  font-weight: 700;
}

.progress-track {
  height: 8px;
  overflow: hidden;
  border-radius: 999px;
  background: var(--ats-input-muted);
  margin: 10px 0;
}

.progress-fill {
  display: block;
  height: 100%;
  background: var(--ats-accent);
  border-radius: inherit;
}

.inline-warning {
  margin-top: 10px;
  border: 1px solid rgba(161, 98, 7, 0.35);
  background: color-mix(in srgb, var(--ats-warn) 16%, var(--ats-panel));
  color: var(--ats-text);
  border-radius: 8px;
  padding: 10px 12px;
  font-size: 13px;
}

.gradio-container button {
  border-color: var(--ats-border) !important;
}

.gradio-container button.primary,
.gradio-container button[class*="primary"] {
  color: var(--ats-inverted-text) !important;
}

.gradio-container button.stop,
.gradio-container button[class*="stop"] {
  color: var(--ats-inverted-text) !important;
}

.gradio-container input:not([type="checkbox"]):not([type="radio"]),
.gradio-container textarea,
.gradio-container select,
.gradio-container [role="textbox"],
.gradio-container [role="combobox"],
.gradio-container [role="listbox"] {
  background: var(--ats-input) !important;
  border-color: var(--ats-border) !important;
  color: var(--ats-text) !important;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--ats-subtle) !important;
}

.gradio-container input[type="checkbox"],
.gradio-container input[type="radio"] {
  accent-color: #f97316 !important;
}

.gradio-container .dataframe,
.gradio-container .dataframe *,
.gradio-container table {
  border-color: var(--ats-border) !important;
}

.gradio-container table,
.gradio-container tbody,
.gradio-container tr,
.gradio-container td {
  background: var(--ats-panel) !important;
}

.gradio-container thead,
.gradio-container th {
  background: var(--ats-input-muted) !important;
}

.gradio-container .tab-nav,
.gradio-container .tabitem,
.gradio-container [role="tablist"] {
  background: var(--ats-panel-soft) !important;
}

.gradio-container [role="tab"] {
  color: var(--ats-muted) !important;
}

.gradio-container [role="tab"][aria-selected="true"] {
  color: var(--ats-accent-strong) !important;
  border-color: var(--ats-accent) !important;
}

.gradio-container .file-preview,
.gradio-container .file-preview *,
.gradio-container .file,
.gradio-container .file * {
  background: var(--ats-input) !important;
  border-color: var(--ats-border) !important;
  color: var(--ats-text) !important;
}

.compact-actions button {
  min-height: 38px !important;
}

.primary-run button {
  min-height: 46px !important;
  font-weight: 750 !important;
}

.data-surface textarea,
.data-surface input,
.data-surface table {
  font-size: 13px !important;
}

.footer-note {
  max-width: 1440px;
  margin: 0 auto;
  padding: 0 18px 22px;
  color: var(--ats-subtle);
  font-size: 12px;
}

@media (max-width: 900px) {
  .app-topbar,
  .flow-rail,
  .runtime-ledger,
  .metric-grid {
    grid-template-columns: 1fr;
  }

  .app-title {
    font-size: 24px;
  }
}
"""


def scan_logs(
    source_mode: str,
    known_source: str,
    custom_path: str,
    uploaded_files: list[str] | str | None,
    uploaded_directory: list[str] | str | None,
    max_file_size_warning_mb: int,
) -> tuple[str, list[list[object]], list[dict[str, object]]]:
    try:
        if source_mode == "Known local source":
            logs = discover_known_source(known_source, max_file_size_warning_mb)
            target = str(KNOWN_LOCAL_SOURCES[known_source])
        elif source_mode == "Custom local path":
            if not custom_path.strip():
                return (
                    _status_panel_html(
                        "warning",
                        "Path required",
                        "Enter a local path before scanning.",
                    ),
                    [],
                    [],
                )
            logs = discover_custom_path(custom_path, max_file_size_warning_mb)
            target = custom_path
        elif source_mode == "Upload files":
            logs = discover_uploaded_files(
                _normalize_file_inputs(uploaded_files), max_file_size_warning_mb
            )
            target = "uploaded files"
        elif source_mode == "Upload directory":
            logs = discover_uploaded_files(
                _normalize_file_inputs(uploaded_directory), max_file_size_warning_mb
            )
            target = "uploaded directory"
        else:
            sample_root = ROOT / "sample_logs"
            logs = discover_roots([sample_root], max_file_size_warning_mb)
            target = str(sample_root)
    except Exception as exc:
        return _status_panel_html("error", "Scan failed", _safe_error(exc)), [], []

    if not logs:
        return (
            _status_panel_html(
                "warning",
                "No trace files found",
                f"No JSONL, NDJSON, or JSON trace files were found in {_inline_code(target)}.",
                detail_is_html=True,
            ),
            [],
            [],
        )

    rows = rows_for_table(logs, selected=True)
    state = [log.to_dict() for log in logs]
    total_kb = sum(float(row[4]) for row in rows)
    return (
        _status_panel_html(
            "success",
            "Scan complete",
            f"Found trace files in {_inline_code(target)}.",
            detail_is_html=True,
            metrics=[
                ("Files", str(len(rows))),
                ("Selected", str(len(rows))),
                ("Total size", f"{total_kb:.1f} KB"),
            ],
        ),
        rows,
        state,
    )


def select_all(table_data: object) -> list[list[object]]:
    rows = _coerce_rows(table_data)
    for row in rows:
        if row:
            row[0] = True
    return rows


def select_none(table_data: object) -> list[list[object]]:
    rows = _coerce_rows(table_data)
    for row in rows:
        if row:
            row[0] = False
    return rows


def select_likely_trace_files(table_data: object) -> list[list[object]]:
    rows = _coerce_rows(table_data)
    for row in rows:
        filename = str(row[3]).lower() if len(row) > 3 else ""
        row[0] = filename.endswith((".jsonl", ".ndjson"))
    return rows


def process_selected_logs(
    table_data: object,
    logs_state: list[dict[str, object]] | None,
    compute_backend: str,
    model_name: str,
    mode: str,
    regex_enabled: bool,
    model_enabled: bool,
    preserve_json_structure: bool,
    include_report: bool,
    chunk_size: int,
    model_batch_size: int,
    max_file_size_warning_mb: int,
    confidence_threshold: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    if compute_backend == MODAL_CLOUD_BACKEND or not model_enabled:
        yield from _process_selected_logs_impl(
            table_data,
            logs_state,
            compute_backend,
            model_name,
            mode,
            regex_enabled,
            model_enabled,
            preserve_json_structure,
            include_report,
            chunk_size,
            model_batch_size,
            max_file_size_warning_mb,
            confidence_threshold,
            seed,
            progress,
        )
        return

    yield from _process_selected_logs_current_runtime(
        table_data,
        logs_state,
        model_name,
        mode,
        regex_enabled,
        model_enabled,
        preserve_json_structure,
        include_report,
        chunk_size,
        model_batch_size,
        max_file_size_warning_mb,
        confidence_threshold,
        seed,
        progress,
    )


@spaces.GPU(duration=1800)
def _process_selected_logs_current_runtime(
    table_data: object,
    logs_state: list[dict[str, object]] | None,
    model_name: str,
    mode: str,
    regex_enabled: bool,
    model_enabled: bool,
    preserve_json_structure: bool,
    include_report: bool,
    chunk_size: int,
    model_batch_size: int,
    max_file_size_warning_mb: int,
    confidence_threshold: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    yield from _process_selected_logs_impl(
        table_data,
        logs_state,
        CURRENT_RUNTIME_BACKEND,
        model_name,
        mode,
        regex_enabled,
        model_enabled,
        preserve_json_structure,
        include_report,
        chunk_size,
        model_batch_size,
        max_file_size_warning_mb,
        confidence_threshold,
        seed,
        progress,
    )


def _process_selected_logs_impl(
    table_data: object,
    logs_state: list[dict[str, object]] | None,
    compute_backend: str,
    model_name: str,
    mode: str,
    regex_enabled: bool,
    model_enabled: bool,
    preserve_json_structure: bool,
    include_report: bool,
    chunk_size: int,
    model_batch_size: int,
    max_file_size_warning_mb: int,
    confidence_threshold: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    if not regex_enabled and not model_enabled:
        yield (
            _status_panel_html(
                "warning",
                "No redaction pass enabled",
                "Enable deterministic secret scanning, model PII redaction, or both.",
            ),
            None,
            [],
            "",
        )
        return

    logs = logs_state or []
    selected = selected_relative_paths(table_data)
    selected_logs = [log for log in logs if str(log.get("relative_path")) in selected]
    if not selected_logs:
        yield (
            _status_panel_html(
                "warning",
                "No files selected",
                "Select at least one trace file before processing.",
            ),
            None,
            [],
            "",
        )
        return

    if (
        model_enabled
        and compute_backend == MODAL_CLOUD_BACKEND
        and model_name not in MODAL_MODEL_OPTIONS
    ):
        yield (
            _status_panel_html(
                "warning",
                "Unsupported Modal model",
                "Modal cloud GPU currently supports OpenMed/privacy-filter-nemotron only. Choose current runtime for Apple MLX models.",
            ),
            None,
            [],
            "",
        )
        return

    config = RedactionConfig(
        model_name=model_name,
        mode=mode,  # type: ignore[arg-type]
        regex_enabled=regex_enabled,
        model_enabled=model_enabled,
        preserve_json_structure=preserve_json_structure,
        include_report=include_report,
        chunk_size=int(chunk_size),
        model_batch_size=int(model_batch_size),
        confidence_threshold=float(confidence_threshold),
        seed=int(seed),
    )

    workspace = make_output_workspace()
    sanitized_root = workspace / "sanitized"
    model_redactor = _build_model_redactor(compute_backend, int(model_batch_size))
    file_reports: list[FileProcessReport] = []
    total_lines = sum(int(log.get("line_count") or 0) for log in selected_logs)
    total_bytes = sum(int(log.get("size_bytes") or 0) for log in selected_logs)
    start_time = time.monotonic()
    aggregate_regex: Counter[str] = Counter()
    aggregate_pii: Counter[str] = Counter()

    if model_enabled:
        yield (
            _status_html(
                phase=_loading_phase(compute_backend),
                file_index=0,
                file_total=len(selected_logs),
                current_file="waiting for first inference",
                processed_units=0,
                total_units=total_lines or total_bytes,
                unit_name="lines" if total_lines else "bytes",
                start_time=start_time,
                regex_counts=aggregate_regex,
                pii_counts=aggregate_pii,
            ),
            None,
            [],
            "",
        )
        try:
            model_redactor.prepare_model(config)
        except ModelRedactionError as exc:
            model_redactor.release()
            yield (
                _status_panel_html("error", "Model unavailable", _safe_error(exc)),
                None,
                [],
                "",
            )
            return

    processed_lines = 0
    for file_index, log in enumerate(selected_logs, start=1):
        input_path = Path(str(log["path"]))
        relative_path = _safe_relative_output_path(str(log["relative_path"]))
        output_path = sanitized_root / relative_path
        line_count = int(log.get("line_count") or 0)
        file_size = int(log.get("size_bytes") or 0)
        large_warning = ""
        if file_size > max_file_size_warning_mb * 1024 * 1024:
            large_warning = _inline_warning_html(
                f"{relative_path} is larger than {max_file_size_warning_mb} MB."
            )

        try:
            for event in process_jsonl_file_iter(
                input_path=input_path,
                output_path=output_path,
                relative_path=relative_path,
                config=config,
                total_lines=line_count,
                model_redactor=model_redactor,
            ):
                if event.progress is not None:
                    current_processed = processed_lines + event.progress.line_number
                    denominator = total_lines or total_bytes
                    fraction = current_processed / denominator if denominator else 0
                    progress(
                        min(fraction, 1.0),
                        desc=f"{relative_path} line {event.progress.line_number}",
                    )
                    regex_counts = (
                        aggregate_regex + event.progress.counts_by_regex_secret_rule
                    )
                    pii_counts = aggregate_pii + event.progress.counts_by_pii_label
                    yield (
                        _status_html(
                            phase="Processing",
                            file_index=file_index,
                            file_total=len(selected_logs),
                            current_file=relative_path,
                            processed_units=current_processed,
                            total_units=denominator,
                            unit_name="lines" if total_lines else "bytes",
                            start_time=start_time,
                            regex_counts=regex_counts,
                            pii_counts=pii_counts,
                            current_line=event.progress.line_number,
                            file_lines=line_count,
                        )
                        + large_warning,
                        None,
                        report_preview_rows(file_reports),
                        _preview_sanitized_output(sanitized_root, file_reports),
                    )

                if event.report is not None:
                    file_reports.append(event.report)
                    aggregate_regex.update(event.report.counts_by_regex_secret_rule)
                    aggregate_pii.update(event.report.counts_by_pii_label)
                    processed_lines += event.report.lines_processed
        except ModelRedactionError as exc:
            model_redactor.release()
            yield (
                _status_panel_html("error", "Model redaction failed", _safe_error(exc)),
                None,
                report_preview_rows(file_reports),
                "",
            )
            return
        except Exception as exc:
            error_report = FileProcessReport(
                input_relative_path=relative_path,
                output_relative_path=relative_path,
                bytes_in=file_size,
                errors=[f"File processing failed: {_safe_error(exc)}"],
            )
            file_reports.append(error_report)

    if not file_reports:
        model_redactor.release()
        yield (
            _status_panel_html(
                "warning",
                "No files processed",
                "The run ended before any report was created.",
            ),
            None,
            [],
            "",
        )
        return

    zip_path = build_zip_archive(workspace, file_reports, config)
    model_redactor.release()
    progress(1.0, desc="Archive ready")
    yield (
        _status_html(
            phase="Complete",
            file_index=len(selected_logs),
            file_total=len(selected_logs),
            current_file="archive ready",
            processed_units=total_lines or total_bytes,
            total_units=total_lines or total_bytes,
            unit_name="lines" if total_lines else "bytes",
            start_time=start_time,
            regex_counts=aggregate_regex,
            pii_counts=aggregate_pii,
        ),
        str(zip_path),
        report_preview_rows(file_reports),
        _preview_sanitized_output(sanitized_root, file_reports),
    )


def _build_model_redactor(compute_backend: str, model_batch_size: int) -> PIIRedactor:
    if compute_backend == MODAL_CLOUD_BACKEND:
        return ModalPIIRedactor(batch_size=model_batch_size)
    return OpenMedPIIRedactor()


def _loading_phase(compute_backend: str) -> str:
    if compute_backend == MODAL_CLOUD_BACKEND:
        return "Connecting to Modal cloud GPU backend"
    return "Loading current-runtime OpenMed model"


def update_source_visibility(source_mode: str):
    return (
        gr.update(visible=source_mode == "Known local source"),
        gr.update(visible=source_mode == "Custom local path"),
        gr.update(visible=source_mode == "Upload files"),
        gr.update(visible=source_mode == "Upload directory"),
    )


def update_backend_controls(compute_backend: str):
    if compute_backend == MODAL_CLOUD_BACKEND:
        return (
            gr.update(value=_modal_warning_html(), visible=True),
            gr.update(choices=MODAL_MODEL_OPTIONS, value=MODAL_MODEL_OPTIONS[0]),
        )
    return (
        gr.update(value="", visible=False),
        gr.update(choices=MODEL_OPTIONS, value=MODEL_OPTIONS[0]),
    )


def _app_header_html() -> str:
    return f"""
    <style>{APP_CSS}</style>
    <div class="app-shell">
      <div class="app-topbar">
        <div>
          <div class="eyebrow">Trace Privacy Workbench</div>
          <h1 class="app-title">Agent Trace Privacy Scrubber</h1>
          <p class="app-subtitle">Inspect, redact, and package agent traces with explicit compute boundaries and auditable output reports.</p>
        </div>
        <div class="runtime-ledger" aria-label="Runtime boundaries">
          <div class="ledger-cell"><span>Current runtime</span><strong>Local or Space ZeroGPU</strong></div>
          <div class="ledger-cell"><span>Remote compute</span><strong>Modal opt-in only</strong></div>
        </div>
      </div>
      <div class="flow-rail" aria-label="Workflow">
        <div class="flow-step"><span>01</span>Source</div>
        <div class="flow-step"><span>02</span>Policy</div>
        <div class="flow-step"><span>03</span>Run</div>
        <div class="flow-step"><span>04</span>Review</div>
      </div>
    </div>
    """


def _section_header_html(index: str, title: str) -> str:
    return (
        '<div class="section-heading">'
        f"<h2>{_escape(title)}</h2>"
        f"<span>{_escape(index)}</span>"
        "</div>"
    )


def _modal_warning_html() -> str:
    return _status_panel_html(
        "warning",
        "Remote compute selected",
        "Regex redaction runs first here. Model-enabled string values are then sent to your deployed Modal app.",
    )


def _status_panel_html(
    tone: str,
    title: str,
    detail: str,
    *,
    detail_is_html: bool = False,
    metrics: list[tuple[str, str]] | None = None,
) -> str:
    detail_html = detail if detail_is_html else _escape(detail)
    metrics_html = ""
    if metrics:
        metrics_html = (
            '<div class="metric-grid">'
            + "".join(
                f'<div class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
                for label, value in metrics
            )
            + "</div>"
        )
    return (
        f'<div class="status-panel {_escape(tone)}">'
        f"<strong>{_escape(title)}</strong>"
        f"<p>{detail_html}</p>"
        f"{metrics_html}"
        "</div>"
    )


def _inline_warning_html(message: str) -> str:
    return f'<div class="inline-warning">{_escape(message)}</div>'


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME, fill_width=True) as demo:
        logs_state = gr.State([])

        gr.HTML(_app_header_html())

        with gr.Row(equal_height=False, elem_classes=["workbench-grid"]):
            with gr.Column(scale=4, min_width=360):
                with gr.Group(elem_classes=["tool-panel"]):
                    gr.HTML(_section_header_html("01", "Source"))
                    source_mode = gr.Radio(
                        SOURCE_OPTIONS, value="Known local source", label="Input source"
                    )
                    known_source = gr.Dropdown(
                        list(KNOWN_LOCAL_SOURCES.keys()),
                        value="Codex",
                        label="Known source",
                    )
                    custom_path = gr.Textbox(
                        label="Custom path",
                        placeholder="~/path/to/agent/sessions",
                        visible=False,
                    )
                    uploaded_files = gr.File(
                        label="Upload files",
                        file_count="multiple",
                        type="filepath",
                        file_types=[".jsonl", ".ndjson", ".json"],
                        visible=False,
                    )
                    uploaded_directory = gr.File(
                        label="Upload directory",
                        file_count="directory",
                        type="filepath",
                        file_types=[".jsonl", ".ndjson", ".json"],
                        visible=False,
                    )
                    scan_button = gr.Button("Scan logs", variant="primary", size="md")
                    scan_status = gr.HTML(
                        _status_panel_html(
                            "neutral",
                            "Ready to scan",
                            "Select a source and scan for trace files.",
                        )
                    )

                with gr.Group(elem_classes=["tool-panel"]):
                    gr.HTML(_section_header_html("02", "Redaction policy"))
                    compute_backend = gr.Dropdown(
                        COMPUTE_BACKEND_OPTIONS,
                        value=CURRENT_RUNTIME_BACKEND,
                        label="Compute backend",
                    )
                    model_name = gr.Dropdown(
                        MODEL_OPTIONS,
                        value=MODEL_OPTIONS[0],
                        label="Privacy filter model",
                    )
                    mode = gr.Dropdown(
                        ["mask", "remove", "hash", "replace"],
                        value="mask",
                        label="Redaction mode",
                    )
                    modal_warning = gr.HTML(value="", visible=False)
                    with gr.Row():
                        regex_enabled = gr.Checkbox(True, label="Secret regex")
                        model_enabled = gr.Checkbox(True, label="Model PII")
                    with gr.Row():
                        preserve_json_structure = gr.Checkbox(
                            True, label="Preserve JSON"
                        )
                        include_report = gr.Checkbox(True, label="Detailed report")
                    with gr.Accordion("Advanced", open=False):
                        chunk_size = gr.Slider(
                            1000, 20000, value=6000, step=500, label="Chunk size"
                        )
                        model_batch_size = gr.Slider(
                            1, 128, value=32, step=1, label="Model batch size"
                        )
                        max_file_size_warning_mb = gr.Slider(
                            1, 500, value=50, step=1, label="Large file threshold (MB)"
                        )
                        confidence_threshold = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05, label="Confidence threshold"
                        )
                        seed = gr.Number(
                            value=2026, precision=0, label="Deterministic seed"
                        )

            with gr.Column(scale=7, min_width=560):
                with gr.Group(elem_classes=["tool-panel data-surface"]):
                    gr.HTML(_section_header_html("03", "Selection"))
                    with gr.Row(elem_classes=["compact-actions"]):
                        select_all_button = gr.Button("Select all", size="sm")
                        select_none_button = gr.Button("Clear", size="sm")
                        select_likely_button = gr.Button(
                            "Select JSONL/NDJSON", size="sm"
                        )
                    logs_table = gr.Dataframe(
                        headers=TABLE_HEADERS,
                        datatype=[
                            "bool",
                            "str",
                            "str",
                            "str",
                            "number",
                            "number",
                            "str",
                        ],
                        type="array",
                        row_count=0,
                        interactive=True,
                        label="Discovered logs",
                        wrap=True,
                        max_height=360,
                        show_search="filter",
                        pinned_columns=2,
                        buttons=["copy", "fullscreen"],
                    )

                with gr.Group(elem_classes=["tool-panel"]):
                    gr.HTML(_section_header_html("04", "Run"))
                    with gr.Row(elem_classes=["compact-actions"]):
                        process_button = gr.Button(
                            "Process selected logs",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-run"],
                        )
                        cancel_button = gr.Button("Cancel", variant="stop", size="lg")
                    process_status = gr.HTML(
                        _status_panel_html(
                            "neutral",
                            "Waiting",
                            "Scan and select trace files before processing.",
                        )
                    )

                with gr.Group(elem_classes=["tool-panel data-surface"]):
                    gr.HTML(_section_header_html("05", "Review"))
                    with gr.Tabs():
                        with gr.Tab("Archive"):
                            zip_output = gr.File(label="Sanitized archive")
                        with gr.Tab("Report"):
                            report_table = gr.Dataframe(
                                headers=REPORT_HEADERS,
                                datatype=[
                                    "str",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                    "number",
                                ],
                                type="array",
                                row_count=0,
                                interactive=False,
                                label="Redaction report",
                                max_height=320,
                                show_search="filter",
                                buttons=["copy", "fullscreen"],
                            )
                        with gr.Tab("Preview"):
                            sanitized_preview = gr.Textbox(
                                label="Sanitized preview",
                                lines=12,
                                max_lines=16,
                                interactive=False,
                            )

        gr.HTML(
            f'<div class="footer-note">Version {_escape(APP_VERSION)}. Current-runtime processing stays on this server; Modal is remote compute only when selected.</div>'
        )

        source_mode.change(
            update_source_visibility,
            inputs=[source_mode],
            outputs=[known_source, custom_path, uploaded_files, uploaded_directory],
        )
        compute_backend.change(
            update_backend_controls,
            inputs=[compute_backend],
            outputs=[modal_warning, model_name],
        )
        scan_button.click(
            scan_logs,
            inputs=[
                source_mode,
                known_source,
                custom_path,
                uploaded_files,
                uploaded_directory,
                max_file_size_warning_mb,
            ],
            outputs=[scan_status, logs_table, logs_state],
        )
        select_all_button.click(select_all, inputs=[logs_table], outputs=[logs_table])
        select_none_button.click(select_none, inputs=[logs_table], outputs=[logs_table])
        select_likely_button.click(
            select_likely_trace_files, inputs=[logs_table], outputs=[logs_table]
        )

        run_event = process_button.click(
            process_selected_logs,
            inputs=[
                logs_table,
                logs_state,
                compute_backend,
                model_name,
                mode,
                regex_enabled,
                model_enabled,
                preserve_json_structure,
                include_report,
                chunk_size,
                model_batch_size,
                max_file_size_warning_mb,
                confidence_threshold,
                seed,
            ],
            outputs=[process_status, zip_output, report_table, sanitized_preview],
        )
        cancel_button.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

    return demo


def _normalize_file_inputs(value: list[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    normalized: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized.append(item)
        elif hasattr(item, "name"):
            normalized.append(str(item.name))
    return normalized


def _coerce_rows(table_data: object) -> list[list[object]]:
    if table_data is None:
        return []
    if hasattr(table_data, "values"):
        return table_data.values.tolist()
    if isinstance(table_data, dict) and "data" in table_data:
        return [list(row) for row in table_data["data"]]
    return [list(row) for row in table_data] if isinstance(table_data, list) else []


def _safe_relative_output_path(relative_path: str) -> str:
    path = Path(relative_path)
    parts = [part for part in path.parts if part not in {"", ".", ".."}]
    return Path(*parts).as_posix() if parts else "sanitized.jsonl"


def _status_html(
    *,
    phase: str,
    file_index: int,
    file_total: int,
    current_file: str,
    processed_units: int,
    total_units: int,
    unit_name: str,
    start_time: float,
    regex_counts: Counter[str],
    pii_counts: Counter[str],
    current_line: int | None = None,
    file_lines: int | None = None,
) -> str:
    elapsed = time.monotonic() - start_time
    eta = _eta(elapsed, processed_units, total_units)
    total_label = str(total_units) if total_units else "unknown"
    progress_percent = 0
    if total_units and processed_units <= total_units:
        progress_percent = max(0, min(100, int(processed_units / total_units * 100)))
    tone = "success" if phase == "Complete" else "active"
    line_metric = (
        ("Line", f"{current_line}/{file_lines or 'unknown'}")
        if current_line is not None
        else None
    )
    metrics = [
        ("File", f"{file_index}/{file_total}"),
        ("Progress", f"{processed_units}/{total_label} {unit_name}"),
        ("Elapsed", _format_duration(elapsed)),
        ("ETA", eta),
        ("Regex", str(sum(regex_counts.values()))),
        ("Model PII", str(sum(pii_counts.values()))),
    ]
    if line_metric is not None:
        metrics.insert(2, line_metric)
    metrics_html = (
        '<div class="metric-grid">'
        + "".join(
            f'<div class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></div>'
            for label, value in metrics
        )
        + "</div>"
    )
    if current_line is not None:
        current_file_label = (
            f"{current_file} · line {current_line}/{file_lines or 'unknown'}"
        )
    else:
        current_file_label = current_file
    return (
        f'<div class="run-status {tone}">'
        '<div class="run-status__top">'
        f'<span class="status-pill">{_escape(phase)}</span>'
        f"<strong>{_escape(current_file_label)}</strong>"
        "</div>"
        '<div class="progress-track">'
        f'<span class="progress-fill" style="width: {progress_percent}%"></span>'
        "</div>"
        f"{metrics_html}"
        "</div>"
    )


def _eta(elapsed: float, processed: int, total: int) -> str:
    if not processed or not total or processed > total:
        return "unavailable"
    remaining = total - processed
    if remaining <= 0:
        return "0s"
    return _format_duration(elapsed / processed * remaining)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _preview_sanitized_output(
    sanitized_root: Path, reports: Iterable[FileProcessReport]
) -> str:
    for report in reports:
        output_file = sanitized_root / report.output_relative_path
        if not output_file.is_file():
            continue
        try:
            lines = output_file.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[:8]
        except OSError:
            continue
        preview = "\n".join(lines)
        return preview[:4000]
    return ""


def _inline_code(value: object) -> str:
    return f"<code>{_escape(value)}</code>"


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


if __name__ == "__main__":
    build_app().queue().launch(share=False)
