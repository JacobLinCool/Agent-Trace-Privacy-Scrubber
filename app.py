from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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
from trace_scrubber.redactors import ModelRedactionError, OpenMedPIIRedactor, RedactionConfig
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
    "OpenMed/privacy-filter-nemotron-mlx-8bit",
]

SOURCE_OPTIONS = [
    "Known local source",
    "Custom local path",
    "Upload files",
    "Upload directory",
    "Use sample logs",
]


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
                return "Enter a local path before scanning.", [], []
            logs = discover_custom_path(custom_path, max_file_size_warning_mb)
            target = custom_path
        elif source_mode == "Upload files":
            logs = discover_uploaded_files(_normalize_file_inputs(uploaded_files), max_file_size_warning_mb)
            target = "uploaded files"
        elif source_mode == "Upload directory":
            logs = discover_uploaded_files(_normalize_file_inputs(uploaded_directory), max_file_size_warning_mb)
            target = "uploaded directory"
        else:
            sample_root = ROOT / "sample_logs"
            logs = discover_roots([sample_root], max_file_size_warning_mb)
            target = str(sample_root)
    except Exception as exc:
        return f"Scan failed: {_safe_error(exc)}", [], []

    if not logs:
        return f"No JSONL/NDJSON/JSON trace files found in `{target}`.", [], []

    rows = rows_for_table(logs, selected=True)
    state = [log.to_dict() for log in logs]
    total_kb = sum(float(row[4]) for row in rows)
    return (
        f"Found **{len(rows)}** log file(s) in `{target}` ({total_kb:.1f} KB total).",
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
    model_name: str,
    mode: str,
    regex_enabled: bool,
    model_enabled: bool,
    preserve_json_structure: bool,
    include_report: bool,
    chunk_size: int,
    max_file_size_warning_mb: int,
    confidence_threshold: float,
    seed: int,
    progress: gr.Progress = gr.Progress(),
):
    if not regex_enabled and not model_enabled:
        yield (
            "Enable the deterministic regex sweep, model-based PII redaction, or both before processing.",
            None,
            [],
            "",
        )
        return

    logs = logs_state or []
    selected = selected_relative_paths(table_data)
    selected_logs = [log for log in logs if str(log.get("relative_path")) in selected]
    if not selected_logs:
        yield "Select at least one log file before processing.", None, [], ""
        return

    config = RedactionConfig(
        model_name=model_name,
        mode=mode,  # type: ignore[arg-type]
        regex_enabled=regex_enabled,
        model_enabled=model_enabled,
        preserve_json_structure=preserve_json_structure,
        include_report=include_report,
        chunk_size=int(chunk_size),
        confidence_threshold=float(confidence_threshold),
        seed=int(seed),
    )

    workspace = make_output_workspace()
    sanitized_root = workspace / "sanitized"
    model_redactor = OpenMedPIIRedactor()
    file_reports: list[FileProcessReport] = []
    total_lines = sum(int(log.get("line_count") or 0) for log in selected_logs)
    total_bytes = sum(int(log.get("size_bytes") or 0) for log in selected_logs)
    start_time = time.monotonic()
    aggregate_regex: Counter[str] = Counter()
    aggregate_pii: Counter[str] = Counter()

    if model_enabled:
        yield (
            _status_markdown(
                phase="Loading local OpenMed model",
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
            model_redactor.ensure_available()
        except ModelRedactionError as exc:
            yield f"Model redaction is unavailable: {exc}", None, [], ""
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
            large_warning = f"\n\nWarning: `{relative_path}` is larger than {max_file_size_warning_mb} MB."

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
                    progress(min(fraction, 1.0), desc=f"{relative_path} line {event.progress.line_number}")
                    regex_counts = aggregate_regex + event.progress.counts_by_regex_secret_rule
                    pii_counts = aggregate_pii + event.progress.counts_by_pii_label
                    yield (
                        _status_markdown(
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
            yield f"Model redaction failed: {exc}", None, report_preview_rows(file_reports), ""
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
        yield "No files were processed.", None, [], ""
        return

    zip_path = build_zip_archive(workspace, file_reports, config)
    progress(1.0, desc="Archive ready")
    yield (
        _status_markdown(
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


def update_source_visibility(source_mode: str):
    return (
        gr.update(visible=source_mode == "Known local source"),
        gr.update(visible=source_mode == "Custom local path"),
        gr.update(visible=source_mode == "Upload files"),
        gr.update(visible=source_mode == "Upload directory"),
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title=APP_NAME) as demo:
        logs_state = gr.State([])

        gr.Markdown(
            f"# {APP_NAME}\n"
            "Inspect and sanitize local Codex, Claude Code, and Pi Agent JSONL trace files before publishing."
        )
        gr.Markdown(
            "**Local-first privacy scrubber. Real traces should be processed on your own machine. "
            "This app does not use remote inference APIs.**"
        )
        gr.Markdown(
            "Local path mode reads the filesystem of the machine running this Gradio server. "
            "On a public Space, use only sample or non-sensitive uploaded logs."
        )

        with gr.Group():
            gr.Markdown("### Source")
            source_mode = gr.Radio(SOURCE_OPTIONS, value="Known local source", label="Source input")
            known_source = gr.Dropdown(
                list(KNOWN_LOCAL_SOURCES.keys()),
                value="Codex",
                label="Known local source",
                info="Default trace directories from Hugging Face Agent Traces docs.",
            )
            custom_path = gr.Textbox(
                label="Custom local path",
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

        with gr.Group():
            gr.Markdown("### Scan")
            scan_status = gr.Markdown("No scan has run yet.")
            with gr.Row():
                scan_button = gr.Button("Scan logs", variant="primary")
                select_all_button = gr.Button("Select all")
                select_none_button = gr.Button("Select none")
                select_likely_button = gr.Button("Select likely trace files only")
            logs_table = gr.Dataframe(
                headers=TABLE_HEADERS,
                datatype=["bool", "str", "str", "str", "number", "number", "str"],
                type="array",
                row_count=0,
                interactive=True,
                label="Discovered logs",
                wrap=True,
            )

        with gr.Group():
            gr.Markdown("### Settings")
            with gr.Row():
                model_name = gr.Dropdown(MODEL_OPTIONS, value=MODEL_OPTIONS[0], label="Privacy filter model")
                mode = gr.Dropdown(["mask", "remove", "hash", "replace"], value="mask", label="Redaction mode")
            with gr.Row():
                regex_enabled = gr.Checkbox(True, label="Run deterministic secret regex sweep")
                model_enabled = gr.Checkbox(True, label="Run model-based PII redaction")
                preserve_json_structure = gr.Checkbox(True, label="Preserve JSON structure")
                include_report = gr.Checkbox(True, label="Include detailed redaction report")
            with gr.Accordion("Advanced", open=False):
                chunk_size = gr.Slider(1000, 20000, value=6000, step=500, label="Chunk size")
                max_file_size_warning_mb = gr.Slider(1, 500, value=50, step=1, label="Max file size warning threshold (MB)")
                confidence_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="OpenMed confidence threshold")
                seed = gr.Number(value=42, precision=0, label="Seed for deterministic replacement/hash")

        with gr.Group():
            gr.Markdown("### Processing")
            with gr.Row():
                process_button = gr.Button("Process selected logs", variant="primary")
                cancel_button = gr.Button("Cancel")
            process_status = gr.Markdown("Waiting for selected logs.")

        with gr.Group():
            gr.Markdown("### Output")
            zip_output = gr.File(label="Download sanitized zip archive")
            report_table = gr.Dataframe(
                headers=REPORT_HEADERS,
                datatype=["str", "number", "number", "number", "number", "number", "number", "number"],
                type="array",
                row_count=0,
                interactive=False,
                label="Report preview",
            )
            sanitized_preview = gr.Textbox(
                label="Sanitized preview",
                lines=10,
                max_lines=14,
                interactive=False,
            )

        gr.Markdown(f"Version {APP_VERSION}. For real private logs, run this locally. Do not upload sensitive traces to a public Space.")

        source_mode.change(
            update_source_visibility,
            inputs=[source_mode],
            outputs=[known_source, custom_path, uploaded_files, uploaded_directory],
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
        select_likely_button.click(select_likely_trace_files, inputs=[logs_table], outputs=[logs_table])

        run_event = process_button.click(
            process_selected_logs,
            inputs=[
                logs_table,
                logs_state,
                model_name,
                mode,
                regex_enabled,
                model_enabled,
                preserve_json_structure,
                include_report,
                chunk_size,
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


def _status_markdown(
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
    line_status = ""
    if current_line is not None:
        line_status = f"\n- Current line: {current_line}/{file_lines or 'unknown'}"
    return (
        f"- Phase: **{phase}**\n"
        f"- Current file: {file_index}/{file_total} `{current_file}`"
        f"{line_status}\n"
        f"- Progress: {processed_units}/{total_units or 'unknown'} {unit_name}\n"
        f"- Elapsed: {_format_duration(elapsed)}\n"
        f"- ETA: {eta}\n"
        f"- Regex redactions so far: {sum(regex_counts.values())}\n"
        f"- Model PII redactions so far: {sum(pii_counts.values())}"
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


def _preview_sanitized_output(sanitized_root: Path, reports: Iterable[FileProcessReport]) -> str:
    for report in reports:
        output_file = sanitized_root / report.output_relative_path
        if not output_file.is_file():
            continue
        try:
            lines = output_file.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
        except OSError:
            continue
        preview = "\n".join(lines)
        return preview[:4000]
    return ""


def _safe_error(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else exc.__class__.__name__


if __name__ == "__main__":
    build_app().queue().launch(share=False)
