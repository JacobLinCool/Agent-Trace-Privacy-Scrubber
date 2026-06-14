"""Zip package creation for sanitized traces."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .jsonl_processor import FileProcessReport
from .redactors import RedactionConfig
from .reporting import DISCLAIMER, build_redaction_report

TEMP_PREFIX = "trace-scrubber-"

README_FIRST = f"""Local Agent Trace Privacy Scrubber

{DISCLAIMER}

This archive contains sanitized JSONL files and redaction_report.json. The
report contains counts and categories only; it intentionally does not include
raw matched values.

For real private logs, run the app locally and inspect outputs before upload.
Do not publish this archive until you have reviewed the sanitized traces.
"""


def make_output_workspace() -> Path:
    """Create a temp workspace and opportunistically remove stale ones."""

    cleanup_old_workspaces()
    return Path(tempfile.mkdtemp(prefix=TEMP_PREFIX))


def cleanup_old_workspaces(max_age_hours: int = 6) -> None:
    """Remove old scrubber temp directories while keeping recent downloads."""

    temp_root = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_hours * 3600
    for child in temp_root.iterdir():
        if not child.name.startswith(TEMP_PREFIX) or not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def build_zip_archive(
    output_root: Path,
    file_reports: list[FileProcessReport],
    config: RedactionConfig,
    zip_name: str = "sanitized_agent_traces.zip",
) -> Path:
    """Write report files and zip sanitized output paths."""

    package_root = output_root / "package"
    sanitized_root = output_root / "sanitized"
    package_root.mkdir(parents=True, exist_ok=True)

    report = build_redaction_report(file_reports, config)
    (package_root / "redaction_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (package_root / "README_FIRST.txt").write_text(README_FIRST, encoding="utf-8")

    zip_path = output_root / zip_name
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        for report_item in file_reports:
            output_file = sanitized_root / report_item.output_relative_path
            if output_file.is_file():
                archive.write(output_file, arcname=report_item.output_relative_path)
        archive.write(package_root / "redaction_report.json", arcname="redaction_report.json")
        archive.write(package_root / "README_FIRST.txt", arcname="README_FIRST.txt")
    return zip_path
