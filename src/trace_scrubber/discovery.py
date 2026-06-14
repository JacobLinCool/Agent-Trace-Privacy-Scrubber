"""Trace file discovery helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence

TRACE_EXTENSIONS = {".jsonl", ".ndjson", ".json"}
SKIPPED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}

KNOWN_LOCAL_SOURCES = {
    "Codex": Path("~/.codex/sessions").expanduser(),
    "Claude Code": Path("~/.claude/projects").expanduser(),
    "Pi Agent": Path("~/.pi/agent/sessions").expanduser(),
}


@dataclass(frozen=True)
class LogFile:
    """Metadata for one discoverable trace file."""

    path: str
    relative_path: str
    filename: str
    size_bytes: int
    line_count: int
    agent_guess: str
    status: str

    @property
    def size_kb(self) -> float:
        return round(self.size_bytes / 1024, 2)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["size_kb"] = self.size_kb
        return payload


def guess_agent_type(path: str | Path) -> str:
    """Infer the source agent from a path without reading file contents."""

    normalized = str(path).lower()
    if ".codex/sessions" in normalized or "codex" in normalized:
        return "Codex"
    if ".claude/projects" in normalized or "claude" in normalized:
        return "Claude Code"
    if ".pi/agent/sessions" in normalized or "/pi/" in normalized or "pi-agent" in normalized:
        return "Pi Agent"
    return "Unknown"


def discover_known_source(source_name: str, max_file_size_warning_mb: int = 50) -> list[LogFile]:
    """Discover logs from one of the known local agent directories."""

    root = KNOWN_LOCAL_SOURCES[source_name]
    return discover_roots([root], max_file_size_warning_mb=max_file_size_warning_mb)


def discover_custom_path(path: str, max_file_size_warning_mb: int = 50) -> list[LogFile]:
    """Discover logs below a user-provided local filesystem path."""

    expanded = Path(path).expanduser()
    return discover_roots([expanded], max_file_size_warning_mb=max_file_size_warning_mb)


def discover_uploaded_files(
    paths: Sequence[str | Path] | None,
    max_file_size_warning_mb: int = 50,
) -> list[LogFile]:
    """Discover trace files from browser-uploaded files or directories."""

    clean_paths = [Path(path) for path in paths or [] if path]
    if not clean_paths:
        return []

    common_root = _common_parent(clean_paths)
    return _build_log_files(
        _iter_candidate_files(clean_paths),
        root=common_root,
        max_file_size_warning_mb=max_file_size_warning_mb,
    )


def discover_roots(
    roots: Sequence[str | Path],
    max_file_size_warning_mb: int = 50,
) -> list[LogFile]:
    """Recursively discover trace-like files under one or more roots."""

    discovered: list[LogFile] = []
    for raw_root in roots:
        root = Path(raw_root).expanduser()
        if not root.exists():
            continue
        files = _iter_candidate_files([root])
        base = root if root.is_dir() else root.parent
        discovered.extend(
            _build_log_files(
                files,
                root=base,
                max_file_size_warning_mb=max_file_size_warning_mb,
            )
        )
    return sorted(discovered, key=lambda item: item.relative_path)


def rows_for_table(logs: Sequence[LogFile | dict[str, object]], selected: bool = True) -> list[list[object]]:
    """Convert discovered logs into Gradio Dataframe rows."""

    rows: list[list[object]] = []
    for entry in logs:
        data = entry.to_dict() if isinstance(entry, LogFile) else entry
        rows.append(
            [
                selected,
                data["agent_guess"],
                data["relative_path"],
                data["filename"],
                data["size_kb"],
                data["line_count"],
                data["status"],
            ]
        )
    return rows


def selected_relative_paths(table_data: object) -> set[str]:
    """Read selected paths from a Gradio Dataframe value."""

    rows = _coerce_table_rows(table_data)
    selected: set[str] = set()
    for row in rows:
        if len(row) < 3:
            continue
        if _truthy(row[0]):
            selected.add(str(row[2]))
    return selected


def _build_log_files(
    files: Iterable[Path],
    root: Path,
    max_file_size_warning_mb: int,
) -> list[LogFile]:
    warning_threshold = max_file_size_warning_mb * 1024 * 1024
    logs: list[LogFile] = []
    seen: set[Path] = set()
    for path in files:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)

        if not _is_trace_candidate(path) or _looks_binary(path):
            continue

        try:
            size_bytes = path.stat().st_size
            line_count = count_lines(path)
        except OSError:
            continue

        relative_path = _safe_relative_path(path, root)
        status = "ready"
        if size_bytes > warning_threshold:
            status = f"large file warning > {max_file_size_warning_mb} MB"

        logs.append(
            LogFile(
                path=str(path),
                relative_path=relative_path,
                filename=path.name,
                size_bytes=size_bytes,
                line_count=line_count,
                agent_guess=guess_agent_type(path),
                status=status,
            )
        )
    return logs


def _iter_candidate_files(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_file():
            yield expanded
        elif expanded.is_dir():
            yield from _walk_directory(expanded)


def _walk_directory(root: Path) -> Iterable[Path]:
    for child in root.iterdir():
        if child.is_dir():
            if child.name in SKIPPED_DIR_NAMES:
                continue
            if child.name.startswith(".") and child != root:
                continue
            yield from _walk_directory(child)
        elif child.is_file():
            yield child


def _is_trace_candidate(path: Path) -> bool:
    if path.suffix.lower() not in TRACE_EXTENSIONS:
        return False
    if path.name.startswith("."):
        return False
    return True


def _looks_binary(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(4096)
    except OSError:
        return True
    return b"\x00" in chunk


def count_lines(path: Path) -> int:
    """Count lines in a file without loading it all into memory."""

    line_count = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    if path.stat().st_size > 0 and last_byte != b"\n":
        line_count += 1
    return line_count


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _common_parent(paths: Sequence[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].parent if paths[0].is_file() else paths[0]
    resolved = [str(path.resolve()) for path in paths]
    return Path(os.path.commonpath(resolved))


def _coerce_table_rows(table_data: object) -> list[list[object]]:
    if table_data is None:
        return []
    if hasattr(table_data, "values"):
        return table_data.values.tolist()
    if isinstance(table_data, dict) and "data" in table_data:
        return list(table_data["data"])
    return list(table_data) if isinstance(table_data, list) else []


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "selected"}
    return bool(value)
