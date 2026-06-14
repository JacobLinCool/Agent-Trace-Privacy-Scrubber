"""Deterministic secret and local model redaction."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import gc
import hashlib
import re
from typing import Any, Callable, Iterable, Literal, Protocol

RedactionMode = Literal["mask", "remove", "hash", "replace"]


class ModelRedactionError(RuntimeError):
    """Raised when local model redaction cannot run safely."""


@dataclass(frozen=True)
class RedactionConfig:
    """Runtime redaction settings shared by the UI and processors."""

    model_name: str = "OpenMed/privacy-filter-nemotron"
    mode: RedactionMode = "mask"
    regex_enabled: bool = True
    model_enabled: bool = True
    preserve_json_structure: bool = True
    include_report: bool = True
    chunk_size: int = 6000
    confidence_threshold: float = 0.5
    seed: int = 2026


@dataclass
class TextRedactionResult:
    text: str
    regex_counts: Counter[str] = field(default_factory=Counter)
    pii_counts: Counter[str] = field(default_factory=Counter)


class PIIRedactor(Protocol):
    """Model redactor interface used by local and remote backends."""

    def prepare_model(self, config: RedactionConfig) -> None:
        """Load or validate model resources before processing."""
        ...

    def redact(self, text: str, config: RedactionConfig) -> TextRedactionResult:
        """Redact PII spans from one text value."""
        ...

    def release(self) -> None:
        """Release local resources held by the redactor."""
        ...


@dataclass(frozen=True)
class SecretRule:
    name: str
    pattern: re.Pattern[str]
    value_group: int | str | None = None


@dataclass(frozen=True)
class EntitySpan:
    label: str
    start: int
    end: int
    confidence: float


SECRET_RULES: tuple[SecretRule, ...] = (
    SecretRule(
        "private_key_block",
        re.compile(
            r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"
            r"[\s\S]+?"
            r"-----END (?:RSA |DSA |EC |OPENSSH |ENCRYPTED )?PRIVATE KEY-----",
            re.MULTILINE,
        ),
    ),
    SecretRule("anthropic_api_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{16,}\b")),
    SecretRule(
        "openai_api_key",
        re.compile(r"\bsk-(?:proj-|service-account-)?[A-Za-z0-9_-]{20,}\b"),
    ),
    SecretRule("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    SecretRule(
        "github_token",
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    ),
    SecretRule("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    SecretRule(
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
    SecretRule("aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    SecretRule(
        "aws_secret_assignment",
        re.compile(
            r"\b((?:AWS_)?SECRET_ACCESS_KEY)\s*[:=]\s*(['\"]?)([^'\"\s#]+)(['\"]?)",
            re.IGNORECASE,
        ),
        value_group=3,
    ),
    SecretRule(
        "env_secret_assignment",
        re.compile(
            r"\b([A-Z0-9_]*(?:PASSWORD|PASSWD|API[_-]?KEY|TOKEN|SECRET|PRIVATE[_-]?KEY)"
            r"[A-Z0-9_]*)\s*=\s*(['\"]?)([^'\"\s#]+)(['\"]?)",
            re.IGNORECASE,
        ),
        value_group=3,
    ),
    SecretRule(
        "bearer_token",
        re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=-]{16,})", re.IGNORECASE),
        value_group=1,
    ),
    SecretRule(
        "url_query_secret",
        re.compile(
            r"([?&](?:access_token|api[_-]?key|auth|key|password|secret|token)=)([^&#\s]+)",
            re.IGNORECASE,
        ),
        value_group=2,
    ),
    SecretRule(
        "basic_auth_url",
        re.compile(r"\bhttps?://([^/@:\s]+:[^/@\s]+)@"),
        value_group=1,
    ),
)


class SecretRegexRedactor:
    """Apply deterministic regex sweeps for common secrets."""

    def __init__(self, seed: int = 2026) -> None:
        self.seed = seed

    def redact(self, text: str, mode: RedactionMode) -> TextRedactionResult:
        counts: Counter[str] = Counter()
        redacted = text
        for rule in SECRET_RULES:
            redacted = self._apply_rule(redacted, rule, mode, counts)
        return TextRedactionResult(text=redacted, regex_counts=counts)

    def _apply_rule(
        self,
        text: str,
        rule: SecretRule,
        mode: RedactionMode,
        counts: Counter[str],
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            if rule.value_group is None:
                value = match.group(0)
                if _is_placeholder(value):
                    return value
                counts[rule.name] += 1
                return self._replacement(rule.name, value, mode)

            start, end = match.span(rule.value_group)
            relative_start = start - match.start()
            relative_end = end - match.start()
            matched = match.group(0)
            value = match.group(rule.value_group)
            if _is_placeholder(value):
                return matched
            counts[rule.name] += 1
            replacement = self._replacement(rule.name, value, mode)
            return matched[:relative_start] + replacement + matched[relative_end:]

        return rule.pattern.sub(replace, text)

    def _replacement(self, label: str, value: str, mode: RedactionMode) -> str:
        label = label.lower()
        if mode == "remove":
            return ""
        if mode == "hash":
            return f"<HASHED:{label}:{_stable_hash(value, label, self.seed)}>"
        if mode == "replace":
            return f"<REPLACED:{label}:{_stable_hash(value, label, self.seed)}>"
        return f"<REDACTED:{label}>"


class OpenMedPIIRedactor:
    """Local-only OpenMed PII redactor with lazy import and model loading."""

    def __init__(self) -> None:
        self._extract_pii: Callable[..., Any] | None = None
        self._extract_pii_batch: Callable[..., list[Any]] | None = None
        self._create_privacy_filter_pipeline: Callable[[str], Any] | None = None
        self._privacy_backend_selector: Callable[[str], str] | None = None
        self._privacy_model_resolver: Callable[[str, str], str] | None = None
        self._torch_privacy_filter_pipeline_cls: type[Any] | None = None
        self._is_trusted_for_remote_code: Callable[[str], bool] | None = None
        self._privacy_model_detector: Callable[[str], bool] | None = None
        self._anonymizer_cls: type[Any] | None = None
        self._anonymizer: Any | None = None
        self._privacy_filter_pipelines: dict[str, Any] = {}

    def ensure_available(self) -> None:
        if self._extract_pii is not None:
            return
        try:
            from openmed import extract_pii
            from openmed.core.backends import (
                create_privacy_filter_pipeline,
                resolve_privacy_filter_model,
                select_privacy_filter_backend,
            )
            from openmed.core.anonymizer import Anonymizer
            from openmed.core.pii import (
                _extract_pii_batch,
                _is_privacy_filter_artifact_path,
                _looks_like_privacy_filter_identifier,
            )
            from openmed.torch.privacy_filter import (
                PrivacyFilterTorchPipeline,
                is_trusted_for_remote_code,
            )
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ModelRedactionError(
                "OpenMed is required for model-based PII redaction. Install it with "
                '`pip install -U "openmed[hf]"`, or on Apple Silicon use '
                '`pip install -U "openmed[mlx]"`.'
            ) from exc
        self._extract_pii = extract_pii
        self._extract_pii_batch = _extract_pii_batch
        self._create_privacy_filter_pipeline = create_privacy_filter_pipeline
        self._privacy_backend_selector = select_privacy_filter_backend
        self._privacy_model_resolver = resolve_privacy_filter_model
        self._torch_privacy_filter_pipeline_cls = PrivacyFilterTorchPipeline
        self._is_trusted_for_remote_code = is_trusted_for_remote_code
        self._privacy_model_detector = lambda model_name: (
            _looks_like_privacy_filter_identifier(model_name)
            or _is_privacy_filter_artifact_path(model_name)
        )
        self._anonymizer_cls = Anonymizer

    def prepare_model(self, config: RedactionConfig) -> None:
        """Load reusable model resources before line-by-line processing."""

        self.ensure_available()
        if self._uses_privacy_filter_model(config.model_name):
            self._get_privacy_filter_pipeline(config.model_name)

    def release(self) -> None:
        """Release cached local model resources and accelerator caches."""

        self._privacy_filter_pipelines.clear()
        self._anonymizer = None
        gc.collect()
        self._clear_mlx_cache()
        self._clear_torch_cache()

    def redact(self, text: str, config: RedactionConfig) -> TextRedactionResult:
        if not text:
            return TextRedactionResult(text=text)
        self.ensure_available()

        spans = self._extract_spans(text, config)
        if not spans:
            return TextRedactionResult(text=text)

        counts: Counter[str] = Counter(span.label for span in spans)
        redacted = text
        for span in sorted(spans, key=lambda item: item.start, reverse=True):
            original = redacted[span.start : span.end]
            replacement = self._replacement(span.label, original, config)
            redacted = redacted[: span.start] + replacement + redacted[span.end :]
        return TextRedactionResult(text=redacted, pii_counts=counts)

    def _extract_spans(self, text: str, config: RedactionConfig) -> list[EntitySpan]:
        chunk_size = max(1000, int(config.chunk_size))
        if len(text) <= chunk_size:
            return self._extract_chunk_spans(text, 0, config)

        overlap = min(256, max(32, chunk_size // 20))
        all_spans: list[EntitySpan] = []
        start = 0
        while start < len(text):
            end = _chunk_boundary(text, start, chunk_size)
            chunk = text[start:end]
            all_spans.extend(self._extract_chunk_spans(chunk, start, config))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return _dedupe_overlapping_spans(all_spans)

    def _extract_chunk_spans(
        self,
        chunk: str,
        offset: int,
        config: RedactionConfig,
    ) -> list[EntitySpan]:
        assert self._extract_pii is not None
        try:
            if self._uses_privacy_filter_model(config.model_name):
                assert self._extract_pii_batch is not None
                result = self._extract_pii_batch(
                    [chunk],
                    model_name=config.model_name,
                    confidence_threshold=config.confidence_threshold,
                    privacy_filter_pipeline=self._get_privacy_filter_pipeline(config.model_name),
                )[0]
            else:
                result = self._extract_pii(
                    chunk,
                    model_name=config.model_name,
                    confidence_threshold=config.confidence_threshold,
                )
        except Exception as exc:  # pragma: no cover - depends on model runtime
            raise ModelRedactionError(
                "Local OpenMed model redaction failed. Verify the selected model, "
                "install extras with `openmed[hf]` or `openmed[mlx]`, and ensure the "
                "machine has enough memory for the model."
            ) from exc

        spans: list[EntitySpan] = []
        for entity in getattr(result, "entities", []):
            try:
                start = int(getattr(entity, "start"))
                end = int(getattr(entity, "end"))
                confidence = float(getattr(entity, "confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if start < 0 or end <= start or end > len(chunk):
                continue
            if confidence < config.confidence_threshold:
                continue
            label = _normalize_label(str(getattr(entity, "label", "pii")))
            spans.append(EntitySpan(label=label, start=offset + start, end=offset + end, confidence=confidence))
        return spans

    def _replacement(self, label: str, value: str, config: RedactionConfig) -> str:
        if config.mode == "remove":
            return ""
        if config.mode == "hash":
            return f"<HASHED:{label}:{_stable_hash(value, label, config.seed)}>"
        if config.mode == "replace":
            if self._anonymizer is None:
                assert self._anonymizer_cls is not None
                self._anonymizer = self._anonymizer_cls(
                    consistent=True,
                    seed=config.seed,
                )
            try:
                return str(self._anonymizer.surrogate(value, label))
            except Exception as exc:  # pragma: no cover - optional faker path
                raise ModelRedactionError("OpenMed replacement redaction failed.") from exc
        return f"<REDACTED:{label}>"

    def _uses_privacy_filter_model(self, model_name: str) -> bool:
        if self._privacy_model_detector is None:
            self.ensure_available()
        assert self._privacy_model_detector is not None
        return self._privacy_model_detector(model_name)

    def _get_privacy_filter_pipeline(self, model_name: str) -> Any:
        pipeline = self._privacy_filter_pipelines.get(model_name)
        if pipeline is not None:
            return pipeline

        if self._create_privacy_filter_pipeline is None:
            self.ensure_available()
        pipeline = self._build_privacy_filter_pipeline(model_name)
        self._privacy_filter_pipelines[model_name] = pipeline
        return pipeline

    def _build_privacy_filter_pipeline(self, model_name: str) -> Any:
        assert self._create_privacy_filter_pipeline is not None
        if self._privacy_backend_selector is None:
            self.ensure_available()
        assert self._privacy_backend_selector is not None

        backend = self._privacy_backend_selector(model_name)
        if backend == "mlx":
            return self._create_privacy_filter_pipeline(model_name)

        if self._privacy_model_resolver is None or self._torch_privacy_filter_pipeline_cls is None:
            self.ensure_available()
        assert self._privacy_model_resolver is not None
        assert self._torch_privacy_filter_pipeline_cls is not None
        assert self._is_trusted_for_remote_code is not None

        actual_model = self._privacy_model_resolver(model_name, "torch")
        return self._torch_privacy_filter_pipeline_cls(
            actual_model,
            device=self._select_torch_device(),
            trust_remote_code=self._is_trusted_for_remote_code(actual_model),
        )

    def _select_torch_device(self) -> str:
        try:
            import torch
        except Exception:
            return "cpu"

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_available", None)):
            try:
                if cuda.is_available():
                    return "cuda"
            except Exception:
                pass

        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        if mps_backend is not None and callable(getattr(mps_backend, "is_available", None)):
            try:
                if mps_backend.is_available():
                    return "mps"
            except Exception:
                pass

        return "cpu"

    def _clear_mlx_cache(self) -> None:
        try:
            import mlx.core as mx
        except Exception:
            return

        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
                return
            except Exception:
                pass

        metal = getattr(mx, "metal", None)
        if metal is not None:
            clear_cache = getattr(metal, "clear_cache", None)
            if callable(clear_cache):
                try:
                    clear_cache()
                except Exception:
                    pass

    def _clear_torch_cache(self) -> None:
        try:
            import torch
        except Exception:
            return

        cuda = getattr(torch, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "empty_cache", None)):
            try:
                cuda.empty_cache()
            except Exception:
                pass

        mps = getattr(torch, "mps", None)
        if mps is not None and callable(getattr(mps, "empty_cache", None)):
            try:
                mps.empty_cache()
            except Exception:
                pass


def sanitize_text(
    text: str,
    config: RedactionConfig,
    model_redactor: PIIRedactor | None = None,
) -> TextRedactionResult:
    """Run configured redaction passes over one text string."""

    regex_counts: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    redacted = text

    if config.regex_enabled:
        regex_result = SecretRegexRedactor(seed=config.seed).redact(redacted, config.mode)
        redacted = regex_result.text
        regex_counts.update(regex_result.regex_counts)

    if config.model_enabled:
        model = model_redactor or OpenMedPIIRedactor()
        pii_result = model.redact(redacted, config)
        redacted = pii_result.text
        pii_counts.update(pii_result.pii_counts)

    if config.regex_enabled:
        regex_result = SecretRegexRedactor(seed=config.seed).redact(redacted, config.mode)
        redacted = regex_result.text
        regex_counts.update(regex_result.regex_counts)

    return TextRedactionResult(text=redacted, regex_counts=regex_counts, pii_counts=pii_counts)


def merge_text_results(results: Iterable[TextRedactionResult]) -> TextRedactionResult:
    merged = TextRedactionResult(text="")
    chunks: list[str] = []
    for result in results:
        chunks.append(result.text)
        merged.regex_counts.update(result.regex_counts)
        merged.pii_counts.update(result.pii_counts)
    merged.text = "".join(chunks)
    return merged


def _stable_hash(value: str, label: str, seed: int) -> str:
    material = f"{seed}|{label}|{value}".encode("utf-8", errors="replace")
    return hashlib.sha256(material).hexdigest()[:10]


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return normalized or "pii"


def _is_placeholder(value: str) -> bool:
    return value.startswith(("<REDACTED:", "<HASHED:", "<REPLACED:"))


def _chunk_boundary(text: str, start: int, chunk_size: int) -> int:
    target = min(len(text), start + chunk_size)
    if target >= len(text):
        return len(text)
    boundary_window_start = max(start + chunk_size // 2, target - 500)
    for index in range(target, boundary_window_start, -1):
        if text[index - 1].isspace():
            return index
    return target


def _dedupe_overlapping_spans(spans: list[EntitySpan]) -> list[EntitySpan]:
    if not spans:
        return []

    selected: list[EntitySpan] = []
    for span in sorted(spans, key=lambda item: (item.start, -(item.end - item.start), -item.confidence)):
        if not selected:
            selected.append(span)
            continue

        last = selected[-1]
        if span.start >= last.end:
            selected.append(span)
            continue

        last_score = (last.end - last.start, last.confidence)
        span_score = (span.end - span.start, span.confidence)
        if span_score > last_score:
            selected[-1] = span
    return selected
