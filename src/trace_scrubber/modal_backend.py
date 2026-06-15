"""Modal cloud redaction backend."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from .redactors import ModelRedactionError, RedactionConfig, TextRedactionResult

MODAL_APP_NAME = "agent-trace-privacy-scrubber"
MODAL_FUNCTION_NAME = "redact_text_batch"
# Strings sent per remote round-trip. This is the *network request* batch and is
# intentionally decoupled from ``config.model_batch_size`` (the per-forward-pass
# GPU batch that bounds device memory): packing many short strings into one call
# amortizes Modal's per-invocation overhead without enlarging the GPU batch.
DEFAULT_MODAL_BATCH_SIZE = 256
# Cap the cumulative characters per round-trip so a few multi-MB strings get
# their own call instead of bloating one payload (and so a giant string never
# starves a batch of small ones).
DEFAULT_REQUEST_MAX_CHARS = 1_000_000

FunctionLookup = Callable[[str, str], Any]


class ModalPIIRedactor:
    """PII redactor that delegates model inference to a deployed Modal function."""

    def __init__(
        self,
        *,
        app_name: str = MODAL_APP_NAME,
        function_name: str = MODAL_FUNCTION_NAME,
        batch_size: int = DEFAULT_MODAL_BATCH_SIZE,
        request_max_chars: int = DEFAULT_REQUEST_MAX_CHARS,
        function_lookup: FunctionLookup | None = None,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
        self.batch_size = max(1, int(batch_size))
        self.request_max_chars = max(1, int(request_max_chars))
        self._function_lookup = function_lookup
        self._function: Any | None = None

    def prepare_model(self, config: RedactionConfig) -> None:
        if self._function is not None:
            return

        try:
            self._function = self._lookup_function()
        except ModelRedactionError:
            raise
        except Exception as exc:
            raise ModelRedactionError(
                "Modal backend is not ready. Run `modal token new`, then deploy the "
                "remote worker with `modal deploy modal_app.py`."
            ) from exc

    def redact(self, text: str, config: RedactionConfig) -> TextRedactionResult:
        return self.redact_many([text], config)[0]

    def redact_many(
        self, texts: list[str], config: RedactionConfig
    ) -> list[TextRedactionResult]:
        if not texts:
            return []
        self.prepare_model(config)
        assert self._function is not None

        results: list[TextRedactionResult] = []
        for batch in _batched(texts, self.batch_size, self.request_max_chars):
            try:
                payload = self._function.remote(batch, _modal_settings(config))
            except Exception as exc:
                raise ModelRedactionError(
                    "Modal cloud redaction failed. Verify Modal credentials, deployment "
                    "status, and remote GPU availability."
                ) from exc
            results.extend(_coerce_modal_results(payload, expected_count=len(batch)))
        return results

    def release(self) -> None:
        """No local model resources are held by the Modal client."""

    def _lookup_function(self) -> Any:
        if self._function_lookup is not None:
            return self._function_lookup(self.app_name, self.function_name)

        try:
            import modal
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise ModelRedactionError(
                "Modal backend selected, but the `modal` package is not installed. "
                "Install it with `pip install modal`."
            ) from exc

        return modal.Function.from_name(self.app_name, self.function_name)


def _modal_settings(config: RedactionConfig) -> dict[str, object]:
    return {
        "model_name": config.model_name,
        "mode": config.mode,
        "chunk_size": int(config.chunk_size),
        "model_batch_size": int(config.model_batch_size),
        "confidence_threshold": float(config.confidence_threshold),
        "seed": int(config.seed),
    }


def _coerce_modal_results(
    payload: object, *, expected_count: int
) -> list[TextRedactionResult]:
    if not isinstance(payload, list) or len(payload) != expected_count:
        raise ModelRedactionError("Modal cloud redaction returned an invalid response.")

    results: list[TextRedactionResult] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ModelRedactionError(
                "Modal cloud redaction returned an invalid response item."
            )
        text = item.get("text")
        if not isinstance(text, str):
            raise ModelRedactionError(
                "Modal cloud redaction returned a response without text."
            )
        results.append(
            TextRedactionResult(
                text=text,
                pii_counts=Counter(_coerce_count_mapping(item.get("pii_counts", {}))),
            )
        )
    return results


def _coerce_modal_result(payload: object) -> TextRedactionResult:
    return _coerce_modal_results(payload, expected_count=1)[0]


def _coerce_count_mapping(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}

    counts: dict[str, int] = {}
    for key, count in value.items():
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            continue
        if normalized_count > 0:
            counts[str(key)] = normalized_count
    return counts


def _batched(
    items: list[str], batch_size: int, max_chars: int | None = None
) -> list[list[str]]:
    """Group items by count and (optionally) cumulative character budget.

    A single item larger than ``max_chars`` still gets its own batch rather than
    being dropped, so oversized strings never starve smaller ones.
    """

    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for item in items:
        item_chars = len(item)
        if current and (
            len(current) >= batch_size
            or (max_chars is not None and current_chars + item_chars > max_chars)
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches
