"""Modal cloud redaction backend."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from .redactors import ModelRedactionError, RedactionConfig, TextRedactionResult

MODAL_APP_NAME = "agent-trace-privacy-scrubber"
MODAL_FUNCTION_NAME = "redact_text_batch"

FunctionLookup = Callable[[str, str], Any]


class ModalPIIRedactor:
    """PII redactor that delegates model inference to a deployed Modal function."""

    def __init__(
        self,
        *,
        app_name: str = MODAL_APP_NAME,
        function_name: str = MODAL_FUNCTION_NAME,
        function_lookup: FunctionLookup | None = None,
    ) -> None:
        self.app_name = app_name
        self.function_name = function_name
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
        if not text:
            return TextRedactionResult(text=text)

        self.prepare_model(config)
        assert self._function is not None
        try:
            payload = self._function.remote([text], _modal_settings(config))
        except Exception as exc:
            raise ModelRedactionError(
                "Modal cloud redaction failed. Verify Modal credentials, deployment "
                "status, and remote GPU availability."
            ) from exc

        return _coerce_modal_result(payload)

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
        "confidence_threshold": float(config.confidence_threshold),
        "seed": int(config.seed),
    }


def _coerce_modal_result(payload: object) -> TextRedactionResult:
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ModelRedactionError("Modal cloud redaction returned an invalid response.")

    item = payload[0]
    text = item.get("text")
    if not isinstance(text, str):
        raise ModelRedactionError("Modal cloud redaction returned a response without text.")

    return TextRedactionResult(
        text=text,
        pii_counts=Counter(_coerce_count_mapping(item.get("pii_counts", {}))),
    )


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
