"""Modal deployment for cloud GPU PII redaction.

Deploy with:

    modal deploy modal_app.py
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any

import modal

APP_NAME = "agent-trace-privacy-scrubber"
FUNCTION_NAME = "redact_text_batch"

MODEL_CACHE_PATH = "/cache"
SRC_PATH = "/root/src"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name("agent-trace-privacy-scrubber-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openmed[hf]>=1.5.5",
        "torch>=2.0",
        "transformers>=4.50",
        "huggingface-hub>=0.30",
        "tqdm>=4.66",
    )
    .add_local_dir("src", remote_path=SRC_PATH, copy=True)
)

_redactor: Any | None = None
_prepared_signature: tuple[str, str, int] | None = None
_cache_committed = False


@app.function(
    image=image,
    gpu=["L4", "A10G"],
    timeout=1800,
    volumes={MODEL_CACHE_PATH: model_cache},
    env={
        "HF_HOME": f"{MODEL_CACHE_PATH}/huggingface",
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_PATH}/huggingface",
    },
    name=FUNCTION_NAME,
)
def redact_text_batch(texts: list[str], settings: dict[str, object]) -> list[dict[str, object]]:
    """Run model-only OpenMed redaction on Modal CUDA hardware."""

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    from trace_scrubber.redactors import OpenMedPIIRedactor, RedactionConfig

    config = RedactionConfig(
        model_name=str(settings["model_name"]),
        mode=str(settings["mode"]),  # type: ignore[arg-type]
        regex_enabled=False,
        model_enabled=True,
        preserve_json_structure=True,
        include_report=True,
        chunk_size=int(settings["chunk_size"]),
        model_batch_size=int(settings["model_batch_size"]),
        confidence_threshold=float(settings["confidence_threshold"]),
        seed=int(settings["seed"]),
    )
    if config.model_name != "OpenMed/privacy-filter-nemotron":
        raise ValueError("Modal cloud GPU backend supports OpenMed/privacy-filter-nemotron only.")

    redactor = _get_redactor(OpenMedPIIRedactor, config)
    responses: list[dict[str, object]] = []
    for result in redactor.redact_many(texts, config):
        responses.append(
            {
                "text": result.text,
                "pii_counts": dict(Counter(result.pii_counts)),
            }
        )
    return responses


def _get_redactor(redactor_cls: type[Any], config: Any) -> Any:
    global _cache_committed, _prepared_signature, _redactor

    signature = (config.model_name, config.mode, config.seed)
    if _redactor is None or _prepared_signature != signature:
        if _redactor is not None:
            _redactor.release()
        _redactor = redactor_cls()
        _redactor.prepare_model(config)
        _prepared_signature = signature
        if not _cache_committed:
            model_cache.commit()
            _cache_committed = True

    return _redactor
