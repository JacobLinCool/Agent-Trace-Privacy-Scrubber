"""Throughput benchmark for the Modal cloud-GPU PII redaction backend.

Measures the model-only OpenMed redaction path (the same code the deployed
`redact_text_batch` function runs) across GPU types and model batch sizes.

Workload: ``--n-chunks`` synthetic agent-trace chunks, each a single chunk of
random length in ``[--min-len, --max-len]`` characters with embedded PII so the
NER model does real work. The same seed -> the same workload on every GPU, so
results are directly comparable.

Run a smoke test on the cheapest GPU first::

    modal run bench_modal.py --gpus L4 --n-chunks 100 --batch-sizes 16

Then the full sweep across all machines / batch sizes::

    modal run bench_modal.py \
        --gpus L4,A10,L40S,A100-40GB,A100-80GB,RTX-PRO-6000,H100 \
        --n-chunks 1000 --batch-sizes 16,32,64
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import modal

APP_NAME = "agent-trace-privacy-scrubber-bench"
MODEL_CACHE_PATH = "/cache"
SRC_PATH = "/root/src"
MODEL_NAME = "OpenMed/privacy-filter-nemotron"

app = modal.App(APP_NAME)
model_cache = modal.Volume.from_name(
    "agent-trace-privacy-scrubber-model-cache", create_if_missing=True
)

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


# --- synthetic workload -----------------------------------------------------

# Deterministic PII fragments; the NER model has to find these spans.
_FIRST = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
          "Irene", "Jack", "Karen", "Leo", "Mona", "Nina", "Oscar", "Paula"]
_LAST = ["Chen", "Smith", "Johnson", "Garcia", "Müller", "Okafor", "Tanaka",
         "Rossi", "Nguyen", "Patel", "Kowalski", "Andersson"]
_CITY = ["Taipei", "Berlin", "Austin", "Lagos", "Osaka", "Milan", "Hanoi",
         "Kraków", "Stockholm", "Toronto", "Lisbon", "Seoul"]
_DOMAIN = ["example.com", "acme.io", "corp.net", "mail.org", "test.dev"]


def _build_chunk(rng: Any, target_len: int) -> str:
    """Assemble a log-like chunk of ~target_len chars with embedded PII."""

    parts: list[str] = []
    length = 0
    i = 0
    while length < target_len:
        first = rng.choice(_FIRST)
        last = rng.choice(_LAST)
        city = rng.choice(_CITY)
        domain = rng.choice(_DOMAIN)
        email = f"{first.lower()}.{last.lower()}@{domain}"
        phone = f"+1-{rng.randint(200, 999)}-{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
        ssn = f"{rng.randint(100, 899)}-{rng.randint(10, 99)}-{rng.randint(1000, 9999)}"
        card = f"{rng.randint(4000, 4999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)}"
        ip = f"{rng.randint(10, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        templates = [
            f"[event] user {first} {last} signed in from {city} (ip {ip}).",
            f"Contact {first} {last} at {email} or {phone} for the {city} account.",
            f"Payment by {first} {last} on card {card}, billing ssn {ssn}.",
            f"Note: {first} {last} ({email}) requested an export of session logs.",
            f"Ticket assigned to {first} {last}; callback number {phone}; office {city}.",
            f"{{\"actor\": \"{first} {last}\", \"email\": \"{email}\", \"src_ip\": \"{ip}\"}}",
        ]
        sentence = templates[i % len(templates)]
        parts.append(sentence)
        length += len(sentence) + 1
        i += 1
    text = " ".join(parts)
    return text[:target_len]


def _build_workload(n_chunks: int, min_len: int, max_len: int, seed: int) -> list[str]:
    import random

    rng = random.Random(seed)
    return [_build_chunk(rng, rng.randint(min_len, max_len)) for _ in range(n_chunks)]


# --- benchmark function -----------------------------------------------------


@app.function(
    image=image,
    gpu="L4",  # overridden per call via .with_options(gpu=...)
    timeout=3600,
    volumes={MODEL_CACHE_PATH: model_cache},
    env={
        "HF_HOME": f"{MODEL_CACHE_PATH}/huggingface",
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_PATH}/huggingface",
    },
)
def benchmark(
    texts: list[str],
    batch_sizes: list[int],
    chunk_size: int,
    seed: int,
) -> dict[str, Any]:
    """Load the model once, then time the redaction pass per batch size."""

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    import torch
    from trace_scrubber.redactors import OpenMedPIIRedactor, RedactionConfig

    gpu_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    total_chars = sum(len(t) for t in texts)
    n = len(texts)

    def make_config(bs: int) -> RedactionConfig:
        return RedactionConfig(
            model_name=MODEL_NAME,
            mode="mask",
            regex_enabled=False,
            model_enabled=True,
            preserve_json_structure=True,
            include_report=True,
            chunk_size=chunk_size,
            model_batch_size=bs,
            confidence_threshold=0.5,
            seed=seed,
        )

    redactor = OpenMedPIIRedactor()

    # Cold model load (from the shared cache volume into GPU memory).
    load_start = time.perf_counter()
    redactor.prepare_model(make_config(batch_sizes[0]))
    model_load_seconds = time.perf_counter() - load_start

    # Warmup so cuDNN autotune / clocks settle before timing.
    warmup_texts = texts[: min(64, n)]
    redactor.redact_many(warmup_texts, make_config(batch_sizes[0]))
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    runs: list[dict[str, Any]] = []
    total_pii = 0
    for bs in batch_sizes:
        config = make_config(bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        results = redactor.redact_many(texts, config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        pii = sum(sum(r.pii_counts.values()) for r in results)
        total_pii = pii
        peak_mem_gb = (
            torch.cuda.max_memory_allocated(0) / 1e9
            if torch.cuda.is_available()
            else 0.0
        )
        runs.append(
            {
                "batch_size": bs,
                "elapsed_s": round(elapsed, 4),
                "chunks_per_s": round(n / elapsed, 2),
                "chars_per_s": round(total_chars / elapsed, 1),
                "peak_gpu_mem_gb": round(peak_mem_gb, 2),
            }
        )

    return {
        "gpu_name": gpu_name,
        "n_chunks": n,
        "total_chars": total_chars,
        "mean_chars": round(total_chars / n, 1),
        "model_load_seconds": round(model_load_seconds, 2),
        "pii_spans_found": total_pii,
        "runs": runs,
    }


@app.local_entrypoint()
def main(
    gpus: str = "L4",
    batch_sizes: str = "16,32,64",
    n_chunks: int = 1000,
    min_len: int = 1000,
    max_len: int = 3000,
    chunk_size: int = 3000,
    seed: int = 2026,
    out: str = "bench_results.json",
) -> None:
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    bs_list = [int(b) for b in batch_sizes.split(",") if b.strip()]

    texts = _build_workload(n_chunks, min_len, max_len, seed)
    actual_chars = sum(len(t) for t in texts)
    print(
        f"workload: {len(texts)} chunks, {actual_chars:,} chars "
        f"(mean {actual_chars / len(texts):.0f}, range {min_len}-{max_len}), "
        f"batch_sizes={bs_list}, gpus={gpu_list}",
        flush=True,
    )

    # Spawn every GPU concurrently; cost is GPU-seconds either way, but this
    # collapses wall-clock.
    handles = {}
    for g in gpu_list:
        handles[g] = benchmark.with_options(gpu=g).spawn(
            texts, bs_list, chunk_size, seed
        )

    collected: dict[str, Any] = {}
    for g, handle in handles.items():
        try:
            res = handle.get()
            collected[g] = res
            print(f"\n=== {g} (device reported: {res['gpu_name']}) ===", flush=True)
            print(
                f"  model load: {res['model_load_seconds']}s | "
                f"pii spans: {res['pii_spans_found']} | mean chars: {res['mean_chars']}",
                flush=True,
            )
            for run in res["runs"]:
                print(
                    f"  bs={run['batch_size']:>3}  "
                    f"{run['chunks_per_s']:>8.2f} chunks/s  "
                    f"{run['chars_per_s']:>12,.0f} chars/s  "
                    f"{run['elapsed_s']:>7.2f}s  "
                    f"peak {run['peak_gpu_mem_gb']}GB",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - report per-GPU failures
            collected[g] = {"error": repr(exc)}
            print(f"\n=== {g} FAILED: {exc!r} ===", flush=True)

    payload = {
        "config": {
            "n_chunks": n_chunks,
            "min_len": min_len,
            "max_len": max_len,
            "chunk_size": chunk_size,
            "seed": seed,
            "batch_sizes": bs_list,
            "total_chars": actual_chars,
        },
        "results": collected,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out}", flush=True)
