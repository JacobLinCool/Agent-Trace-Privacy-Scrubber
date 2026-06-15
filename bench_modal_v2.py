"""Hardware benchmark for the MODIFIED redaction pipeline.

Per GPU (one container each, model loaded once):

  Part A  canonical pure-GPU throughput on 1000 unique chunks (1000-3000 chars)
          at model batch 16/32/64 -- a no-regression check vs the pre-change run.
  Part B  the modified path (CachingPIIRedactor) vs the old path (plain) on REAL
          Codex string leaves, to show what whole-string dedup buys per GPU.

The cache + network-batch decoupling live on the client/network layer, so they
cannot change Part A; Part B isolates the cache's server-side (per-item) effect.

    modal run bench_modal_v2.py \
        --gpus L4,A10,L40S,A100-40GB,A100-80GB,RTX-PRO-6000,H100
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = "agent-trace-privacy-scrubber-bench-v2"
MODEL_CACHE_PATH = "/cache"
SRC_PATH = "/root/src"
MODEL_NAME = "OpenMed/privacy-filter-nemotron"
REAL_LOG = Path.home() / (
    ".codex/sessions/2026/06/15/"
    "rollout-2026-06-15T03-47-32-019ec7ac-d30d-7471-9835-92e1dc534172.jsonl"
)

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

_FIRST = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
          "Irene", "Jack", "Karen", "Leo", "Mona", "Nina", "Oscar", "Paula"]
_LAST = ["Chen", "Smith", "Johnson", "Garcia", "Müller", "Okafor", "Tanaka",
         "Rossi", "Nguyen", "Patel", "Kowalski", "Andersson"]
_CITY = ["Taipei", "Berlin", "Austin", "Lagos", "Osaka", "Milan", "Hanoi",
         "Kraków", "Stockholm", "Toronto", "Lisbon", "Seoul"]
_DOMAIN = ["example.com", "acme.io", "corp.net", "mail.org", "test.dev"]


def _build_chunk(rng: Any, target_len: int) -> str:
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
        ip = f"{rng.randint(10, 220)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"
        templates = [
            f"[event] user {first} {last} signed in from {city} (ip {ip}).",
            f"Contact {first} {last} at {email} or {phone} for the {city} account.",
            f"Note: {first} {last} ({email}) requested an export of session logs.",
            f"{{\"actor\": \"{first} {last}\", \"email\": \"{email}\", \"src_ip\": \"{ip}\"}}",
        ]
        sentence = templates[i % len(templates)]
        parts.append(sentence)
        length += len(sentence) + 1
        i += 1
    return " ".join(parts)[:target_len]


def _build_synthetic(n: int, seed: int) -> list[str]:
    import random

    rng = random.Random(seed)
    return [_build_chunk(rng, rng.randint(1000, 3000)) for _ in range(n)]


def _extract_real_leaves(n_lines: int, trunc: int = 3000) -> list[str]:
    import json as _json

    def walk(v: Any, out: list[str]) -> None:
        if isinstance(v, str):
            out.append(v[:trunc])
        elif isinstance(v, list):
            for x in v:
                walk(x, out)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x, out)

    leaves: list[str] = []
    with REAL_LOG.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except Exception:
                leaves.append(line[:trunc])
                continue
            walk(obj, leaves)
    return leaves


@app.function(
    image=image,
    gpu="L4",
    timeout=3600,
    volumes={MODEL_CACHE_PATH: model_cache},
    env={
        "HF_HOME": f"{MODEL_CACHE_PATH}/huggingface",
        "TRANSFORMERS_CACHE": f"{MODEL_CACHE_PATH}/huggingface",
    },
)
def benchmark_v2(
    synthetic: list[str],
    real_leaves: list[str],
    batch_sizes: list[int],
    seed: int,
) -> dict[str, Any]:
    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    import torch
    from trace_scrubber.redactors import (
        CachingPIIRedactor,
        OpenMedPIIRedactor,
        RedactionConfig,
    )

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    def cfg(bs: int) -> RedactionConfig:
        return RedactionConfig(
            model_name=MODEL_NAME, mode="mask", regex_enabled=False,
            model_enabled=True, chunk_size=3000, model_batch_size=bs,
            confidence_threshold=0.5, seed=seed,
        )

    def sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def free() -> None:
        # Release the caching allocator's retained blocks so an earlier large
        # batch can't starve a later run on a small (24GB) card.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    base = OpenMedPIIRedactor()
    load0 = time.perf_counter()
    base.prepare_model(cfg(batch_sizes[0]))
    model_load = time.perf_counter() - load0
    base.redact_many(synthetic[:64], cfg(batch_sizes[0]))  # warmup
    free()

    # Part A: canonical pure-GPU throughput on unique chunks.
    part_a = []
    for bs in batch_sizes:
        sync()
        t = time.perf_counter()
        base.redact_many(synthetic, cfg(bs))
        sync()
        dt = time.perf_counter() - t
        part_a.append({"batch_size": bs, "chunks_per_s": round(len(synthetic) / dt, 2)})
        free()

    # Part B: modified (cache) vs old (plain) on real leaves. Real agent-trace
    # content is token-dense, so a large batch can exceed a 24GB card's memory
    # (eager O(seq^2) attention) -- fall back to bs=16 if it OOMs.
    bs_mid = batch_sizes[len(batch_sizes) // 2]
    total = len(real_leaves)

    def timed(redactor: Any, bs: int) -> float:
        free()
        start = time.perf_counter()
        redactor.redact_many(real_leaves, cfg(bs))
        sync()
        return time.perf_counter() - start

    part_b_bs = bs_mid
    try:
        old_dt = timed(base, bs_mid)
    except torch.cuda.OutOfMemoryError:
        free()
        part_b_bs = 16
        old_dt = timed(base, part_b_bs)

    cached = CachingPIIRedactor(base)
    new_dt = timed(cached, part_b_bs)
    unique = cached.cache_size

    return {
        "gpu_name": gpu_name,
        "model_load_s": round(model_load, 2),
        "part_a": part_a,
        "part_b": {
            "batch_size": part_b_bs,
            "total_leaves": total,
            "unique_leaves": unique,
            "old_strings_per_s": round(total / old_dt, 1),
            "new_strings_per_s": round(total / new_dt, 1),
            "old_s": round(old_dt, 2),
            "new_s": round(new_dt, 2),
            "cache_speedup": round(old_dt / new_dt, 2),
        },
    }


@app.local_entrypoint()
def main(
    gpus: str = "L4",
    batch_sizes: str = "16,32,64",
    n_synthetic: int = 1000,
    real_lines: int = 200,
    seed: int = 2026,
    out: str = "bench_v2_results.json",
) -> None:
    gpu_list = [g.strip() for g in gpus.split(",") if g.strip()]
    bs_list = [int(b) for b in batch_sizes.split(",") if b.strip()]
    synthetic = _build_synthetic(n_synthetic, seed)
    real_leaves = _extract_real_leaves(real_lines)
    uniq = len(set(real_leaves))
    print(
        f"synthetic: {len(synthetic)} unique chunks | "
        f"real leaves: {len(real_leaves)} ({uniq} unique, "
        f"{100 * (1 - uniq / len(real_leaves)):.0f}% dup) from {real_lines} lines",
        flush=True,
    )

    handles = {
        g: benchmark_v2.with_options(gpu=g).spawn(synthetic, real_leaves, bs_list, seed)
        for g in gpu_list
    }
    collected: dict[str, Any] = {}
    for g, h in handles.items():
        try:
            res = h.get()
            collected[g] = res
            a = "  ".join(
                f"bs{r['batch_size']}={r['chunks_per_s']:.1f}" for r in res["part_a"]
            )
            b = res["part_b"]
            print(f"\n=== {g} ({res['gpu_name']}) load {res['model_load_s']}s ===", flush=True)
            print(f"  Part A pure-GPU chunks/s:  {a}", flush=True)
            print(
                f"  Part B real leaves bs{b['batch_size']}: "
                f"old {b['old_strings_per_s']}/s -> new {b['new_strings_per_s']}/s "
                f"({b['cache_speedup']}x via dedup, {b['unique_leaves']}/{b['total_leaves']} unique)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            collected[g] = {"error": repr(exc)}
            print(f"\n=== {g} FAILED: {exc!r} ===", flush=True)

    with open(out, "w") as fh:
        json.dump(
            {
                "config": {
                    "n_synthetic": n_synthetic,
                    "real_lines": real_lines,
                    "real_unique": uniq,
                    "batch_sizes": bs_list,
                    "seed": seed,
                },
                "results": collected,
            },
            fh,
            indent=2,
        )
    print(f"\nwrote {out}", flush=True)
