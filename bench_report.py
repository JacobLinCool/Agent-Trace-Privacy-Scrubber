"""Render the Modal GPU benchmark results into throughput + cost tables.

Usage: python bench_report.py [bench_results.json]
"""

from __future__ import annotations

import json
import sys

# Modal per-second on-demand prices (USD/sec) as given on the pricing page.
PRICE_PER_SEC = {
    "H100": 0.001097,
    "RTX-PRO-6000": 0.000842,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
}

# Display order: cheapest -> most expensive.
ORDER = ["L4", "A10", "L40S", "A100-40GB", "A100-80GB", "RTX-PRO-6000", "H100"]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "bench_results.json"
    with open(path) as fh:
        data = json.load(fh)

    results = data["results"]
    cfg = data["config"]
    batch_sizes = cfg["batch_sizes"]

    print(
        f"# Workload: {cfg['n_chunks']} chunks, lengths {cfg['min_len']}-{cfg['max_len']} "
        f"chars ({cfg['total_chars']:,} total), seed {cfg['seed']}\n"
    )

    # --- throughput table (chunks/s) ---
    head = ["GPU", "$/sec", "load(s)"] + [f"bs={b} chunks/s" for b in batch_sizes]
    print("## Throughput (chunks/sec)\n")
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")
    bench: dict[str, dict[int, float]] = {}
    for g in ORDER:
        res = results.get(g)
        if not res or "error" in (res or {}):
            err = (res or {}).get("error", "missing")
            print(f"| {g} | {PRICE_PER_SEC[g]:.6f} | - | " + " | ".join(["FAIL"] * len(batch_sizes)) + f" | <{err}>")
            continue
        runs = {r["batch_size"]: r for r in res["runs"]}
        bench[g] = {b: runs[b]["chunks_per_s"] for b in batch_sizes}
        row = [g, f"{PRICE_PER_SEC[g]:.6f}", f"{res['model_load_seconds']}"]
        row += [f"{runs[b]['chunks_per_s']:.1f}" for b in batch_sizes]
        print("| " + " | ".join(row) + " |")

    # --- cost-efficiency at best batch size ---
    print("\n## Cost efficiency (using each GPU's best batch size)\n")
    print("| GPU | best bs | chunks/s | $/1000 chunks | chunks per $ | rel. throughput | rel. $/chunk |")
    print("|---|---|---|---|---|---|---|")
    best: dict[str, tuple[int, float, float]] = {}
    for g in ORDER:
        if g not in bench:
            continue
        bs, cps = max(bench[g].items(), key=lambda kv: kv[1])
        price = PRICE_PER_SEC[g]
        cost_per_1k = price / cps * 1000
        best[g] = (bs, cps, cost_per_1k)
    if best:
        base_cps = best["L4"][1] if "L4" in best else min(b[1] for b in best.values())
        base_cost = best["L4"][2] if "L4" in best else min(b[2] for b in best.values())
        for g in ORDER:
            if g not in best:
                continue
            bs, cps, cost_per_1k = best[g]
            price = PRICE_PER_SEC[g]
            print(
                f"| {g} | {bs} | {cps:.1f} | ${cost_per_1k:.5f} | "
                f"{cps / price:,.0f} | {cps / base_cps:.2f}x | {cost_per_1k / base_cost:.2f}x |"
            )


if __name__ == "__main__":
    main()
