#!/usr/bin/env python3
"""Export heuristic realistic floor estimates per SOL-gap candidate.

The estimates are intentionally tagged with confidence/assumptions. They are
triage floors, not measured replacement for prototype kernels.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = REPO_ROOT / "investigation_results" / "sol_gap_candidates.csv"
DEFAULT_INTERLEAVED = REPO_ROOT / "investigation_results" / "interleaved_3config_results.csv"
DEFAULT_OUTPUT = REPO_ROOT / "investigation_results" / "per_repro_realistic_floors.csv"
DEFAULT_SUMMARY = REPO_ROOT / "investigation_results" / "realistic_floor_summary.csv"


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        return float(value) if value != "" else default
    except ValueError:
        return default


def load_interleaved_best(path: Path) -> dict[str, dict[str, float | str]]:
    if not path.exists():
        return {}
    best: dict[str, dict[str, float | str]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rid = row["repro_id"]
            if rid not in best or row["config_label"] == row.get("best_3config_label"):
                best[rid] = {
                    "best_3config_label": row.get("best_3config_label", ""),
                    "best_3config_us": f(row, "best_3config_us"),
                    "best_3config_over_sol": f(row, "best_3config_over_sol"),
                }
    return best


def has(row: dict[str, str], needle: str) -> bool:
    return needle in (row.get("top_ops", "") + ";" + row.get("ops_sample", ""))


def estimate_floor(row: dict[str, str]) -> dict[str, str | float]:
    rid = row["repro_id"]
    prefix = row["pattern_prefix"]
    n_kernels = max(1, int(f(row, "n_kernels", 1)))
    sol = f(row, "memcopy_sol_us")
    best = f(row, "best_compile_us")
    launch3 = f(row, "launch_adjusted_sol_us_3p0")
    lag3 = f(row, "launch_adjusted_gap_3p0")
    excess3 = f(row, "excess_us_vs_launch_adjusted_sol_3p0")

    floor_kind = "unknown"
    confidence = "low"
    multiplier = 1.0
    additive_us = 3.0 * n_kernels
    rationale = "fallback launch-adjusted floor"
    assumptions = "heuristic; not prototype-measured"

    if rid == "sum_adeaebad93f7" or (has(row, "aten.gather.default") and has(row, "aten.sort")):
        floor_kind = "irregular_gather_reduce_heuristic"
        confidence = "low"
        multiplier = 2.5
        additive_us = 6.0
        rationale = "irregular gather/sort/cat/reduce; memcopy SOL underestimates source and intermediate traffic"
        assumptions = "assumes fused gather-reduce still performs irregular loads and tiny RNG/sort remains separate"
    elif prefix == "pointwise" and (has(row, "aten.slice_scatter.default") or has(row, "aten.select_scatter.default")):
        floor_kind = "layout_stencil_functional_update_heuristic"
        confidence = "low"
        multiplier = 2.2 if n_kernels >= 8 else 1.8
        additive_us = 4.0 * min(n_kernels, 12)
        rationale = "slice/select scatter reconstruction needs canonicalized stencil/layout floor"
        assumptions = "assumes explicit-offset interior kernel plus separate boundary work"
    elif prefix == "pointwise" and n_kernels >= 8 and lag3 <= 1.25:
        floor_kind = "launch_dominated_adjusted_floor"
        confidence = "high"
        multiplier = 1.0
        additive_us = 3.0 * n_kernels
        rationale = "raw gap mostly explained by launch count"
        assumptions = "3us launch floor sufficient for deprioritization"
    elif prefix == "pointwise":
        floor_kind = "pointwise_expression_heuristic"
        confidence = "medium"
        multiplier = 1.35 if has(row, "f64") or "f64" in row.get("ops_sample", "") else 1.15
        additive_us = 3.0 * n_kernels
        rationale = "pointwise expression has math/register/index overhead beyond memcopy"
        assumptions = "assumes CSE/tiling can reduce but not eliminate expression overhead"
    elif prefix.startswith("amax_sum") or prefix in {"any_amax_sum", "argmax_mean", "amax_argmax_sum", "argmax_amax_sum"}:
        if has(row, "aten.exp.default") or has(row, "aten.log.default") or has(row, "aten.gather.default") or has(row, "aten.topk"):
            floor_kind = "online_softmax_ce_heuristic"
            confidence = "medium"
            # Large vocab/row online softmax has measured heuristic data around 1.15-1.35x memcopy-ish SOL;
            # smaller masked/windowed cases need more layout/mask overhead.
            multiplier = 1.45 if sol >= 80 else 1.9
            additive_us = 4.0 * n_kernels
            rationale = "softmax/CE needs exp/log/reduction or gather; online algorithm floor is above memcopy"
            assumptions = "uses online-softmax heuristic class; not per-shape prototype except known data"
        else:
            floor_kind = "generic_amax_reduction_heuristic"
            confidence = "medium"
            multiplier = 1.6
            additive_us = 3.0 * n_kernels
            rationale = "amax/argmax reduction has tree and predicate overhead beyond memcopy"
            assumptions = "assumes tuned persistent/tiled reduction"
    elif prefix.startswith("var_mean") or prefix.startswith("mean") or prefix in {"sum_mean", "mean_var", "mean_var_mean", "any_mean"}:
        floor_kind = "norm_template_heuristic"
        confidence = "low_to_medium"
        multiplier = 1.8 if prefix.startswith("var_mean") or "var" in prefix else 1.45
        additive_us = 4.0 * n_kernels
        rationale = "norm/stat reductions require Welford/sum math, affine epilogues, and side outputs"
        assumptions = "assumes semantic BN/LN/RMSNorm template, not generic reduction"
    elif prefix.startswith("sum"):
        if has(row, "aten.scatter_add") or has(row, "aten.index_put.default") or has(row, "_low_memory_max_pool_offsets"):
            floor_kind = "structured_scatter_reduce_heuristic"
            confidence = "low"
            multiplier = 2.0
            additive_us = 5.0 * min(n_kernels, 12)
            rationale = "structured scatter/pool/upsample backward floor needs direct-reduce template"
            assumptions = "assumes dense scatter materialization removed but irregular/atomic/index overhead remains"
        else:
            floor_kind = "multi_output_reduction_heuristic"
            confidence = "medium"
            multiplier = 1.55 if prefix in {"sum_sum", "sum_sum_sum", "sum_sum_mean"} else 1.35
            additive_us = 3.0 * n_kernels
            rationale = "reduction tree and multi-output accumulator overhead beyond memcopy"
            assumptions = "assumes tuned one-pass or split compatible reductions"
    else:
        floor_kind = "rare_unknown_launch_adjusted"
        confidence = "low"
        multiplier = 1.5
        additive_us = 3.0 * n_kernels
        rationale = "rare prefix not explicitly modeled"
        assumptions = "manual review needed if prioritized"

    estimated = max(sol * multiplier + additive_us, sol + 1.0 * n_kernels)
    estimated = min(estimated, best) if best == best else estimated
    speedup_to_floor = best / estimated if estimated and best == best else math.nan
    excess_to_floor = best - estimated if best == best else math.nan
    floor_over_sol = estimated / sol if sol else math.nan
    return {
        "floor_kind": floor_kind,
        "floor_confidence": confidence,
        "estimated_realistic_floor_us": estimated,
        "estimated_floor_over_sol": floor_over_sol,
        "potential_speedup_to_floor": speedup_to_floor,
        "excess_us_to_floor": excess_to_floor,
        "launch_adjusted_gap_3us": lag3,
        "launch_adjusted_excess_us_3us": excess3,
        "floor_rationale": rationale,
        "floor_assumptions": assumptions,
    }


def write_summary(path: Path, rows: list[dict[str, str | float]]) -> None:
    groups: dict[str, list[dict[str, str | float]]] = {}
    for row in rows:
        groups.setdefault(str(row["floor_kind"]), []).append(row)
    summary = []
    for kind, items in groups.items():
        count = len(items)
        best_sum = sum(float(item["best_compile_us"]) for item in items)
        floor_sum = sum(float(item["estimated_realistic_floor_us"]) for item in items)
        excess_sum = sum(float(item["excess_us_to_floor"]) for item in items)
        speedup = best_sum / floor_sum if floor_sum else math.nan
        examples = ";".join(str(item["repro_id"]) for item in sorted(items, key=lambda x: float(x["excess_us_to_floor"]), reverse=True)[:8])
        confidence_counts: dict[str, int] = {}
        for item in items:
            confidence_counts[str(item["floor_confidence"])] = confidence_counts.get(str(item["floor_confidence"]), 0) + 1
        summary.append(
            {
                "floor_kind": kind,
                "count": count,
                "aggregate_best_compile_us": best_sum,
                "aggregate_estimated_floor_us": floor_sum,
                "aggregate_excess_us_to_floor": excess_sum,
                "aggregate_potential_speedup": speedup,
                "confidence_counts": ";".join(f"{k}:{v}" for k, v in sorted(confidence_counts.items())),
                "top_excess_examples": examples,
            }
        )
    summary.sort(key=lambda row: row["aggregate_excess_us_to_floor"], reverse=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--interleaved", type=Path, default=DEFAULT_INTERLEAVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    interleaved = load_interleaved_best(args.interleaved)
    output_rows = []
    with args.candidates.open() as handle:
        for row in csv.DictReader(handle):
            estimate = estimate_floor(row)
            best3 = interleaved.get(row["repro_id"], {})
            out = dict(row)
            out.update(best3)
            out.update(estimate)
            best3_us = out.get("best_3config_us")
            floor_us = estimate["estimated_realistic_floor_us"]
            try:
                best3_float = float(best3_us) if best3_us not in (None, "") else math.nan
            except (TypeError, ValueError):
                best3_float = math.nan
            out["best_3config_speedup_to_floor"] = (best3_float / floor_us) if best3_float == best3_float and floor_us else math.nan
            output_rows.append(out)
    output_rows.sort(key=lambda row: float(row["excess_us_to_floor"]), reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    write_summary(args.summary, output_rows)
    print(f"wrote {args.output} ({len(output_rows)} rows)")
    print(f"wrote {args.summary}")


if __name__ == "__main__":
    main()
