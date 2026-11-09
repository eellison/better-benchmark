#!/usr/bin/env python3
"""Export benchmark result tables for SOL-gap investigations.

Reads the current sweep JSON artifacts and writes CSV/SQLite tables that can be
refreshed while long-running sweeps continue to populate. The legacy
``sol_gap_candidates`` output retains every raw logical-I/O gap for diagnostics;
``single_kernel_roofline_candidates`` is the queue where the copy proxy is a
direct per-kernel comparison, while multi-kernel rows are ranked separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "sweep_pr184905_baseline.json"
DEFAULT_INTERLEAVED = REPO_ROOT / "sweep_3config_interleaved.json"
DEFAULT_OUTDIR = REPO_ROOT / "investigation_results"
LAUNCH_OVERHEAD_US_VALUES = (1.0, 3.0, 5.0)
# These thresholds only route a region to an explicit config sweep. They do not
# predict that combo kernels will win or enable them automatically.
MEGA_REGION_MIN_KERNELS = 16
MEGA_REGION_MIN_OPS = 128


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def repro_id(repro_path: str) -> str:
    return Path(repro_path).parent.name


def normalize_repro_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        try:
            return str(path_obj.relative_to(REPO_ROOT))
        except ValueError:
            return str(path_obj)
    return str(path_obj)


def parse_pattern_prefix(name: str) -> str:
    return re.sub(r"_[0-9a-f]{12}$", "", name)


def read_meta(repro_path: str) -> dict[str, Any]:
    meta_path = (REPO_ROOT / normalize_repro_path(repro_path)).with_name("meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def meta_summary(meta: dict[str, Any]) -> tuple[str, str, int | str, str, int | None]:
    ops = meta.get("origin_ops") or meta.get("ops") or meta.get("aten_ops") or []
    if not isinstance(ops, list):
        ops = []
    op_counts: dict[str, int] = {}
    for op in ops:
        op_counts[str(op)] = op_counts.get(str(op), 0) + 1
    top_ops = ";".join(f"{op}:{count}" for op, count in sorted(op_counts.items(), key=lambda item: (-item[1], item[0]))[:12])
    models = meta.get("models") or []
    if not isinstance(models, list):
        models = []
    n_ops = meta.get("n_ops")
    if not isinstance(n_ops, int) or isinstance(n_ops, bool) or n_ops < 0:
        n_ops = None
    return (
        ";".join(str(op) for op in ops[:40]),
        top_ops,
        meta.get("n_models", len(models)),
        ";".join(str(model) for model in models[:8]),
        n_ops,
    )


def normalize_kernel_count(value: Any) -> int | None:
    """Return a non-negative generated-kernel count, or None when unknown."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def multi_kernel_enabled(results: dict[str, Any]) -> bool:
    """Whether a bench_parallel run compiled with triton.multi_kernel, which
    generates several variants per kernel but launches one. bench_parallel
    records --inductor-config settings in the file-level _metadata."""
    metadata = results.get("_metadata")
    config = metadata.get("inductor_config") if isinstance(metadata, dict) else None
    return isinstance(config, dict) and bool(config.get("triton.multi_kernel"))


def classify_sol_comparison(
    n_kernels: int | None, n_ops: int | None, *, multi_kernel: bool = False
) -> str:
    """Classify how a logical-byte copy proxy may be interpreted.

    ``n_kernels`` counts generated kernels, not launches. A copy of the
    region's external input/output bytes is a useful direct roofline only when
    the compiled region is one kernel. Multi-kernel rows retain that ratio as a
    logical-I/O diagnostic, but must be ranked in a separate codegen/launch
    queue. Large launch forests get an explicit class so they can trigger
    combo-kernel or partition-boundary sweeps rather than be mistaken for one
    pathological kernel. Rows whose count is missing, or whose generated count
    overstates launches (multi_kernel variants), join neither queue.
    """
    if n_kernels is None:
        return "unknown_kernel_count"
    if n_kernels == 0:
        return "no_generated_kernel"
    if n_kernels == 1:
        return "single_kernel_copy_roofline"
    if multi_kernel:
        return "unknown_launch_count"
    if (
        n_kernels >= MEGA_REGION_MIN_KERNELS
        and n_ops is not None
        and n_ops >= MEGA_REGION_MIN_OPS
    ):
        return "multi_kernel_mega_region"
    return "multi_kernel_logical_io"


def baseline_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    multi_kernel = multi_kernel_enabled(results)
    for repro_path, cases in results.items():
        if repro_path.startswith("_") or not isinstance(cases, dict):
            continue
        normalized = normalize_repro_path(repro_path)
        rid = repro_id(normalized)
        prefix = parse_pattern_prefix(rid)
        meta = read_meta(normalized)
        ops_sample, top_ops, n_models, models, n_ops = meta_summary(meta)
        for case_name, record in cases.items():
            if not isinstance(record, dict) or "memcopy_sol_us" not in record:
                continue
            compiled_us = record.get("compiled_us")
            coord_us = record.get("coord_descent_us")
            sol_us = record.get("memcopy_sol_us")
            best_label = "compiled"
            best_us = compiled_us
            if coord_us is not None and (best_us is None or coord_us < best_us):
                best_label = "coord_descent"
                best_us = coord_us
            best_gap = best_us / sol_us if best_us and sol_us else math.nan
            n_kernels = normalize_kernel_count(record.get("n_kernels"))
            comparison_class = classify_sol_comparison(
                n_kernels, n_ops, multi_kernel=multi_kernel
            )
            mitigation_factors = record.get("sol_mitigation_factors") or []
            if not isinstance(mitigation_factors, list):
                mitigation_factors = []
            launch_adjusted = {}
            for launch_overhead_us in LAUNCH_OVERHEAD_US_VALUES:
                adjusted_sol_us = (
                    sol_us + launch_overhead_us * n_kernels
                    if sol_us is not None and n_kernels is not None
                    else None
                )
                suffix = str(launch_overhead_us).replace(".", "p")
                launch_adjusted[f"launch_adjusted_sol_us_{suffix}"] = adjusted_sol_us
                launch_adjusted[f"launch_adjusted_gap_{suffix}"] = (best_us / adjusted_sol_us) if best_us and adjusted_sol_us else math.nan
                launch_adjusted[f"excess_us_vs_launch_adjusted_sol_{suffix}"] = (best_us - adjusted_sol_us) if best_us is not None and adjusted_sol_us is not None else math.nan
            rows.append(
                {
                    "repro_id": rid,
                    "repro_path": normalized,
                    "case": case_name,
                    "pattern_prefix": prefix,
                    "compiled_us": compiled_us,
                    "coord_descent_us": coord_us,
                    "best_compile_label": best_label,
                    "best_compile_us": best_us,
                    "memcopy_sol_us": sol_us,
                    "best_compile_over_sol": best_gap,
                    "gap_default": record.get("gap_default"),
                    "gap_cd": record.get("gap_cd"),
                    "total_bytes": record.get("total_bytes"),
                    "n_kernels": n_kernels,
                    "n_ops": n_ops,
                    "sol_comparison_class": comparison_class,
                    "single_copy_roofline_applicable": (
                        comparison_class == "single_kernel_copy_roofline"
                    ),
                    "combo_sweep_recommended": (
                        comparison_class == "multi_kernel_mega_region"
                    ),
                    "sol_mitigation_factors": ";".join(
                        str(factor) for factor in mitigation_factors
                    ),
                    **launch_adjusted,
                    "n_models": n_models,
                    "models": models,
                    "ops_sample": ops_sample,
                    "top_ops": top_ops,
                }
            )
    rows.sort(key=lambda row: (row["best_compile_over_sol"] if row["best_compile_over_sol"] == row["best_compile_over_sol"] else -1), reverse=True)
    for idx, row in enumerate(rows, 1):
        row["sol_gap_rank"] = idx
    return rows


def split_sol_gap_candidates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split raw SOL gaps into directly comparable and multi-kernel queues."""
    raw = [row for row in rows if row["best_compile_over_sol"] > 1.1]
    single_kernel = [
        row for row in raw if row["single_copy_roofline_applicable"]
    ]
    multi_kernel = [
        dict(row)
        for row in raw
        if row["sol_comparison_class"].startswith("multi_kernel_")
    ]
    multi_kernel.sort(
        key=lambda row: (
            row["sol_comparison_class"] == "multi_kernel_mega_region",
            row.get("best_compile_us") or 0,
            row.get("n_kernels") or 0,
            row.get("n_ops") or 0,
        ),
        reverse=True,
    )
    for rank, row in enumerate(multi_kernel, 1):
        row["multi_kernel_priority_rank"] = rank
    return raw, single_kernel, multi_kernel


def interleaved_rows(results: dict[str, Any], baseline_by_repro: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repro_path, record in results.items():
        if repro_path.startswith("_") or not isinstance(record, dict):
            continue
        normalized = normalize_repro_path(repro_path)
        rid = repro_id(normalized)
        base = baseline_by_repro.get(normalized, {})
        sol_us = base.get("memcopy_sol_us")
        best_compile_us = base.get("best_compile_us")
        configs = record.get("configs", {})
        if not isinstance(configs, dict):
            continue
        valid_configs = [(label, cfg.get("us"), cfg.get("all_rounds")) for label, cfg in configs.items() if isinstance(cfg, dict) and cfg.get("us") is not None]
        if not valid_configs:
            continue
        best_label, best_us, _ = min(valid_configs, key=lambda item: item[1])
        for label, us, rounds in valid_configs:
            rows.append(
                {
                    "repro_id": rid,
                    "repro_path": normalized,
                    "config_label": label,
                    "config_us": us,
                    "rounds": json.dumps(rounds or []),
                    "speedup_vs_default": (configs.get("default", {}).get("us") / us) if configs.get("default", {}).get("us") and us else None,
                    "speedup_vs_best_compile_baseline": (best_compile_us / us) if best_compile_us and us else None,
                    "config_over_sol": (us / sol_us) if sol_us and us else None,
                    "best_3config_label": best_label,
                    "best_3config_us": best_us,
                    "best_3config_over_sol": (best_us / sol_us) if sol_us and best_us else None,
                    "round_count": record.get("rounds"),
                    "rep_per_round": record.get("rep_per_round"),
                    "total_bytes": record.get("total_bytes", base.get("total_bytes")),
                }
            )
    rows.sort(key=lambda row: (row["repro_id"], row["config_label"]))
    return rows


def grouped_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["best_compile_over_sol"] > 1.1:
            groups.setdefault(row["pattern_prefix"], []).append(row)
    output = []
    for prefix, items in groups.items():
        gaps = [item["best_compile_over_sol"] for item in items]
        output.append(
            {
                "pattern_prefix": prefix,
                "count": len(items),
                "max_gap": max(gaps),
                "geomean_gap": math.exp(sum(math.log(gap) for gap in gaps) / len(gaps)),
                "median_gap": sorted(gaps)[len(gaps) // 2],
                "representative_repros": ";".join(item["repro_id"] for item in sorted(items, key=lambda row: row["best_compile_over_sol"], reverse=True)[:12]),
                "top_ops": items[0].get("top_ops", ""),
                "models": items[0].get("models", ""),
            }
        )
    output.sort(key=lambda row: (row["max_gap"], row["count"]), reverse=True)
    return output


def launch_adjusted_priority_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        launch_gap = row.get("launch_adjusted_gap_3p0")
        excess_us = row.get("excess_us_vs_launch_adjusted_sol_3p0")
        if not launch_gap or launch_gap != launch_gap or excess_us is None or excess_us != excess_us:
            continue
        priority_score = max(0.0, excess_us) * math.log2(max(1.0, row.get("best_compile_over_sol", 1.0)))
        output.append(
            {
                "repro_id": row["repro_id"],
                "repro_path": row["repro_path"],
                "pattern_prefix": row["pattern_prefix"],
                "raw_sol_gap": row["best_compile_over_sol"],
                "launch_adjusted_gap_1us": row.get("launch_adjusted_gap_1p0"),
                "launch_adjusted_gap_3us": launch_gap,
                "launch_adjusted_gap_5us": row.get("launch_adjusted_gap_5p0"),
                "best_compile_us": row["best_compile_us"],
                "memcopy_sol_us": row["memcopy_sol_us"],
                "n_kernels": row["n_kernels"],
                "excess_us_vs_launch_adjusted_sol_3us": excess_us,
                "priority_score_3us": priority_score,
                "models": row.get("models", ""),
                "top_ops": row.get("top_ops", ""),
            }
        )
    output.sort(key=lambda row: (row["priority_score_3us"], row["excess_us_vs_launch_adjusted_sol_3us"]), reverse=True)
    for idx, row in enumerate(output, 1):
        row["launch_adjusted_priority_rank_3us"] = idx
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_sqlite(path: Path, tables: dict[str, list[dict[str, Any]]]) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    try:
        for table, rows in tables.items():
            if not rows:
                continue
            columns = list(rows[0])
            col_defs = ", ".join(f'"{column}"' for column in columns)
            conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f'INSERT INTO "{table}" ({col_defs}) VALUES ({placeholders})',
                [[row.get(column) for column in columns] for row in rows],
            )
        conn.execute('CREATE INDEX IF NOT EXISTS idx_baseline_gap ON baseline_results(best_compile_over_sol)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_baseline_repro ON baseline_results(repro_id)')
        conn.commit()
    finally:
        conn.close()


def write_summary(path: Path, baseline: list[dict[str, Any]], interleaved: list[dict[str, Any]], groups: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    gt11 = [row for row in baseline if row["best_compile_over_sol"] > 1.1]
    gt2 = [row for row in baseline if row["best_compile_over_sol"] > 2.0]
    gt5 = [row for row in baseline if row["best_compile_over_sol"] > 5.0]
    launch_gt11 = [row for row in baseline if row.get("launch_adjusted_gap_3p0", math.nan) > 1.1]
    launch_gt2 = [row for row in baseline if row.get("launch_adjusted_gap_3p0", math.nan) > 2.0]
    _, single_kernel_candidates, multi_kernel_candidates = split_sol_gap_candidates(
        baseline
    )
    mega_regions = [
        row
        for row in multi_kernel_candidates
        if row["sol_comparison_class"] == "multi_kernel_mega_region"
    ]
    unknown_count = [
        row for row in baseline if row["sol_comparison_class"].startswith("unknown_")
    ]
    covered_repros = {row["repro_id"] for row in interleaved}
    best_rows = [row for row in interleaved if row["config_label"] == row["best_3config_label"]]
    config_wins: dict[str, int] = {}
    for row in best_rows:
        config_wins[row["best_3config_label"]] = config_wins.get(row["best_3config_label"], 0) + 1
    lines = [
        "# SOL Gap Investigation Summary",
        "",
        f"Baseline source: `{metadata.get('baseline_path')}`",
        f"Interleaved source: `{metadata.get('interleaved_path')}`",
        f"Baseline rows: {len(baseline)}",
        f"Rows with best compile / SOL > 1.1x: {len(gt11)}",
        f"Rows with best compile / SOL > 2.0x: {len(gt2)}",
        f"Rows with best compile / SOL > 5.0x: {len(gt5)}",
        f"Single-kernel copy-roofline candidates > 1.1x: {len(single_kernel_candidates)}",
        f"Multi-kernel logical-I/O gaps > 1.1x: {len(multi_kernel_candidates)}",
        f"Multi-kernel mega-regions requiring a combo/partition sweep: {len(mega_regions)}",
        f"Rows with an unknown kernel or launch count (in neither queue): {len(unknown_count)}",
        f"Rows with best compile / (SOL + 3us * generated kernels) > 1.1x: {len(launch_gt11)}",
        f"Rows with best compile / (SOL + 3us * generated kernels) > 2.0x: {len(launch_gt2)}",
        f"Grouped prefixes above 1.1x: {len(groups)}",
        f"Interleaved repros currently covered: {len(covered_repros)}",
        "",
        "## 3-config winners so far",
    ]
    for label, count in sorted(config_wins.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {label}: {count}")
    lines.extend(["", "## Top raw SOL gaps (diagnostic for multi-kernel rows)", ""])
    for row in baseline[:25]:
        lines.append(
            f"- {row['best_compile_over_sol']:.2f}x `{row['repro_id']}`: "
            f"best={row['best_compile_label']} {row['best_compile_us']:.1f}us, "
            f"SOL={row['memcopy_sol_us']:.1f}us, generated kernels={row['n_kernels']}, ops={row['top_ops'][:160]}"
        )
    lines.extend(["", "## Single-kernel copy-roofline candidates", ""])
    for row in single_kernel_candidates[:25]:
        lines.append(
            f"- {row['best_compile_over_sol']:.2f}x `{row['repro_id']}`: "
            f"best={row['best_compile_us']:.1f}us, SOL={row['memcopy_sol_us']:.1f}us, "
            f"mitigations={row['sol_mitigation_factors'] or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Multi-kernel regions (ranked separately; raw copy gap is not a whole-region floor)",
            "",
        ]
    )
    for row in multi_kernel_candidates[:25]:
        lines.append(
            f"- {row['sol_comparison_class']} `{row['repro_id']}`: "
            f"best={row['best_compile_us']:.1f}us, raw={row['best_compile_over_sol']:.2f}x, "
            f"generated kernels={row['n_kernels']}, FX ops={row['n_ops']}, "
            f"combo_sweep={row['combo_sweep_recommended']}"
        )
    lines.extend(["", "## Largest grouped families", ""])
    for row in sorted(groups, key=lambda item: (item["count"], item["max_gap"]), reverse=True)[:25]:
        lines.append(
            f"- {row['pattern_prefix']}: count={row['count']}, max={row['max_gap']:.2f}x, "
            f"geomean={row['geomean_gap']:.2f}x, reps={row['representative_repros']}"
        )
    lines.extend(["", "## Top launch-adjusted priorities", ""])
    for row in launch_adjusted_priority_rows(baseline)[:25]:
        lines.append(
            f"- score={row['priority_score_3us']:.1f}, launch_gap={row['launch_adjusted_gap_3us']:.2f}x, "
            f"excess={row['excess_us_vs_launch_adjusted_sol_3us']:.1f}us, raw={row['raw_sol_gap']:.2f}x, "
            f"kernels={row['n_kernels']} `{row['repro_id']}`"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--interleaved", type=Path, default=DEFAULT_INTERLEAVED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    baseline_data = load_json(args.baseline)
    interleaved_data = load_json(args.interleaved)
    baseline = baseline_rows(baseline_data)
    baseline_by_repro = {row["repro_path"]: row for row in baseline}
    interleaved = interleaved_rows(interleaved_data, baseline_by_repro)
    candidates, single_kernel_candidates, multi_kernel_candidates = (
        split_sol_gap_candidates(baseline)
    )
    launch_adjusted_priorities = launch_adjusted_priority_rows(candidates)
    groups = grouped_rows(baseline)

    args.outdir.mkdir(parents=True, exist_ok=True)
    tables = {
        "baseline_results": baseline,
        "sol_gap_candidates": candidates,
        "single_kernel_roofline_candidates": single_kernel_candidates,
        "multi_kernel_region_candidates": multi_kernel_candidates,
        "interleaved_3config_results": interleaved,
        "gap_groups": groups,
        "launch_adjusted_priorities": launch_adjusted_priorities,
    }
    write_csv(args.outdir / "baseline_results.csv", baseline)
    write_csv(args.outdir / "sol_gap_candidates.csv", candidates)
    write_csv(
        args.outdir / "single_kernel_roofline_candidates.csv",
        single_kernel_candidates,
    )
    write_csv(
        args.outdir / "multi_kernel_region_candidates.csv",
        multi_kernel_candidates,
    )
    write_csv(args.outdir / "interleaved_3config_results.csv", interleaved)
    write_csv(args.outdir / "gap_groups.csv", groups)
    write_csv(args.outdir / "launch_adjusted_priorities.csv", launch_adjusted_priorities)
    write_sqlite(args.outdir / "benchmark_results.sqlite", tables)
    write_summary(
        args.outdir / "SOL_GAP_SUMMARY.md",
        baseline,
        interleaved,
        groups,
        {"baseline_path": str(args.baseline), "interleaved_path": str(args.interleaved)},
    )
    print(f"wrote {args.outdir}")
    print(f"baseline rows: {len(baseline)}")
    print(f">1.1x candidates: {len(candidates)}")
    print(f"single-kernel copy-roofline candidates: {len(single_kernel_candidates)}")
    print(f"multi-kernel logical-I/O candidates: {len(multi_kernel_candidates)}")
    print(f"interleaved config rows: {len(interleaved)}")
    print(f"launch-adjusted priority rows: {len(launch_adjusted_priorities)}")


if __name__ == "__main__":
    main()
