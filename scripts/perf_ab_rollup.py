#!/usr/bin/env python3
"""Compare two corpus sweeps using exact kernel-shape matches.

The kernel report needs only two ``bench_parallel.py`` JSON files. Optional
model occurrence sidecars add an extern-diluted per-model estimate; incomplete
or approximate model estimates are listed but excluded from aggregate results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMING_FIELDS = ("compiled_us", "coord_descent_us")


def _phash(repro_path: str) -> str:
    return Path(repro_path).parent.name.rsplit("_", 1)[-1]


def _is_genai(name: str) -> bool:
    return name.partition("/")[0].lower() == "genai"


def _selected_timing(measurement: dict, timing: str) -> tuple[object, str]:
    if timing == "auto":
        for field in ("coord_descent_us", "compiled_us"):
            value = measurement.get(field)
            if value is not None:
                return value, field
        return None, ""
    return measurement.get(timing), timing


def load_arm_points(path: Path, timing: str = "auto") -> dict:
    """Load uniquely identified points from one timing axis.

    ``auto`` preserves the historical preference for coordinate-descent timing,
    with per-point fallback to compiled timing. Explicit axes never fall back.
    """
    if timing not in (*TIMING_FIELDS, "auto"):
        raise ValueError(f"unknown timing axis: {timing}")
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: sweep root must be an object")

    points = {}
    for repro_path, measurements in data.items():
        if (
            not isinstance(repro_path, str)
            or repro_path.startswith("_")
            or not isinstance(measurements, dict)
        ):
            continue
        pattern_hash = _phash(repro_path)
        repro_dir = Path(repro_path).parent.name
        for label, measurement in measurements.items():
            if not isinstance(label, str) or not isinstance(measurement, dict):
                continue
            raw_us, selected_field = _selected_timing(measurement, timing)
            if raw_us is None:
                continue
            if isinstance(raw_us, bool):
                raise ValueError(
                    f"{path}: {repro_path}/{label} has invalid "
                    f"{selected_field}={raw_us!r}"
                )
            try:
                us = float(raw_us)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: {repro_path}/{label} has invalid "
                    f"{selected_field}={raw_us!r}"
                ) from exc
            if not math.isfinite(us) or us <= 0:
                raise ValueError(
                    f"{path}: {repro_path}/{label} has invalid "
                    f"{selected_field}={raw_us!r}"
                )
            shape_hash = label.rsplit("_", 1)[-1]
            key = (pattern_hash, shape_hash)
            if key in points:
                raise ValueError(
                    f"{path}: duplicate pattern/shape point {pattern_hash}/{shape_hash}"
                )
            points[key] = {
                "pattern_hash": pattern_hash,
                "shape_hash": shape_hash,
                "repro_dir": repro_dir,
                "label": label,
                "timing": selected_field,
                "us": us,
            }
    return points


def load_arm(path: Path, timing: str = "auto") -> dict:
    return {key: point["us"] for key, point in load_arm_points(path, timing).items()}


def paired_arms(base_path: Path, head_path: Path, timing: str):
    """Return exact arms and a same-shape representative for each pattern."""
    base = load_arm(base_path, timing)
    head = load_arm(head_path, timing)
    common_by_pattern = defaultdict(list)
    for pattern_hash, shape_hash in set(base) & set(head):
        common_by_pattern[pattern_hash].append(shape_hash)

    representatives = {}
    for pattern_hash, shapes in common_by_pattern.items():
        ordered = sorted(shapes, key=lambda shape: (base[(pattern_hash, shape)], shape))
        shape = ordered[len(ordered) // 2]
        representatives[pattern_hash] = (
            base[(pattern_hash, shape)],
            head[(pattern_hash, shape)],
            shape,
        )
    return base, head, representatives


def fusible_us(base, head, representatives, pattern_hash, shape_hash):
    key = (pattern_hash, shape_hash)
    if key in base and key in head:
        return base[key], head[key], "exact"
    if key in base or key in head:
        return None, None, "unmatched"
    if pattern_hash in representatives:
        base_us, head_us, _ = representatives[pattern_hash]
        return base_us, head_us, "paired_pattern_fallback"
    return None, None, "unmatched"


def _occurrence_count(value, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context}: occurrence count must be a positive integer")
    return value


def _finite(value: float, context: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{context} is not finite")
    return value


def _reduction_pct(base_us: float, head_us: float, context: str) -> float:
    return _finite((1.0 - head_us / base_us) * 100.0, context)


def _speedup(base_us: float, head_us: float, context: str) -> float:
    return _finite(base_us / head_us, context)


def _geomean_speedup(pairs, context: str) -> float:
    log_ratio = statistics.mean(
        math.log(base_us) - math.log(head_us) for base_us, head_us in pairs
    )
    try:
        speedup = math.exp(log_ratio)
    except OverflowError as exc:
        raise ValueError(f"{context} is not finite") from exc
    return _finite(
        (_finite(speedup, context) - 1.0) * 100.0,
        f"{context} percent",
    )


def _weighted_add(total: float, value: float, count: int, context: str) -> float:
    try:
        result = total + value * count
    except OverflowError as exc:
        raise ValueError(f"{context} is not finite") from exc
    return _finite(result, context)


def _load_occurrence_record(path: Path) -> dict:
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{path}: occurrence sidecar must be an object")
    for field in ("suite", "mode", "model"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{path}: missing non-empty {field}")
    if not isinstance(record.get("fusible"), dict):
        raise ValueError(f"{path}: fusible must be an object")
    if not isinstance(record.get("extern"), list):
        raise ValueError(f"{path}: extern must be a list")
    trace_errors = record.get("trace_errors", [])
    if not isinstance(trace_errors, list):
        raise ValueError(f"{path}: trace_errors must be a list")
    record["trace_errors"] = trace_errors
    return record


def rollup_models(
    base_path: Path,
    head_path: Path,
    occdir: Path,
    timing: str = "auto",
):
    """Return complete per-model estimates and explicit exclusion accounting."""
    base, head, representatives = paired_arms(base_path, head_path, timing)
    models = {}
    per_model_coverage = {}
    sidecars = sorted(
        path for path in Path(occdir).glob("*.json") if not path.name.startswith("_")
    )
    if not sidecars:
        raise ValueError(f"no occurrence sidecars found in {occdir}")

    for sidecar in sidecars:
        record = _load_occurrence_record(sidecar)
        name = f"{record['suite']}/{record['mode']}/{record['model']}"
        if name in models:
            raise ValueError(
                f"{sidecar}: duplicate occurrence accounting for model {name}"
            )

        fusible_base = 0.0
        delta = 0.0
        counts = {"exact": 0, "paired_pattern_fallback": 0, "unmatched": 0}
        fallback_shapes = {}
        for pattern_hash, shapes in record["fusible"].items():
            if not isinstance(pattern_hash, str) or not isinstance(shapes, dict):
                raise ValueError(
                    f"{sidecar}: fusible entries must map pattern to shapes"
                )
            for shape_hash, raw_count in shapes.items():
                count = _occurrence_count(
                    raw_count, f"{sidecar}:{pattern_hash}/{shape_hash}"
                )
                base_us, head_us, kind = fusible_us(
                    base, head, representatives, pattern_hash, shape_hash
                )
                counts[kind] += count
                if base_us is None or head_us is None:
                    continue
                if kind == "paired_pattern_fallback":
                    fallback_shapes[f"{pattern_hash}/{shape_hash}"] = representatives[
                        pattern_hash
                    ][2]
                fusible_base = _weighted_add(
                    fusible_base,
                    base_us,
                    count,
                    f"{sidecar}: model baseline",
                )
                delta = _weighted_add(
                    delta,
                    base_us - head_us,
                    count,
                    f"{sidecar}: model delta",
                )

        extern_total = 0.0
        extern_counts = {"priced": 0, "unpriced": 0}
        for extern in record["extern"]:
            if not isinstance(extern, dict):
                raise ValueError(f"{sidecar}: every extern entry must be an object")
            count = _occurrence_count(
                extern.get("count"),
                f"{sidecar}:{extern.get('target', 'extern')}",
            )
            raw_us = extern.get("baseline_us")
            if raw_us is None:
                extern_counts["unpriced"] += count
                continue
            try:
                us = float(raw_us)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{sidecar}: invalid extern baseline_us={raw_us!r}"
                ) from exc
            if not math.isfinite(us) or us <= 0:
                raise ValueError(f"{sidecar}: invalid extern baseline_us={raw_us!r}")
            extern_counts["priced"] += count
            extern_total = _weighted_add(
                extern_total,
                us,
                count,
                f"{sidecar}: extern baseline",
            )

        denominator = _finite(fusible_base + extern_total, f"{sidecar}: model baseline")
        delta = _finite(delta, f"{sidecar}: model delta")
        exclusion_reasons = []
        if counts["paired_pattern_fallback"]:
            exclusion_reasons.append("paired_pattern_fallback")
        if counts["unmatched"]:
            exclusion_reasons.append("unmatched_kernel")
        if extern_counts["unpriced"]:
            exclusion_reasons.append("unpriced_extern")
        if record["trace_errors"]:
            exclusion_reasons.append("trace_errors")
        if denominator <= 0:
            exclusion_reasons.append("no_priced_baseline")
        complete = not exclusion_reasons
        models[name] = (
            _finite(delta / denominator * 100.0, f"{sidecar}: model estimate")
            if complete
            else None
        )
        per_model_coverage[name] = {
            **counts,
            "extern": extern_counts,
            "trace_errors": len(record["trace_errors"]),
            "paired_pattern_fallback_shapes": fallback_shapes,
            "priced": denominator > 0,
            "exclusion_reasons": exclusion_reasons,
            "included": complete,
        }

    model_names = [name for name in models if not _is_genai(name)]
    micro_names = [name for name in models if _is_genai(name)]

    def scope(names):
        included = sum(per_model_coverage[name]["included"] for name in names)
        return {
            "total": len(names),
            "included": included,
            "excluded": len(names) - included,
        }

    coverage = {
        "models": scope(model_names),
        "microbenchmarks": scope(micro_names),
        "kernel_occurrences": {
            key: sum(item[key] for item in per_model_coverage.values())
            for key in ("exact", "paired_pattern_fallback", "unmatched")
        },
        "extern_occurrences": {
            key: sum(item["extern"][key] for item in per_model_coverage.values())
            for key in ("priced", "unpriced")
        },
        "trace_errors": sum(
            item["trace_errors"] for item in per_model_coverage.values()
        ),
        "excluded_entries": sorted(
            name for name, item in per_model_coverage.items() if not item["included"]
        ),
        "per_model": per_model_coverage,
    }
    return models, coverage


def cut(models, genai_excl=True):
    pcts = [
        pct
        for name, pct in models.items()
        if pct is not None and (not genai_excl or not _is_genai(name))
    ]
    if not pcts:
        return None
    log_ratios = []
    for pct in pcts:
        if not math.isfinite(pct) or pct >= 100.0:
            raise ValueError(f"invalid model reduction percent: {pct!r}")
        log_ratios.append(-math.log1p(-pct / 100.0))
    try:
        geomean = math.exp(statistics.mean(log_ratios))
    except OverflowError as exc:
        raise ValueError("model geomean speedup is not finite") from exc
    return {
        "n": len(pcts),
        "median": round(statistics.median(pcts), 4),
        "mean": round(statistics.mean(pcts), 4),
        "geomean": round(
            _finite(
                (_finite(geomean, "model geomean speedup") - 1.0) * 100.0,
                "model geomean speedup percent",
            ),
            4,
        ),
    }


def fusible_kernel_summary(base_path: Path, head_path: Path, timing: str = "auto"):
    """Summarize the exact intersection of kernel-shape timing points."""
    base = load_arm(base_path, timing)
    head = load_arm(head_path, timing)
    common = set(base) & set(head)
    if not common:
        return None
    reductions = [
        _reduction_pct(base[key], head[key], f"{key[0]}/{key[1]} reduction")
        for key in common
    ]
    pairs = [(base[key], head[key]) for key in common]
    return {
        "n": len(common),
        "median": round(statistics.median(reductions), 4),
        "mean": round(statistics.mean(reductions), 4),
        "geomean": round(_geomean_speedup(pairs, "kernel geomean speedup"), 4),
    }


def kernel_mover_summary(
    base_path: Path,
    head_path: Path,
    timing: str = "auto",
    limit: int = 10,
):
    """Rank exact kernel-shape improvements and regressions."""
    base = load_arm_points(base_path, timing)
    head = load_arm_points(head_path, timing)
    rows = []
    for key in sorted(set(base) & set(head)):
        base_point = base[key]
        head_point = head[key]
        base_us = base_point["us"]
        head_us = head_point["us"]
        context = f"{key[0]}/{key[1]}"
        reduction = _reduction_pct(base_us, head_us, f"{context} reduction")
        rows.append(
            {
                "pattern_hash": key[0],
                "shape_hash": key[1],
                "repro_dir": base_point["repro_dir"],
                "label": base_point["label"],
                "base_us": round(base_us, 4),
                "head_us": round(head_us, 4),
                "reduction_pct": round(reduction, 4),
                "speedup": round(_speedup(base_us, head_us, f"{context} speedup"), 4),
                "_reduction": reduction,
            }
        )

    improvements = sorted(
        (row for row in rows if row["_reduction"] > 0),
        key=lambda row: (-row["_reduction"], -row["base_us"], row["label"]),
    )
    regressions = sorted(
        (row for row in rows if row["_reduction"] < 0),
        key=lambda row: (row["_reduction"], -row["base_us"], row["label"]),
    )

    def public(ranked):
        return [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in ranked[:limit]
        ]

    return {
        "limit": limit,
        "matched_points": len(rows),
        "base_only_points": len(set(base) - set(head)),
        "head_only_points": len(set(head) - set(base)),
        "top_improvements": public(improvements),
        "top_regressions": public(regressions),
    }


def segment_by_suite_mode(per_model: dict):
    groups = defaultdict(dict)
    for name, pct in per_model.items():
        parts = name.split("/")
        if len(parts) >= 3:
            groups[f"{parts[0].lower()}/{parts[1].lower()}"][name] = pct

    rows = []
    for suite_mode, entries in groups.items():
        summary = cut(entries, genai_excl=False) or {
            "n": 0,
            "median": None,
            "mean": None,
            "geomean": None,
        }
        rows.append(
            (
                suite_mode,
                {
                    **summary,
                    "total": len(entries),
                    "excluded": sum(pct is None for pct in entries.values()),
                },
            )
        )
    return sorted(
        rows,
        key=lambda row: (
            row[1]["geomean"] if row[1]["geomean"] is not None else -math.inf
        ),
        reverse=True,
    )


def careful_rollup(
    base_json: Path,
    head_json: Path,
    occdir: Path,
    include_genai: bool = False,
    timing: str = "auto",
):
    """Compatibility wrapper for callers of the original script API."""
    del include_genai
    per_model, _ = rollup_models(base_json, head_json, occdir, timing)
    summaries = {
        "genai_excl": cut(per_model, genai_excl=True),
        "genai_incl": cut(per_model, genai_excl=False),
    }
    return per_model, summaries


def _fmt_stats(summary: dict | None) -> str:
    if not summary:
        return "(no matched entries)"
    return (
        f"median reduction {summary['median']:+.3f}%   "
        f"mean reduction {summary['mean']:+.3f}%   "
        f"geomean speedup {summary['geomean']:+.3f}%   n={summary['n']}"
    )


def print_report(
    *,
    base_json: Path,
    head_json: Path,
    timing: str,
    fusible: dict,
    kernel_movers: dict,
    per_model: dict | None,
    summaries: dict | None,
    model_coverage: dict | None,
    include_genai: bool,
    top_models: int,
):
    print("=" * 78)
    print("A/B rollup (exact kernel-shape intersection)")
    print(f"  base: {base_json}")
    print(f"  head: {head_json}")
    print(f"  timing: {timing}")
    print("=" * 78)
    print("FUSIBLE KERNEL")
    print("  " + _fmt_stats(fusible))
    print(
        "KERNEL POINT COVERAGE  "
        f"matched={kernel_movers['matched_points']}  "
        f"base-only={kernel_movers['base_only_points']}  "
        f"head-only={kernel_movers['head_only_points']}"
    )

    if kernel_movers["limit"]:
        improvements = kernel_movers["top_improvements"]
        regressions = kernel_movers["top_regressions"]
        if improvements:
            print(f"TOP KERNEL IMPROVEMENTS (top {len(improvements)})")
            for row in improvements:
                print(
                    f"  {row['reduction_pct']:+7.2f}% reduction  "
                    f"{row['speedup']:5.2f}x  "
                    f"{row['base_us']:9.1f}->{row['head_us']:9.1f}us  "
                    f"{row['repro_dir']}[{row['label']}]"
                )
        if regressions:
            print(f"TOP KERNEL REGRESSIONS (top {len(regressions)}; worst first)")
            for row in regressions:
                print(
                    f"  {-row['reduction_pct']:+7.2f}% slower     "
                    f"{row['speedup']:5.2f}x  "
                    f"{row['base_us']:9.1f}->{row['head_us']:9.1f}us  "
                    f"{row['repro_dir']}[{row['label']}]"
                )

    print()
    print("MODEL (projected e2e; genai microbenchmarks excluded)")
    if per_model is None:
        print("  skipped: no occurrence accounting supplied")
        print()
        return
    print("  " + _fmt_stats(summaries["genai_excl"]))
    coverage = model_coverage
    print(
        f"  included models={coverage['models']['included']}/"
        f"{coverage['models']['total']}  microbenchmarks="
        f"{coverage['microbenchmarks']['included']}/"
        f"{coverage['microbenchmarks']['total']}"
    )
    occurrences = coverage["kernel_occurrences"]
    extern = coverage["extern_occurrences"]
    print(
        f"  kernel occurrences exact={occurrences['exact']}  "
        f"fallback={occurrences['paired_pattern_fallback']}  "
        f"unmatched={occurrences['unmatched']}; "
        f"extern priced={extern['priced']}  unpriced={extern['unpriced']}"
    )
    if coverage["excluded_entries"]:
        print("  excluded incomplete entries:")
        for name in coverage["excluded_entries"]:
            reasons = ",".join(coverage["per_model"][name]["exclusion_reasons"])
            print(f"    {name}: {reasons}")

    suite_rows = segment_by_suite_mode(per_model)
    if suite_rows:
        print()
        print("SUITE/MODE (genai rows are microbenchmarks)")
        for suite_mode, summary in suite_rows:
            geomean = (
                f"{summary['geomean']:+.3f}%"
                if summary["geomean"] is not None
                else "n/a"
            )
            print(
                f"  {geomean:>9s}  n={summary['n']:3d}  "
                f"excluded={summary['excluded']:2d}/{summary['total']:<3d}  "
                f"{suite_mode}"
            )

    if top_models:
        ranked = [
            (name, pct)
            for name, pct in per_model.items()
            if pct is not None and (include_genai or not _is_genai(name))
        ]
        ranked.sort(key=lambda item: -item[1])
        if ranked:
            print()
            print(f"TOP MODELS (top {top_models})")
            for name, pct in ranked[:top_models]:
                print(f"  {pct:+7.2f}%  {name}")
            if len(ranked) > top_models:
                print(f"BOTTOM MODELS (bottom {min(top_models, len(ranked))})")
                for name, pct in reversed(ranked[-top_models:]):
                    print(f"  {pct:+7.2f}%  {name}")
    print()


def print_kernel_table(base_json: Path, head_json: Path):
    """Print the existing ``bench_report.py`` comparison for extra detail."""
    print("=" * 78)
    print("RAW bench_report.py COMPARISON (compiled_us; directional detail only)")
    print("=" * 78)
    print("This table is not exact-shape weighted or extern-diluted.")
    report = ROOT / "scripts/bench_report.py"
    proc = subprocess.run(
        [sys.executable, str(report), "--compare", str(base_json), str(head_json)],
        capture_output=True,
        text=True,
    )
    sys.stdout.write(proc.stdout)
    if proc.returncode:
        sys.stderr.write(proc.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Shape-aligned A/B rollup of two corpus sweep JSONs"
    )
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--head", required=True, type=Path)
    parser.add_argument(
        "--occdir",
        type=Path,
        help="Optional per-model occurrence sidecars",
    )
    parser.add_argument(
        "--timing",
        choices=("auto", *TIMING_FIELDS),
        default="auto",
        help="Timing axis. auto preserves coord_descent_us then compiled_us fallback.",
    )
    parser.add_argument(
        "--include-genai",
        action="store_true",
        help="Include genai microbenchmarks in detailed top/bottom rankings",
    )
    parser.add_argument("--kernel-table", action="store_true")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--top-kernels", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    for label, path in (("base", args.base), ("head", args.head)):
        if not path.is_file():
            parser.error(f"{label} JSON not found: {path}")
    if args.top < 0 or args.top_kernels < 0:
        parser.error("--top and --top-kernels must be non-negative")

    occdir = args.occdir
    if occdir is not None and not occdir.is_dir():
        parser.error(f"occurrence directory not found: {occdir}")

    try:
        fusible = fusible_kernel_summary(args.base, args.head, args.timing)
        movers = kernel_mover_summary(
            args.base, args.head, args.timing, args.top_kernels
        )
        per_model = summaries = coverage = None
        if occdir is not None:
            per_model, coverage = rollup_models(
                args.base, args.head, occdir, args.timing
            )
            summaries = {
                "genai_excl": cut(per_model, genai_excl=True),
                "genai_incl": cut(per_model, genai_excl=False),
            }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    report = {
        "base": str(args.base),
        "head": str(args.head),
        "timing": args.timing,
        "include_genai": args.include_genai,
        "include_genai_in_rankings": args.include_genai,
        "metric_semantics": {
            "median": "latency reduction percent",
            "mean": "latency reduction percent",
            "geomean": "geometric mean speedup percent",
        },
        "fusible_kernel": fusible,
        "kernel_movers": movers,
        "summaries": summaries,
        "suite_mode": (
            dict(segment_by_suite_mode(per_model)) if per_model is not None else {}
        ),
        "coverage": coverage,
        "per_model": per_model or {},
    }
    if args.json:
        try:
            rendered = json.dumps(report, indent=2, allow_nan=False)
        except (OverflowError, ValueError) as exc:
            parser.error(f"could not serialize finite report: {exc}")
        print(rendered)
        return

    print_report(
        base_json=args.base,
        head_json=args.head,
        timing=args.timing,
        fusible=fusible,
        kernel_movers=movers,
        per_model=per_model,
        summaries=summaries,
        model_coverage=coverage,
        include_genai=args.include_genai,
        top_models=args.top,
    )
    if args.kernel_table:
        print_kernel_table(args.base, args.head)


if __name__ == "__main__":
    main()
