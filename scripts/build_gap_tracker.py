#!/usr/bin/env python3
"""
Build the oracle_vs_compile_tracker.csv from measurement logs.

Re-parses all measurement logs with improved parsing and classifies each
oracle as true_floor, partial_floor, not_true_floor, or failed.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_CSV = ROOT / "investigation_results" / "oracle_kernel_work_queue.csv"
SWEEP_JSON = ROOT / "sweep_3config_interleaved.json"
OUTPUT_CSV = ROOT / "investigation_results" / "oracle_vs_compile_tracker.csv"
LOG_DIR = ROOT / "investigation_results" / "oracle_measurement_logs"


def load_queue():
    with open(QUEUE_CSV) as f:
        return list(csv.DictReader(f))


def load_sweep_best_compile():
    with open(SWEEP_JSON) as f:
        data = json.load(f)
    best = {}
    for key, val in data.items():
        if key == "_metadata":
            continue
        parts = key.split("/")
        if len(parts) >= 3:
            repro_id = parts[2]
        else:
            continue
        if not isinstance(val, dict) or "configs" not in val:
            continue
        min_us = float("inf")
        for cfg_name, cfg_data in val["configs"].items():
            if isinstance(cfg_data, dict) and "us" in cfg_data:
                us = cfg_data["us"]
                if us < min_us:
                    min_us = us
        if min_us < float("inf"):
            best[repro_id] = min_us
    return best


def parse_oracle_from_log(log_path: Path) -> tuple[float | None, str, str]:
    """Parse oracle timing from log file.

    Returns (timing_us, impl_note, status).
    Status: true_floor, partial_floor, not_true_floor, scaffold, compile_error, runtime_error, failed
    """
    if not log_path.exists():
        return None, "", "no_log"

    content = log_path.read_text()

    # Check for errors
    if "Return code: 1" in content and "Return code: 0" not in content:
        if "CompilationError" in content or "arange" in content:
            return None, "", "compile_error"
        if "AssertionError" in content:
            return None, "", "runtime_error"
        if "UnboundLocalError" in content:
            return None, "", "code_bug"
        return None, "", "failed"

    # Empty stdout means scaffold with no real output
    if "--- STDOUT ---\n\n--- STDERR ---" in content or "--- STDOUT ---\n--- STDERR ---" in content:
        return None, "", "scaffold_no_output"

    # Parse output section
    stdout_match = re.search(r"--- STDOUT ---\n(.*?)(?:--- STDERR ---|$)", content, re.DOTALL)
    output = stdout_match.group(1) if stdout_match else content

    # Look for oracle_us= pattern (most reliable)
    m = re.search(r"oracle_us[=:]\s*([\d.]+)", output)
    if m:
        return float(m.group(1)), "oracle_us_parsed", "true_floor"

    # Look for specific oracle timing patterns (before compiled timing)
    # Pattern: "Oracle ...:" followed by "Median: X us"
    oracle_patterns = [
        # "Oracle (description): Median: X us" - two-line pattern
        r"Oracle[^:]*:.*?Median:\s*([\d.]+)\s*us",
        # "Oracle (description): X us" on same line
        r"Oracle[^:]*:\s*([\d.]+)\s*us",
        # "oracle (description): X us"
        r"oracle\s*\([^)]*\):\s*([\d.]+)\s*us",
        # "Oracle ... only: X us"
        r"Oracle[^:]*only:\s*([\d.]+)\s*us",
    ]

    for pat in oracle_patterns:
        m = re.search(pat, output, re.DOTALL | re.IGNORECASE)
        if m:
            val = float(m.group(1))
            # Check if this is labeled as "full" (includes upstream)
            line_context = m.group(0).lower()
            if "full" in line_context and "upstream" not in line_context:
                return val, "full_fused_oracle", "true_floor"
            elif "full" in line_context and "upstream" in line_context:
                # Skip "full including upstream" - look for the fused-only timing
                pass
            else:
                return val, "oracle_timing", "true_floor"

    # For oracles with "fused ... reduction+epilogue" pattern
    m = re.search(r"oracle\s*\(fused[^)]*\):\s*([\d.]+)\s*us", output, re.IGNORECASE)
    if m:
        return float(m.group(1)), "fused_oracle", "partial_floor"

    # Look for "Median: X us" - first occurrence is usually oracle
    medians = re.findall(r"Median:\s*([\d.]+)\s*us", output)
    if medians:
        return float(medians[0]), "first_median", "true_floor"

    return None, "", "parse_failure"


def classify_fix(family: str) -> str:
    family_map = {
        "online_softmax_cross_entropy": "online_softmax_fusion",
        "multi_output_reduction_templates": "multi_output_reduction",
        "structured_pool_upsample_backward_reduce": "scatter_reduce_fusion",
        "norm_template_canonicalization": "norm_template_fusion",
        "layout_indexing_stencil_fusion": "stencil_layout_fusion",
        "irregular_gather_reduce": "gather_reduce_fusion",
    }
    return family_map.get(family, family)


def classify_fix_status(gap_ratio: float, oracle_us: float, compile_us: float, notes: str) -> str:
    """Classify whether the gap is actionable."""
    if math.isnan(gap_ratio):
        return "unknown"
    if gap_ratio < 0.8:
        return "not_true_floor"  # Oracle is slower, not a real floor
    if gap_ratio < 1.1:
        return "at_floor"  # Already near floor
    if gap_ratio < 1.5:
        return "small_gap"  # Minor gap
    if gap_ratio < 3.0:
        return "needs_work"  # Meaningful gap
    return "large_gap"  # Huge opportunity


def main():
    print("=" * 70)
    print("Building Oracle vs Compile Gap Tracker")
    print("=" * 70)

    queue = load_queue()
    unmeasured = [r for r in queue if r.get("status") == "implemented_unmeasured"]
    best_compile = load_sweep_best_compile()

    print(f"Unmeasured oracles: {len(unmeasured)}")
    print(f"Sweep entries: {len(best_compile)}")

    results = []
    failures = []

    for row in unmeasured:
        repro_id = row["repro_id"]
        family = row.get("family", "unknown")
        oracle_path = row.get("canonical_oracle_path", "")

        log_file = LOG_DIR / f"{repro_id}.log"
        oracle_us, impl_note, status = parse_oracle_from_log(log_file)

        compile_us = best_compile.get(repro_id, float("nan"))

        if oracle_us is None:
            failures.append((repro_id, family, status))
            continue

        gap_ratio = compile_us / oracle_us if oracle_us > 0 and not math.isnan(compile_us) else float("nan")
        gap_us = compile_us - oracle_us if not math.isnan(compile_us) else float("nan")
        fix_status = classify_fix_status(gap_ratio, oracle_us, compile_us, row.get("notes", ""))

        results.append({
            "repro_id": repro_id,
            "family": family,
            "best_compile_us": f"{compile_us:.3f}" if not math.isnan(compile_us) else "",
            "oracle_floor_us": f"{oracle_us:.3f}",
            "gap_ratio": f"{gap_ratio:.3f}" if not math.isnan(gap_ratio) else "",
            "gap_us": f"{gap_us:.3f}" if not math.isnan(gap_us) else "",
            "inductor_fix": classify_fix(family),
            "fix_status": fix_status,
        })

    # Write tracker CSV
    fieldnames = ["repro_id", "family", "best_compile_us", "oracle_floor_us",
                  "gap_ratio", "gap_us", "inductor_fix", "fix_status"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Sort by gap_us descending (biggest opportunity first)
        results.sort(key=lambda r: -float(r["gap_us"]) if r["gap_us"] else 0)
        for r in results:
            writer.writerow(r)

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Successfully measured: {len(results)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print(f"\nFailures:")
        for rid, fam, reason in failures:
            print(f"  {rid} ({fam}): {reason}")

    # Separate true floors from non-floors
    true_floors = [r for r in results if r["fix_status"] not in ("not_true_floor", "unknown")]
    non_floors = [r for r in results if r["fix_status"] == "not_true_floor"]

    print(f"\n{'=' * 70}")
    print("TRUE FLOOR ORACLES (compile can reach these)")
    print(f"{'=' * 70}")
    print(f"{'repro_id':<35} {'compile':>10} {'oracle':>10} {'ratio':>7} {'gap_us':>10} {'status':>12}")
    print("-" * 90)
    for r in results:
        if r["fix_status"] == "not_true_floor":
            continue
        compile_str = f"{float(r['best_compile_us']):.1f}" if r["best_compile_us"] else "N/A"
        oracle_str = f"{float(r['oracle_floor_us']):.1f}"
        ratio_str = f"{float(r['gap_ratio']):.2f}x" if r["gap_ratio"] else "N/A"
        gap_str = f"{float(r['gap_us']):.1f}" if r["gap_us"] else "N/A"
        print(f"  {r['repro_id']:<33} {compile_str:>10} {oracle_str:>10} {ratio_str:>7} {gap_str:>10} {r['fix_status']:>12}")

    print(f"\n{'=' * 70}")
    print("NOT TRUE FLOORS (oracle slower than compile -- scaffold/upstream)")
    print(f"{'=' * 70}")
    for r in non_floors:
        compile_str = f"{float(r['best_compile_us']):.1f}" if r["best_compile_us"] else "N/A"
        oracle_str = f"{float(r['oracle_floor_us']):.1f}"
        print(f"  {r['repro_id']:<33} compile={compile_str}, oracle={oracle_str}")

    # Gap statistics for true floors only
    valid_floors = [r for r in true_floors if r["gap_ratio"] and float(r["gap_ratio"]) > 1.0]
    if valid_floors:
        ratios = [float(r["gap_ratio"]) for r in valid_floors]
        gaps = [float(r["gap_us"]) for r in valid_floors]

        total_gap = sum(gaps)
        geomean_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
        gt_1_5x = sum(1 for r in ratios if r > 1.5)
        gt_2x = sum(1 for r in ratios if r > 2.0)
        gt_3x = sum(1 for r in ratios if r > 3.0)

        print(f"\n{'=' * 70}")
        print(f"GAP STATISTICS (true floors with positive gap: {len(valid_floors)})")
        print(f"{'=' * 70}")
        print(f"  Total recoverable gap: {total_gap:.1f} us")
        print(f"  Geomean gap ratio: {geomean_ratio:.3f}x")
        print(f"  Oracles >1.5x from floor: {gt_1_5x}/{len(valid_floors)}")
        print(f"  Oracles >2x from floor: {gt_2x}/{len(valid_floors)}")
        print(f"  Oracles >3x from floor: {gt_3x}/{len(valid_floors)}")

        # By family
        print(f"\n  By inductor fix family:")
        families = {}
        for r in valid_floors:
            fix = r["inductor_fix"]
            if fix not in families:
                families[fix] = []
            families[fix].append(r)

        for fix, fam_results in sorted(families.items(), key=lambda x: -sum(float(r["gap_us"]) for r in x[1])):
            fam_gaps = [float(r["gap_us"]) for r in fam_results]
            fam_ratios = [float(r["gap_ratio"]) for r in fam_results]
            fam_geomean = math.exp(sum(math.log(r) for r in fam_ratios) / len(fam_ratios))
            print(f"    {fix}: {len(fam_results)} oracles, "
                  f"total_gap={sum(fam_gaps):.1f} us, "
                  f"geomean_ratio={fam_geomean:.2f}x, "
                  f"max_gap={max(fam_gaps):.1f} us")

    print(f"\nTracker written to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
