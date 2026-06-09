#!/usr/bin/env python3
"""
Batch measure all implemented_unmeasured oracle kernels and build a gap tracker.

Handles different oracle CLI patterns:
  - online_softmax / cross_entropy: run without flags (bench is default), use --no-compile
  - multi_output_reduction / norm_template: need --bench flag
  - stencil/scatter: bench by default (unless --no-bench)

Parses output for timing lines containing 'oracle_us=' or 'Median:' patterns.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_CSV = ROOT / "investigation_results" / "oracle_kernel_work_queue.csv"
SWEEP_JSON = ROOT / "sweep_3config_interleaved.json"
OUTPUT_CSV = ROOT / "investigation_results" / "oracle_vs_compile_tracker.csv"
LOG_DIR = ROOT / "investigation_results" / "oracle_measurement_logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)


def load_unmeasured():
    """Load all implemented_unmeasured rows from the work queue."""
    with open(QUEUE_CSV) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return [r for r in rows if r.get("status") == "implemented_unmeasured"]


def load_sweep_best_compile():
    """Load best compile time (min across 3 configs) for each repro from sweep."""
    with open(SWEEP_JSON) as f:
        data = json.load(f)

    best = {}
    for key, val in data.items():
        if key == "_metadata":
            continue
        # Extract repro_id from path like repros/canonical/REPRO_ID/repro.py
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


def determine_run_args(oracle_path: str, family: str) -> list[str]:
    """Determine the right CLI args based on oracle type."""
    path_str = oracle_path.lower()

    # Read the file to check for --bench flag
    try:
        content = Path(ROOT / oracle_path).read_text()
    except Exception:
        content = ""

    # Check what arguments are available
    has_bench_flag = '--bench' in content and 'add_argument' in content and '"--bench"' in content
    has_no_compile = '--no-compile' in content
    has_no_bench = '--no-bench' in content

    args = [sys.executable, str(ROOT / oracle_path)]

    if has_bench_flag:
        # multi_output_reduction, norm_template: need explicit --bench
        args.append("--bench")
    # online_softmax / cross_entropy: run by default, skip compile for speed
    if has_no_compile:
        args.append("--no-compile")

    # Add --no-append to avoid writing to shared CSVs during batch
    if '--no-append' in content:
        args.append("--no-append")

    return args


def parse_oracle_timing(output: str) -> float | None:
    """Parse oracle timing from stdout/stderr output.

    Looks for patterns like:
      oracle_us=1234.567
      Median: 1234.5 us
      Oracle ...: 1234.5 us
      oracle_us: 1234.567
    """
    # Pattern 1: oracle_us=NUMBER
    m = re.search(r"oracle_us[=:]\s*([\d.]+)", output)
    if m:
        return float(m.group(1))

    # Pattern 2: "Median: NUMBER us"
    m = re.search(r"Median:\s*([\d.]+)\s*us", output)
    if m:
        return float(m.group(1))

    # Pattern 3: "Oracle ... : NUMBER us"
    m = re.search(r"Oracle[^:]*:\s*([\d.]+)\s*us", output)
    if m:
        return float(m.group(1))

    # Pattern 4: look for "XXX us" on a line by itself or after timing label
    # Search from bottom up (last timing is usually the relevant one)
    lines = output.strip().split("\n")
    for line in reversed(lines):
        m = re.search(r"([\d.]+)\s*us", line)
        if m and "SOL" not in line and "sol" not in line.lower():
            val = float(m.group(1))
            if val > 0.1:  # Sanity: skip obviously wrong values
                return val

    return None


def run_oracle(repro_id: str, oracle_path: str, family: str) -> tuple[float | None, str]:
    """Run a single oracle and return (timing_us, log_text)."""
    full_path = ROOT / oracle_path
    if not full_path.exists():
        return None, f"ERROR: file not found: {oracle_path}"

    args = determine_run_args(oracle_path, family)
    cmd_str = " ".join(args)

    log_lines = [f"Command: {cmd_str}", f"CWD: {ROOT}", ""]

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=180,  # 3 min timeout
            cwd=str(ROOT),
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        output = result.stdout + "\n" + result.stderr
        log_lines.append(f"Return code: {result.returncode}")
        log_lines.append(f"--- STDOUT ---\n{result.stdout}")
        log_lines.append(f"--- STDERR ---\n{result.stderr}")

        if result.returncode != 0:
            # Try without --bench if it failed
            if "--bench" in args:
                args_retry = [a for a in args if a != "--bench"]
                result2 = subprocess.run(
                    args_retry,
                    capture_output=True,
                    text=True,
                    timeout=180,
                    cwd=str(ROOT),
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
                )
                output = result2.stdout + "\n" + result2.stderr
                log_lines.append(f"\n--- RETRY without --bench ---")
                log_lines.append(f"Return code: {result2.returncode}")
                log_lines.append(f"--- STDOUT ---\n{result2.stdout}")
                log_lines.append(f"--- STDERR ---\n{result2.stderr}")

        timing = parse_oracle_timing(output)
        return timing, "\n".join(log_lines)

    except subprocess.TimeoutExpired:
        return None, "\n".join(log_lines + ["ERROR: timeout (180s)"])
    except Exception as e:
        return None, "\n".join(log_lines + [f"ERROR: {e}"])


def classify_fix(family: str) -> str:
    """Classify what inductor fix family would close the gap."""
    family_map = {
        "online_softmax_cross_entropy": "online_softmax_fusion",
        "multi_output_reduction_templates": "multi_output_reduction",
        "structured_pool_upsample_backward_reduce": "scatter_reduce_fusion",
        "norm_template_canonicalization": "norm_template_fusion",
        "layout_indexing_stencil_fusion": "stencil_layout_fusion",
        "irregular_gather_reduce": "gather_reduce_fusion",
    }
    return family_map.get(family, family)


def main():
    print("=" * 70)
    print("Oracle Kernel Batch Measurement")
    print("=" * 70)

    unmeasured = load_unmeasured()
    print(f"\nFound {len(unmeasured)} implemented_unmeasured oracles")

    best_compile = load_sweep_best_compile()
    print(f"Loaded {len(best_compile)} repro timings from sweep_3config_interleaved.json")

    results = []
    failures = []

    for i, row in enumerate(unmeasured):
        repro_id = row["repro_id"]
        oracle_path = row.get("canonical_oracle_path", "")
        family = row.get("family", "unknown")

        print(f"\n[{i+1}/{len(unmeasured)}] {repro_id} ({family})")
        print(f"  Oracle: {oracle_path}")

        if not oracle_path:
            print(f"  SKIP: no oracle path")
            failures.append((repro_id, "no_oracle_path"))
            continue

        timing_us, log_text = run_oracle(repro_id, oracle_path, family)

        # Save log
        log_file = LOG_DIR / f"{repro_id}.log"
        with open(log_file, "w") as f:
            f.write(log_text)

        if timing_us is None:
            print(f"  FAILED: could not parse timing")
            failures.append((repro_id, "parse_failure"))
            continue

        compile_us = best_compile.get(repro_id)
        if compile_us is None:
            print(f"  WARNING: no compile timing in sweep for {repro_id}")
            compile_us = float("nan")

        gap_ratio = compile_us / timing_us if timing_us > 0 else float("nan")
        gap_us = compile_us - timing_us if not math.isnan(compile_us) else float("nan")

        print(f"  Oracle: {timing_us:.1f} us")
        if not math.isnan(compile_us):
            print(f"  Best compile: {compile_us:.1f} us")
            print(f"  Gap ratio: {gap_ratio:.2f}x")
            print(f"  Gap: {gap_us:.1f} us")

        results.append({
            "repro_id": repro_id,
            "family": family,
            "best_compile_us": f"{compile_us:.3f}" if not math.isnan(compile_us) else "",
            "oracle_floor_us": f"{timing_us:.3f}",
            "gap_ratio": f"{gap_ratio:.3f}" if not math.isnan(gap_ratio) else "",
            "gap_us": f"{gap_us:.3f}" if not math.isnan(gap_us) else "",
            "inductor_fix": classify_fix(family),
            "fix_status": "needs_work",
        })

    # Write tracker CSV
    print(f"\n{'=' * 70}")
    print(f"Writing tracker to {OUTPUT_CSV}")
    fieldnames = ["repro_id", "family", "best_compile_us", "oracle_floor_us",
                  "gap_ratio", "gap_us", "inductor_fix", "fix_status"]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary stats
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total oracles attempted: {len(unmeasured)}")
    print(f"Successfully measured: {len(results)}")
    print(f"Failed: {len(failures)}")

    if failures:
        print(f"\nFailures:")
        for rid, reason in failures:
            print(f"  {rid}: {reason}")

    if results:
        # Filter to those with valid gap ratios
        valid = [r for r in results if r["gap_ratio"]]
        if valid:
            ratios = [float(r["gap_ratio"]) for r in valid]
            gaps = [float(r["gap_us"]) for r in valid]

            total_gap = sum(gaps)
            geomean_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
            gt_1_5x = sum(1 for r in ratios if r > 1.5)
            gt_2x = sum(1 for r in ratios if r > 2.0)
            gt_3x = sum(1 for r in ratios if r > 3.0)

            print(f"\nGap Statistics ({len(valid)} oracles with valid data):")
            print(f"  Total recoverable gap: {total_gap:.1f} us")
            print(f"  Geomean gap ratio: {geomean_ratio:.3f}x")
            print(f"  Oracles >1.5x from floor: {gt_1_5x}/{len(valid)}")
            print(f"  Oracles >2x from floor: {gt_2x}/{len(valid)}")
            print(f"  Oracles >3x from floor: {gt_3x}/{len(valid)}")

            # By family
            print(f"\nBy family:")
            families = {}
            for r in valid:
                fam = r["family"]
                if fam not in families:
                    families[fam] = []
                families[fam].append(r)

            for fam, fam_results in sorted(families.items(), key=lambda x: -sum(float(r["gap_us"]) for r in x[1])):
                fam_gaps = [float(r["gap_us"]) for r in fam_results]
                fam_ratios = [float(r["gap_ratio"]) for r in fam_results]
                fam_geomean = math.exp(sum(math.log(r) for r in fam_ratios) / len(fam_ratios))
                print(f"  {fam}: {len(fam_results)} oracles, "
                      f"total_gap={sum(fam_gaps):.1f} us, "
                      f"geomean_ratio={fam_geomean:.2f}x")

    print(f"\nTracker written to: {OUTPUT_CSV}")
    print(f"Logs in: {LOG_DIR}/")


if __name__ == "__main__":
    main()
