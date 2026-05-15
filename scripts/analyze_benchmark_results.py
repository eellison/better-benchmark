"""Analyze benchmark results from TORCH_COMPILE_DEBUG output.

Extracts kernels, identifies patterns, computes statistics, and generates
a comprehensive report of code generation quality across the benchmark suite.

Usage:
    python analyze_benchmark_results.py /tmp/benchmark_traces/huggingface/ [/tmp/benchmark_traces_training/huggingface/]
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from extract_kernels_from_debug import find_output_code_files, parse_kernel_info, get_model_name_from_path


def analyze_kernel(kernel_info):
    """Analyze a single kernel for potential issues."""
    meta = kernel_info["meta"]
    issues = []

    kt = meta.get("kernel_type", "")
    hint = meta.get("reduction_hint", "")
    tiling = meta.get("tiling_scores", {})
    size_hints = meta.get("size_hints", {})

    # Check for non-persistent reductions that might benefit from being persistent
    if kt == "reduction" and hint == "INNER":
        # Non-persistent with INNER hint — check if rnumel is small enough for persistent
        r_size = size_hints.get("r0_", 0)
        if isinstance(r_size, int) and 64 < r_size <= 2048:
            issues.append(f"non-persistent reduction with r0_={r_size} (could be persistent)")

    # Check for DEFAULT hint that should be INNER (our tiling_scores fix)
    if kt in ("reduction", "persistent_reduction") and hint == "DEFAULT":
        if tiling:
            r_score = tiling.get("r0_", 0)
            x_score = tiling.get("x", 1)
            if x_score > 0 and r_score / x_score >= 32:
                issues.append(f"DEFAULT hint but tiling scores suggest INNER (ratio={r_score/x_score:.0f})")

    # Check for very large reductions (might need split)
    if kt == "reduction":
        r_size = size_hints.get("r0_", 0)
        if isinstance(r_size, int) and r_size > 65536:
            issues.append(f"very large reduction r0_={r_size}")

    # Check for near-power-of-2 issues
    if kt in ("reduction", "persistent_reduction"):
        r_size = size_hints.get("r0_", 0)
        if isinstance(r_size, int) and r_size > 0:
            next_pow2 = 1
            while next_pow2 < r_size:
                next_pow2 *= 2
            waste = 1.0 - (r_size / next_pow2)
            if waste > 0.4 and r_size > 32:
                issues.append(f"near-power-of-2 waste: r0_={r_size}, next_pow2={next_pow2}, waste={waste:.0%}")

    return issues


def generate_report(inference_dir, training_dir=None, output_dir="/tmp/benchmark_analysis"):
    """Generate comprehensive analysis report."""
    os.makedirs(output_dir, exist_ok=True)

    all_kernels = []

    # Process inference
    print("Processing inference results...")
    inf_files = find_output_code_files(inference_dir)
    print(f"  Found {len(inf_files)} output_code.py files")
    for path in inf_files:
        model = get_model_name_from_path(path)
        parsed = parse_kernel_info(path)
        for k in parsed["kernels"]:
            k["model"] = model
            k["mode"] = "inference"
            k["output_code_path"] = path
            k["issues"] = analyze_kernel(k)
            all_kernels.append(k)

    # Process training
    if training_dir and os.path.exists(training_dir):
        print("Processing training results...")
        train_files = find_output_code_files(training_dir)
        print(f"  Found {len(train_files)} output_code.py files")
        for path in train_files:
            model = get_model_name_from_path(path)
            parsed = parse_kernel_info(path)
            for k in parsed["kernels"]:
                k["model"] = model
                k["mode"] = "training"
                k["output_code_path"] = path
                k["issues"] = analyze_kernel(k)
                all_kernels.append(k)

    print(f"\nTotal kernels analyzed: {len(all_kernels)}")

    # Statistics
    stats = {
        "total_kernels": len(all_kernels),
        "by_type": dict(Counter(k["meta"].get("kernel_type", "unknown") for k in all_kernels)),
        "by_hint": dict(Counter(k["meta"].get("reduction_hint", "none") for k in all_kernels)),
        "by_mode": dict(Counter(k["mode"] for k in all_kernels)),
        "by_model": dict(Counter(k["model"] for k in all_kernels).most_common()),
        "models_count": len(set(k["model"] for k in all_kernels)),
    }

    # Find kernels with issues
    kernels_with_issues = [k for k in all_kernels if k["issues"]]
    issue_counts = Counter()
    for k in kernels_with_issues:
        for issue in k["issues"]:
            # Categorize by issue type (first word before colon)
            category = issue.split(":")[0] if ":" in issue else issue.split("(")[0].strip()
            issue_counts[category] += 1

    stats["kernels_with_issues"] = len(kernels_with_issues)
    stats["issue_categories"] = dict(issue_counts.most_common())

    # Detailed issue report
    issue_report = []
    for k in sorted(kernels_with_issues, key=lambda x: len(x["issues"]), reverse=True):
        issue_report.append({
            "model": k["model"],
            "mode": k["mode"],
            "kernel_name": k["kernel_name"],
            "kernel_type": k["meta"].get("kernel_type"),
            "reduction_hint": k["meta"].get("reduction_hint"),
            "size_hints": k["meta"].get("size_hints"),
            "tiling_scores": k["meta"].get("tiling_scores"),
            "issues": k["issues"],
        })

    # Save results
    with open(os.path.join(output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    with open(os.path.join(output_dir, "issues.json"), "w") as f:
        json.dump(issue_report, f, indent=2)

    # Save full kernel index (without source code)
    kernel_index = []
    for k in all_kernels:
        kernel_index.append({
            "model": k["model"],
            "mode": k["mode"],
            "kernel_name": k["kernel_name"],
            "meta": k["meta"],
            "issues": k["issues"],
        })
    with open(os.path.join(output_dir, "kernel_index.json"), "w") as f:
        json.dump(kernel_index, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f" Benchmark Analysis Report")
    print(f"{'='*60}")
    print(f"\nKernel Statistics:")
    print(f"  Total kernels: {stats['total_kernels']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  By hint: {stats['by_hint']}")
    print(f"  By mode: {stats['by_mode']}")
    print(f"  Models: {stats['models_count']}")

    print(f"\nIssues Found:")
    print(f"  Kernels with issues: {stats['kernels_with_issues']}")
    for cat, count in issue_counts.most_common():
        print(f"    {cat}: {count}")

    if issue_report[:10]:
        print(f"\nTop Issues:")
        for r in issue_report[:10]:
            print(f"  [{r['model']}] {r['kernel_name'][:50]}")
            for issue in r["issues"]:
                print(f"    - {issue}")

    print(f"\nResults saved to: {output_dir}/")
    return stats, issue_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_dir", help="Directory with inference debug output")
    parser.add_argument("training_dir", nargs="?", help="Directory with training debug output")
    parser.add_argument("--output", "-o", default="/tmp/benchmark_analysis")
    args = parser.parse_args()

    generate_report(args.inference_dir, args.training_dir, args.output)


if __name__ == "__main__":
    main()
