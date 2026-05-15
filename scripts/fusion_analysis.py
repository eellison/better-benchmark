"""Analyze fusion patterns across the benchmark suite.

Looks at:
1. What operations get fused together
2. Which reductions are split vs fused
3. Potential missed fusion opportunities
4. Kernel count per model (more kernels = more launch overhead)

Usage:
    python fusion_analysis.py /tmp/benchmark_traces/huggingface/ /tmp/benchmark_traces_training/huggingface/
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


def get_model_from_path(path):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == "torch_compile_debug" and i > 0:
            return parts[i - 1]
    return "unknown"


def find_output_code_files(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "output_code.py":
                results.append(os.path.join(root, f))
    return sorted(results)


def analyze_output_code(path):
    """Analyze a single output_code.py for fusion patterns."""
    with open(path) as f:
        content = f.read()

    model = get_model_from_path(path)
    result = {
        "model": model,
        "path": path,
        "triton_kernels": 0,
        "extern_kernels": 0,
        "kernel_types": Counter(),
        "fused_ops": [],
        "reduction_sizes": [],
        "extern_ops": [],
    }

    # Count triton kernels
    triton_matches = re.findall(r"async_compile\.triton\('([^']+)'", content)
    result["triton_kernels"] = len(triton_matches)

    for name in triton_matches:
        # Classify
        if "_red_" in name or "triton_red_" in name:
            result["kernel_types"]["reduction"] += 1
        elif "_per_" in name or "triton_per_" in name:
            result["kernel_types"]["persistent_reduction"] += 1
        elif "_poi_" in name or "triton_poi_" in name:
            result["kernel_types"]["pointwise"] += 1
        else:
            result["kernel_types"]["other"] += 1

        # Extract fused ops
        fused_part = re.sub(r"^triton_(red|per|poi|fused)_fused_", "", name)
        fused_part = re.sub(r"_\d+$", "", fused_part)
        result["fused_ops"].append(fused_part)

    # Count extern kernels (mm, bmm, addmm, conv, etc)
    extern_calls = re.findall(r"extern_kernels\.(\w+)\(", content)
    result["extern_kernels"] = len(extern_calls)
    result["extern_ops"] = extern_calls

    # Get reduction sizes from size_hints
    for m in re.finditer(r"size_hints=\{([^}]+)\}", content):
        hints_str = m.group(1)
        r_match = re.search(r"'r0_':\s*(\d+)", hints_str)
        if r_match:
            result["reduction_sizes"].append(int(r_match.group(1)))

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories with TORCH_COMPILE_DEBUG output")
    parser.add_argument("--output", "-o", default="/tmp/benchmark_analysis/fusion_report.json")
    args = parser.parse_args()

    all_analyses = []
    for d in args.dirs:
        mode = "training" if "training" in d else "inference"
        files = find_output_code_files(d)
        for path in files:
            analysis = analyze_output_code(path)
            analysis["mode"] = mode
            all_analyses.append(analysis)

    # Aggregate stats
    total_triton = sum(a["triton_kernels"] for a in all_analyses)
    total_extern = sum(a["extern_kernels"] for a in all_analyses)

    print(f"{'='*70}")
    print(f"  Fusion Analysis Report")
    print(f"{'='*70}")
    print(f"\nTotal compilation units: {len(all_analyses)}")
    print(f"Total Triton kernels: {total_triton}")
    print(f"Total extern kernels: {total_extern}")
    print(f"Ratio (triton / extern): {total_triton / total_extern:.2f}" if total_extern > 0 else "")

    # Per-model summary
    print(f"\n{'='*70}")
    print(f"  Per-Model Kernel Counts (inference)")
    print(f"{'='*70}")
    model_stats = defaultdict(lambda: {"triton": 0, "extern": 0, "graphs": 0})
    for a in all_analyses:
        if a["mode"] == "inference":
            model_stats[a["model"]]["triton"] += a["triton_kernels"]
            model_stats[a["model"]]["extern"] += a["extern_kernels"]
            model_stats[a["model"]]["graphs"] += 1

    print(f"{'Model':<40s} {'Triton':>7s} {'Extern':>7s} {'Total':>7s} {'Graphs':>6s}")
    print(f"{'-'*40} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for model in sorted(model_stats.keys()):
        s = model_stats[model]
        total = s["triton"] + s["extern"]
        print(f"{model:<40s} {s['triton']:7d} {s['extern']:7d} {total:7d} {s['graphs']:6d}")

    # Extern kernel breakdown
    extern_counter = Counter()
    for a in all_analyses:
        extern_counter.update(a["extern_ops"])

    print(f"\n{'='*70}")
    print(f"  Extern Kernel Types (not fused into Triton)")
    print(f"{'='*70}")
    for op, count in extern_counter.most_common(20):
        print(f"  {count:4d}x  {op}")

    # Reduction size analysis
    print(f"\n{'='*70}")
    print(f"  Reduction Size Distribution")
    print(f"{'='*70}")
    all_r_sizes = []
    for a in all_analyses:
        all_r_sizes.extend(a["reduction_sizes"])

    if all_r_sizes:
        buckets = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, float('inf'))]
        for lo, hi in buckets:
            count = sum(1 for s in all_r_sizes if lo <= s < hi)
            pct = count / len(all_r_sizes) * 100
            label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
            print(f"  {label:>12s}: {count:4d} ({pct:4.1f}%)")

    # Fusion pattern frequency
    print(f"\n{'='*70}")
    print(f"  Most Common Fusion Patterns")
    print(f"{'='*70}")
    fused_ops_counter = Counter()
    for a in all_analyses:
        fused_ops_counter.update(a["fused_ops"])

    for pattern, count in fused_ops_counter.most_common(30):
        print(f"  {count:4d}x  {pattern[:80]}")

    # Identify models with high extern/triton ratio (potential fusion opportunities)
    print(f"\n{'='*70}")
    print(f"  Models with High Extern/Triton Ratio (inference)")
    print(f"  (high ratio = more ops not fused into Triton)")
    print(f"{'='*70}")
    ratios = []
    for model, s in model_stats.items():
        if s["triton"] > 0:
            ratio = s["extern"] / s["triton"]
            ratios.append((model, ratio, s["extern"], s["triton"]))
    ratios.sort(key=lambda x: -x[1])
    for model, ratio, ext, tri in ratios[:15]:
        print(f"  {ratio:5.2f}  {model} (extern={ext}, triton={tri})")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    report = {
        "total_compilation_units": len(all_analyses),
        "total_triton_kernels": total_triton,
        "total_extern_kernels": total_extern,
        "extern_ops": dict(extern_counter.most_common()),
        "fusion_patterns": dict(fused_ops_counter.most_common(50)),
        "model_stats": {
            model: dict(s) for model, s in model_stats.items()
        },
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
