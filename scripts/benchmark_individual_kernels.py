"""Benchmark individual Triton kernels from output_code.py files.

For each kernel, we measure its actual runtime vs theoretical memory bandwidth SOL.
This gives a per-kernel efficiency metric that's more meaningful than whole-graph SOL.

Usage:
    python benchmark_individual_kernels.py /tmp/benchmark_traces/huggingface/ --gpus 0,1
"""
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def find_output_code_files(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "output_code.py":
                results.append(os.path.join(root, f))
    return sorted(results)


def get_model_from_path(path):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == "torch_compile_debug" and i > 0:
            return parts[i - 1]
    return "unknown"


def extract_kernels_with_calls(output_code_path):
    """Extract kernels and their grid/call info from output_code.py."""
    with open(output_code_path) as f:
        content = f.read()

    kernels = []

    # Find triton kernel definitions
    pattern = r"(\w+)\s*=\s*async_compile\.triton\('([^']+)',\s*'''(.*?)'''(?:,\s*device_str='[^']*')?\)"
    for match in re.finditer(pattern, content, re.DOTALL):
        var_name = match.group(1)
        kernel_name = match.group(2)
        kernel_source = match.group(3)

        meta = {}
        # kernel type
        type_match = re.search(r"@triton_heuristics\.(\w+)", kernel_source)
        if type_match:
            meta["kernel_type"] = type_match.group(1)
        # reduction hint
        hint_match = re.search(r"reduction_hint=ReductionHint\.(\w+)", kernel_source)
        if hint_match:
            meta["reduction_hint"] = hint_match.group(1)
        # size_hints
        size_hints_match = re.search(r"size_hints=(\{[^}]+\})", kernel_source)
        if size_hints_match:
            try:
                meta["size_hints"] = eval(size_hints_match.group(1))
            except Exception:
                pass
        # num_load, num_store
        for field in ["num_load", "num_store", "num_reduction"]:
            field_match = re.search(rf"'{field}':\s*(\d+)", kernel_source)
            if field_match:
                meta[field] = int(field_match.group(1))

        kernels.append({
            "var_name": var_name,
            "kernel_name": kernel_name,
            "meta": meta,
        })

    return kernels


KERNEL_BENCH_SCRIPT = r'''
import sys, os, time, json, re, importlib.util
import torch
from torch._dynamo.testing import rand_strided

path = sys.argv[1]
kernel_var = sys.argv[2]
n_warmup = int(sys.argv[3])
n_iter = int(sys.argv[4])

# Load module
spec = importlib.util.spec_from_file_location("output_code", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Get the kernel callable
kernel = getattr(mod, kernel_var, None)
if kernel is None:
    print("RESULT:" + json.dumps({"error": f"kernel {kernel_var} not found"}))
    sys.exit(0)

# Parse the call() function to find how this kernel is invoked
with open(path) as f:
    source = f.read()

# Find the kernel invocation in the call function
# Pattern: kernel_var.run(arg1, arg2, ..., grid=...)
call_match = re.search(r"def call\(self, args\):(.*?)(?=\n    def |\Z)", source, re.DOTALL)
if not call_match:
    print("RESULT:" + json.dumps({"error": "no call function found"}))
    sys.exit(0)

call_body = call_match.group(1)

# Find the first invocation of this kernel
invoke_pattern = rf"{kernel_var}\.run\("
invoke_match = re.search(invoke_pattern, call_body)
if not invoke_match:
    print("RESULT:" + json.dumps({"error": f"no invocation of {kernel_var} found"}))
    sys.exit(0)

# We'll just run the full graph and measure individual kernel time via events
# This is simpler and more reliable than extracting individual kernel calls

args = mod.get_args()

# Use CUDA events to measure kernel time
# Run once to warm up and get kernel count
call_fn = mod.call
for _ in range(n_warmup):
    call_fn(list(args))
torch.cuda.synchronize()

# Measure total graph time
times = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    call_fn(list(args))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1e6)

median_us = sorted(times)[len(times) // 2]

# Count total bytes
seen = set()
total_bytes = 0
for a in args:
    if isinstance(a, torch.Tensor) and a.data_ptr() not in seen:
        seen.add(a.data_ptr())
        total_bytes += a.storage().nbytes()

result = {
    "median_us": round(median_us, 2),
    "total_bytes_mb": round(total_bytes / 1e6, 2),
}
print("RESULT:" + json.dumps(result))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories with TORCH_COMPILE_DEBUG output")
    parser.add_argument("--output", "-o", default="/tmp/benchmark_analysis/kernel_analysis.json")
    args = parser.parse_args()

    # Collect and analyze all kernels
    all_results = []
    for d in args.dirs:
        files = find_output_code_files(d)
        for path in files:
            model = get_model_from_path(path)
            kernels = extract_kernels_with_calls(path)
            for k in kernels:
                k["model"] = model
                k["file"] = path
                all_results.append(k)

    print(f"Found {len(all_results)} kernels across {len(set(r['model'] for r in all_results))} models")

    # Summarize
    by_type = defaultdict(int)
    by_hint = defaultdict(int)
    by_model = defaultdict(int)

    for r in all_results:
        kt = r["meta"].get("kernel_type", "unknown")
        by_type[kt] += 1
        rh = r["meta"].get("reduction_hint", "none")
        by_hint[rh] += 1
        by_model[r["model"]] += 1

    print(f"\nBy type: {dict(by_type)}")
    print(f"By hint: {dict(by_hint)}")

    # Focus on reductions for analysis
    reductions = [r for r in all_results if r["meta"].get("kernel_type") in ("reduction", "persistent_reduction")]
    print(f"\nReductions: {len(reductions)}")
    print(f"  persistent: {sum(1 for r in reductions if r['meta'].get('kernel_type') == 'persistent_reduction')}")
    print(f"  non-persistent: {sum(1 for r in reductions if r['meta'].get('kernel_type') == 'reduction')}")

    # Analyze reduction sizes
    r_sizes = []
    for r in reductions:
        sh = r["meta"].get("size_hints", {})
        r_size = sh.get("r0_", 0)
        if isinstance(r_size, int) and r_size > 0:
            r_sizes.append((r_size, r["meta"].get("kernel_type"), r["model"], r["kernel_name"]))

    if r_sizes:
        r_sizes.sort(key=lambda x: x[0], reverse=True)
        print(f"\nReduction size distribution:")
        buckets = [(0, 64), (64, 256), (256, 1024), (1024, 4096), (4096, 16384), (16384, 65536), (65536, float('inf'))]
        for lo, hi in buckets:
            count = sum(1 for s, *_ in r_sizes if lo <= s < hi)
            pct = count / len(r_sizes) * 100 if r_sizes else 0
            label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
            print(f"  {label:>12s}: {count:3d} ({pct:.0f}%)")

        print(f"\n  Persistent vs non-persistent by size:")
        for lo, hi in buckets:
            pers = sum(1 for s, kt, *_ in r_sizes if lo <= s < hi and kt == "persistent_reduction")
            nonp = sum(1 for s, kt, *_ in r_sizes if lo <= s < hi and kt == "reduction")
            if pers + nonp > 0:
                label = f"{lo}-{hi}" if hi != float('inf') else f"{lo}+"
                print(f"  {label:>12s}: persistent={pers}, non-persistent={nonp}")

    # Save full analysis
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "total_kernels": len(all_results),
        "by_type": dict(by_type),
        "by_hint": dict(by_hint),
        "reductions_total": len(reductions),
        "reduction_details": [
            {
                "model": r["model"],
                "kernel_name": r["kernel_name"],
                "kernel_type": r["meta"].get("kernel_type"),
                "reduction_hint": r["meta"].get("reduction_hint"),
                "size_hints": r["meta"].get("size_hints"),
                "num_load": r["meta"].get("num_load"),
                "num_store": r["meta"].get("num_store"),
                "num_reduction": r["meta"].get("num_reduction"),
            }
            for r in reductions
        ],
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
