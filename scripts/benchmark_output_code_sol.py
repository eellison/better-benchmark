"""Benchmark output_code.py files from TORCH_COMPILE_DEBUG for SOL gap analysis.

Each output_code.py is a standalone script with get_args() and call(args).
We benchmark: actual runtime vs memory-bandwidth SOL.

Usage:
    python benchmark_output_code_sol.py /tmp/benchmark_traces/huggingface/ --gpus 0,1
"""
import argparse
import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


WORKER_SCRIPT = r'''
import sys, os, time, json

path = sys.argv[1]
n_warmup = int(sys.argv[2])
n_iter = int(sys.argv[3])

import torch
from torch._dynamo.testing import rand_strided

# Load the output_code module
import importlib.util
spec = importlib.util.spec_from_file_location("output_code", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Get args and the call function
args = mod.get_args()
call_fn = mod.call

# Warmup
for _ in range(n_warmup):
    call_fn(list(args))
torch.cuda.synchronize()

# Benchmark
times = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    call_fn(list(args))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1e6)  # microseconds

median_us = sorted(times)[len(times) // 2]

# Count bytes accessed (sum of all input tensor storages)
seen_ptrs = set()
total_bytes = 0
for a in args:
    if isinstance(a, torch.Tensor) and a.data_ptr() not in seen_ptrs:
        seen_ptrs.add(a.data_ptr())
        total_bytes += a.storage().nbytes()

# SOL: pure memory copy of the same bytes
n_elems = max(total_bytes // 4, 1)  # float32 elements
src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
dst = torch.empty_like(src)
for _ in range(n_warmup):
    dst.copy_(src)
torch.cuda.synchronize()
sol_times = []
for _ in range(n_iter):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dst.copy_(src)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    sol_times.append((t1 - t0) * 1e6)

sol_us = sorted(sol_times)[len(sol_times) // 2]
sol_bw = total_bytes / (sol_us * 1e-6) / 1e9 if sol_us > 0 else 0

# Count kernels in the file (approximate from source)
with open(path) as f:
    src_text = f.read()
kernel_count = src_text.count("async_compile.triton(")

result = {
    "median_us": round(median_us, 2),
    "sol_us": round(sol_us, 2),
    "gap": round(median_us / sol_us, 3) if sol_us > 0 else None,
    "total_bytes_mb": round(total_bytes / 1e6, 2),
    "sol_bw_gbps": round(sol_bw, 1),
    "kernel_count": kernel_count,
}
print("RESULT:" + json.dumps(result))
'''


def get_model_from_path(path):
    parts = Path(path).parts
    for i, p in enumerate(parts):
        if p == "torch_compile_debug" and i > 0:
            return parts[i - 1]
    return "unknown"


def run_one(path, gpu_id, n_warmup=5, n_iter=20):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        r = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, path, str(n_warmup), str(n_iter)],
            capture_output=True, text=True, timeout=120, env=env,
        )
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                result = json.loads(line[7:])
                result["file"] = path
                result["model"] = get_model_from_path(path)
                result["error"] = None
                return result
        return {"file": path, "model": get_model_from_path(path), "error": (r.stderr or "no output")[-500:]}
    except subprocess.TimeoutExpired:
        return {"file": path, "model": get_model_from_path(path), "error": "TIMEOUT"}
    except Exception as e:
        return {"file": path, "model": get_model_from_path(path), "error": str(e)}


def find_output_code_files(base_dir):
    results = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == "output_code.py":
                results.append(os.path.join(root, f))
    return sorted(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+", help="Directories with TORCH_COMPILE_DEBUG output")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs")
    parser.add_argument("--output", "-o", default="/tmp/benchmark_analysis/sol_results.json")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iter", type=int, default=20)
    args = parser.parse_args()

    # Collect files
    files = []
    for d in args.dirs:
        files.extend(find_output_code_files(d))
    print(f"Found {len(files)} output_code.py files to benchmark", file=sys.stderr)

    if not files:
        print("No files found!", file=sys.stderr)
        return

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            gpu_ids = [int(x.strip()) for x in out.strip().split("\n")]
        except Exception:
            gpu_ids = [0]

    n_gpus = len(gpu_ids)
    print(f"Using {n_gpus} GPUs: {gpu_ids}", file=sys.stderr)

    gpu_lock = threading.Lock()
    gpu_available = list(gpu_ids)

    def run_with_gpu(path):
        with gpu_lock:
            gpu = gpu_available.pop(0)
        try:
            return run_one(path, gpu, args.warmup, args.iter)
        finally:
            with gpu_lock:
                gpu_available.append(gpu)

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=n_gpus) as executor:
        futures = {executor.submit(run_with_gpu, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if result.get("error"):
                print(f"[{i+1}/{len(files)}] FAIL {result['model']}: {result['error'][:80]}", file=sys.stderr)
            else:
                print(f"[{i+1}/{len(files)}] {result['model']}: {result['median_us']:.0f}us, "
                      f"{result['gap']:.2f}x SOL, {result['kernel_count']} kernels, "
                      f"{result['total_bytes_mb']:.1f}MB", file=sys.stderr)

    elapsed = time.time() - start

    # Separate OK and errors
    ok = [r for r in results if r.get("gap") is not None]
    ok.sort(key=lambda r: -r["gap"])
    errs = [r for r in results if r.get("error")]

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output = {
        "elapsed_s": round(elapsed, 1),
        "total_files": len(files),
        "successes": len(ok),
        "errors": len(errs),
        "results": ok + errs,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Completed {len(ok)} benchmarks, {len(errs)} errors in {elapsed:.0f}s", file=sys.stderr)
    if ok:
        gaps = [r["gap"] for r in ok]
        print(f"SOL gap: median={sorted(gaps)[len(gaps)//2]:.2f}x, "
              f"mean={sum(gaps)/len(gaps):.2f}x, "
              f"max={max(gaps):.2f}x, min={min(gaps):.2f}x", file=sys.stderr)

        print(f"\nTop 20 worst SOL gaps:", file=sys.stderr)
        for r in ok[:20]:
            print(f"  {r['gap']:6.2f}x  {r['median_us']:8.0f}us  {r['total_bytes_mb']:6.1f}MB  "
                  f"k={r['kernel_count']:2d}  {r['model']}", file=sys.stderr)

        # By model average
        from collections import defaultdict
        model_gaps = defaultdict(list)
        for r in ok:
            model_gaps[r["model"]].append(r["gap"])
        print(f"\nBy model (avg gap):", file=sys.stderr)
        model_avgs = [(m, sum(gs)/len(gs), len(gs)) for m, gs in model_gaps.items()]
        model_avgs.sort(key=lambda x: -x[1])
        for m, avg, n in model_avgs[:15]:
            print(f"  {avg:5.2f}x  {m} (n={n})", file=sys.stderr)

    print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
