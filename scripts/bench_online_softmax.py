"""
Training data collection for online_softmax vs 3-pass softmax heuristic.

For each (numel, rnumel, dtype, has_dropout) combination, benchmarks:
  - online_softmax ENABLED (default inductor behavior)
  - online_softmax DISABLED (falls back to 3-pass: amax -> sum_exp -> normalize)

Saves raw data suitable for training a decision heuristic.

Usage:
    python scripts/bench_online_softmax.py
    python scripts/bench_online_softmax.py --gpus 0,1 --output results.json
    python scripts/bench_online_softmax.py --quick  # smaller grid for testing
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# We run each measurement in a subprocess to ensure:
# 1. Clean inductor cache between online/3pass variants
# 2. No state leakage between measurements
# 3. Isolated GPU usage


def make_worker_script(
    numel: int,
    rnumel: int,
    dtype_str: str,
    has_dropout: bool,
    online_softmax_enabled: bool,
    gpu_idx: int,
    n_warmup: int = 25,
    n_rep: int = 100,
) -> str:
    """Generate a self-contained benchmark script for one configuration."""
    # Map dtype string to torch dtype
    dtype_map = {"f32": "torch.float32", "bf16": "torch.bfloat16", "f16": "torch.float16"}
    torch_dtype = dtype_map[dtype_str]

    return f'''
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

# Configure online_softmax
inductor_config.online_softmax = {online_softmax_enabled}
# Also set threshold to 0 to force 3-pass when disabled
if not {online_softmax_enabled}:
    inductor_config.online_softmax_rnumel_threshold = 0

from triton.testing import do_bench

NUMEL = {numel}
RNUMEL = {rnumel}
DTYPE = {torch_dtype}
HAS_DROPOUT = {has_dropout}
N_WARMUP = {n_warmup}
N_REP = {n_rep}

# Create input
x = torch.randn(NUMEL, RNUMEL, dtype=DTYPE, device="cuda")

if HAS_DROPOUT:
    def fn(x):
        s = torch.nn.functional.softmax(x.float(), dim=-1).to(x.dtype)
        return torch.nn.functional.dropout(s, p=0.1, training=True)
else:
    def fn(x):
        return torch.nn.functional.softmax(x.float(), dim=-1).to(x.dtype)

# Compile
torch._dynamo.reset()
compiled_fn = torch.compile(fn)

# Warmup (triggers compilation + autotuning)
with torch.no_grad():
    for _ in range(5):
        _ = compiled_fn(x)
    torch.cuda.synchronize()

# CUDAGraph capture
try:
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            _ = compiled_fn(x)
    torch.cuda.synchronize()

    # Benchmark the graph replay
    time_us = do_bench(lambda: g.replay(), warmup=N_WARMUP, rep=N_REP, return_mode="min") * 1000
except Exception as e:
    # Fallback if CUDAGraph capture fails (e.g., dropout with RNG)
    with torch.no_grad():
        time_us = do_bench(lambda: compiled_fn(x), warmup=N_WARMUP, rep=N_REP, return_mode="min") * 1000

# Count kernels via inductor metrics
import torch._inductor.metrics as metrics
n_kernels = metrics.generated_kernel_count

result = {{
    "time_us": time_us,
    "n_kernels": n_kernels,
}}
print(json.dumps(result))
'''


def run_single_benchmark(
    numel: int,
    rnumel: int,
    dtype_str: str,
    has_dropout: bool,
    online_softmax_enabled: bool,
    gpu_idx: int,
    n_warmup: int = 25,
    n_rep: int = 100,
    timeout: int = 120,
) -> dict | None:
    """Run a single benchmark in a subprocess and return results."""
    script = make_worker_script(
        numel, rnumel, dtype_str, has_dropout, online_softmax_enabled, gpu_idx, n_warmup, n_rep
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
    # Use a fresh inductor cache for each measurement
    cache_dir = tempfile.mkdtemp(prefix="bench_softmax_")
    env["TORCHINDUCTOR_CACHE_DIR"] = cache_dir

    try:
        result = subprocess.run(
            [sys.executable, "-u", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        if result.returncode != 0:
            return None

        # Parse the last line of stdout as JSON (skip any torch warnings)
        for line in reversed(result.stdout.strip().split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        return None
    finally:
        # Clean up cache dir
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)


def build_shape_grid(quick: bool = False) -> list[dict]:
    """Build the grid of (numel, rnumel, dtype, has_dropout) combinations."""
    if quick:
        rnumels = [1024, 4096, 16384, 65536, 131072]
        numels = [128, 1024, 8192]
        dtypes = ["bf16"]
        dropouts = [False]
    else:
        rnumels = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        numels = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        dtypes = ["f32", "bf16"]
        dropouts = [False, True]

    shapes = []
    for rnumel in rnumels:
        for numel in numels:
            for dtype in dtypes:
                for has_dropout in dropouts:
                    shapes.append({
                        "numel": numel,
                        "rnumel": rnumel,
                        "dtype": dtype,
                        "has_dropout": has_dropout,
                    })
    return shapes


def _run_pair_parallel(
    numel, rnumel, dtype_str, has_dropout,
    gpu_online, gpu_3pass, n_warmup, n_rep, timeout,
):
    """Run online and 3pass benchmarks in parallel on separate GPUs."""
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_online = executor.submit(
            run_single_benchmark,
            numel, rnumel, dtype_str, has_dropout,
            True, gpu_online, n_warmup, n_rep, timeout,
        )
        future_3pass = executor.submit(
            run_single_benchmark,
            numel, rnumel, dtype_str, has_dropout,
            False, gpu_3pass, n_warmup, n_rep, timeout,
        )
        result_online = future_online.result()
        result_3pass = future_3pass.result()

    return result_online, result_3pass


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark online_softmax vs 3-pass softmax across shapes"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/scratch_space/better_benchmark/online_softmax_heuristic_data.json"),
        help="Output JSON file path",
    )
    parser.add_argument("--gpus", default="0,1", help="Comma-separated GPU indices to use")
    parser.add_argument("--quick", action="store_true", help="Use smaller grid for testing")
    parser.add_argument("--n-warmup", type=int, default=25, help="Warmup iterations for do_bench")
    parser.add_argument("--n-rep", type=int, default=100, help="Repetitions for do_bench")
    parser.add_argument("--timeout", type=int, default=180, help="Per-measurement timeout (seconds)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output file")
    parser.add_argument("--parallel-pairs", action="store_true", default=True,
                        help="Run online/3pass on separate GPUs in parallel (default: True)")
    parser.add_argument("--no-parallel-pairs", dest="parallel_pairs", action="store_false",
                        help="Run online/3pass sequentially")
    args = parser.parse_args()

    gpu_indices = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    if not gpu_indices:
        gpu_indices = [0]

    # Need at least 2 GPUs for parallel pair mode
    parallel_pairs = args.parallel_pairs and len(gpu_indices) >= 2

    shapes = build_shape_grid(quick=args.quick)
    print(f"Online softmax heuristic benchmark")
    print(f"  Shapes to test: {len(shapes)}")
    print(f"  GPUs: {gpu_indices}")
    print(f"  Parallel pairs: {parallel_pairs}")
    print(f"  Output: {args.output}")
    print()

    # Load existing results if resuming
    existing_results = []
    existing_keys = set()
    if args.resume and args.output.exists():
        existing_results = json.loads(args.output.read_text())
        for r in existing_results:
            key = (r["numel"], r["rnumel"], r["dtype"], r["has_dropout"])
            existing_keys.add(key)
        print(f"  Resuming: {len(existing_results)} existing results loaded")

    results = list(existing_results)
    total = len(shapes)
    done = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    # Alternate GPUs for online vs 3pass to avoid thermal bias
    for i, shape in enumerate(shapes):
        key = (shape["numel"], shape["rnumel"], shape["dtype"], shape["has_dropout"])
        if key in existing_keys:
            skipped += 1
            continue

        numel = shape["numel"]
        rnumel = shape["rnumel"]
        dtype_str = shape["dtype"]
        has_dropout = shape["has_dropout"]

        # Alternate which GPU gets online vs 3pass to cancel thermal bias
        if (i % 2) == 0:
            gpu_online = gpu_indices[0]
            gpu_3pass = gpu_indices[1 % len(gpu_indices)]
        else:
            gpu_online = gpu_indices[1 % len(gpu_indices)]
            gpu_3pass = gpu_indices[0]

        if parallel_pairs:
            result_online, result_3pass = _run_pair_parallel(
                numel, rnumel, dtype_str, has_dropout,
                gpu_online, gpu_3pass,
                args.n_warmup, args.n_rep, args.timeout,
            )
        else:
            # Sequential: run online first, then 3pass
            result_online = run_single_benchmark(
                numel, rnumel, dtype_str, has_dropout,
                online_softmax_enabled=True,
                gpu_idx=gpu_online,
                n_warmup=args.n_warmup,
                n_rep=args.n_rep,
                timeout=args.timeout,
            )
            result_3pass = run_single_benchmark(
                numel, rnumel, dtype_str, has_dropout,
                online_softmax_enabled=False,
                gpu_idx=gpu_3pass,
                n_warmup=args.n_warmup,
                n_rep=args.n_rep,
                timeout=args.timeout,
            )

        if result_online is None or result_3pass is None:
            failed += 1
            status = "FAIL"
            if result_online is None:
                status += "(online)"
            if result_3pass is None:
                status += "(3pass)"
            print(
                f"  [{i+1}/{total}] {status}  numel={numel} rnumel={rnumel} "
                f"dtype={dtype_str} dropout={has_dropout}",
                flush=True,
            )
            continue

        time_online = result_online["time_us"]
        time_3pass = result_3pass["time_us"]
        online_wins = time_online < time_3pass
        speedup_3pass = time_online / time_3pass if time_3pass > 0 else 1.0

        entry = {
            "numel": numel,
            "rnumel": rnumel,
            "dtype": dtype_str,
            "has_dropout": has_dropout,
            "time_online_us": round(time_online, 2),
            "time_3pass_us": round(time_3pass, 2),
            "online_wins": online_wins,
            "speedup_of_3pass": round(speedup_3pass, 4),
            "kernels_online": result_online.get("n_kernels"),
            "kernels_3pass": result_3pass.get("n_kernels"),
        }
        results.append(entry)
        done += 1

        winner = "online" if online_wins else "3pass"
        delta = abs(1 - speedup_3pass) * 100
        print(
            f"  [{i+1}/{total}] numel={numel:>6} rnumel={rnumel:>6} "
            f"dtype={dtype_str:>3} drop={str(has_dropout):>5} | "
            f"online={time_online:>8.1f}us  3pass={time_3pass:>8.1f}us  "
            f"winner={winner}({delta:.1f}%)",
            flush=True,
        )

        # Incremental save every 10 results
        if done % 10 == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=2))

    # Final save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Done: {done} measured, {skipped} skipped (resume), {failed} failed")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed/max(done,1):.1f}s per shape)")
    print(f"Output: {args.output}")

    # Analysis
    if results:
        _analyze_results(results)


def _analyze_results(results: list[dict]):
    """Fit a simple decision boundary and print analysis."""
    print(f"\n{'='*70}")
    print("ANALYSIS: Online softmax vs 3-pass decision boundary")
    print(f"{'='*70}\n")

    # Overall stats
    online_wins_count = sum(1 for r in results if r["online_wins"])
    total = len(results)
    print(f"Overall: online wins {online_wins_count}/{total} ({online_wins_count*100//total}%)")

    # By rnumel
    print("\nBy rnumel (online win rate):")
    by_rnumel = {}
    for r in results:
        rn = r["rnumel"]
        if rn not in by_rnumel:
            by_rnumel[rn] = {"wins": 0, "total": 0, "speedups": []}
        by_rnumel[rn]["total"] += 1
        if r["online_wins"]:
            by_rnumel[rn]["wins"] += 1
        by_rnumel[rn]["speedups"].append(r["speedup_of_3pass"])

    for rn in sorted(by_rnumel.keys()):
        d = by_rnumel[rn]
        avg_speedup = sum(d["speedups"]) / len(d["speedups"])
        print(
            f"  rnumel={rn:>7}: online wins {d['wins']:>3}/{d['total']:>3} "
            f"({d['wins']*100//d['total']:>3}%)  "
            f"avg 3pass speedup={avg_speedup:.3f}x"
        )

    # By dtype
    print("\nBy dtype:")
    for dtype in ["f32", "bf16", "f16"]:
        subset = [r for r in results if r["dtype"] == dtype]
        if not subset:
            continue
        wins = sum(1 for r in subset if r["online_wins"])
        avg_speedup = sum(r["speedup_of_3pass"] for r in subset) / len(subset)
        print(f"  {dtype}: online wins {wins}/{len(subset)} ({wins*100//len(subset)}%), avg 3pass speedup={avg_speedup:.3f}x")

    # By dropout
    print("\nBy dropout:")
    for has_drop in [False, True]:
        subset = [r for r in results if r["has_dropout"] == has_drop]
        if not subset:
            continue
        wins = sum(1 for r in subset if r["online_wins"])
        avg_speedup = sum(r["speedup_of_3pass"] for r in subset) / len(subset)
        label = "with dropout" if has_drop else "no dropout"
        print(f"  {label}: online wins {wins}/{len(subset)} ({wins*100//len(subset)}%), avg 3pass speedup={avg_speedup:.3f}x")

    # By numel (parallelism)
    print("\nBy numel (parallelism):")
    by_numel = {}
    for r in results:
        n = r["numel"]
        if n not in by_numel:
            by_numel[n] = {"wins": 0, "total": 0, "speedups": []}
        by_numel[n]["total"] += 1
        if r["online_wins"]:
            by_numel[n]["wins"] += 1
        by_numel[n]["speedups"].append(r["speedup_of_3pass"])

    for n in sorted(by_numel.keys()):
        d = by_numel[n]
        avg_speedup = sum(d["speedups"]) / len(d["speedups"])
        print(
            f"  numel={n:>6}: online wins {d['wins']:>3}/{d['total']:>3} "
            f"({d['wins']*100//d['total']:>3}%)  "
            f"avg 3pass speedup={avg_speedup:.3f}x"
        )

    # Find threshold
    print("\n" + "-" * 70)
    print("DECISION BOUNDARY ESTIMATE:")
    print("-" * 70)

    # Simple threshold: find rnumel where online stops winning majority
    threshold_rnumel = None
    for rn in sorted(by_rnumel.keys()):
        d = by_rnumel[rn]
        win_rate = d["wins"] / d["total"]
        if win_rate < 0.5 and threshold_rnumel is None:
            threshold_rnumel = rn

    if threshold_rnumel is not None:
        print(f"  Simple threshold: use online_softmax when rnumel < {threshold_rnumel}")
    else:
        print("  No clear threshold found (online wins at all rnumel sizes)")

    # More nuanced: check if numel affects the threshold
    print("\n  Numel-dependent threshold (rnumel where online stops winning):")
    for n in sorted(by_numel.keys()):
        subset = [r for r in results if r["numel"] == n]
        by_rn = {}
        for r in subset:
            rn = r["rnumel"]
            if rn not in by_rn:
                by_rn[rn] = {"wins": 0, "total": 0}
            by_rn[rn]["total"] += 1
            if r["online_wins"]:
                by_rn[rn]["wins"] += 1
        thresh = "always online"
        for rn in sorted(by_rn.keys()):
            if by_rn[rn]["wins"] / by_rn[rn]["total"] < 0.5:
                thresh = f"< {rn}"
                break
        print(f"    numel={n:>6}: online wins when rnumel {thresh}")


if __name__ == "__main__":
    main()
