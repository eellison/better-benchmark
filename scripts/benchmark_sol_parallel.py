"""Benchmark all single-kernel repros for SOL gap, parallelized across GPUs.

Usage:
    python scripts/benchmark_sol_parallel.py repros/multi_kernel_inference/  # or any dir of repros
    python scripts/benchmark_sol_parallel.py --filelist files.txt
    python scripts/benchmark_sol_parallel.py repros/ --gpus 0,1,2,3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import as_completed
from pathlib import Path

WORKER_SCRIPT = '''
import sys, os, builtins, json, time
import torch
import torch._dynamo
import torch._inductor.config as cfg
import torch._inductor.metrics as m
import importlib.util

cfg.force_disable_caches = True
builtins.device = torch.device
builtins.inf = float("inf")
builtins.nan = float("nan")

os.environ["TORCH_LOGS"] = "+inductor_metrics"
torch._logging._init_logs()

path = sys.argv[1]
n_warmup = int(sys.argv[2])
n_iter = int(sys.argv[3])

spec = importlib.util.spec_from_file_location("region", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

model = mod.Repro().cuda()
inputs = mod.make_inputs()

def count_physical_bytes(inputs, outputs):
    seen = set()
    total = 0
    all_tensors = list(inputs)
    if isinstance(outputs, torch.Tensor):
        all_tensors.append(outputs)
    elif isinstance(outputs, (tuple, list)):
        all_tensors.extend(outputs)
    for t in all_tensors:
        if isinstance(t, torch.Tensor) and t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.storage().nbytes()
    return total

with torch.no_grad():
    eager_out = model(*inputs)
physical_bytes = count_physical_bytes(inputs, eager_out)

def bench_compiled(model, inputs, n_warmup, n_iter):
    with torch.no_grad():
        for _ in range(n_warmup):
            model(*inputs)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            model(*inputs)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1e6

# Default compile
m.reset()
compiled = torch.compile(model)
with torch.no_grad():
    compiled(*inputs)
kernel_count = m.generated_kernel_count + m.generated_cpp_vec_kernel_count
inductor_bytes = m.num_bytes_accessed
compiled_us = bench_compiled(compiled, inputs, n_warmup, n_iter)

total_bytes = inductor_bytes if inductor_bytes > 0 else physical_bytes

# Coordinate descent compile
cfg.coordinate_descent_tuning = True
torch._dynamo.reset()
compiled_cd = torch.compile(model)
with torch.no_grad():
    compiled_cd(*inputs)
cd_us = bench_compiled(compiled_cd, inputs, n_warmup, n_iter)

# CUDA Graph capture of compiled model
graph_us = None
try:
    cfg.coordinate_descent_tuning = False
    torch._dynamo.reset()
    compiled_graph = torch.compile(model)
    with torch.no_grad():
        compiled_graph(*inputs)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            with torch.no_grad():
                compiled_graph(*inputs)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        with torch.no_grad():
            compiled_graph(*inputs)

    for _ in range(n_warmup):
        g.replay()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        g.replay()
    torch.cuda.synchronize()
    graph_us = (time.perf_counter() - t0) / n_iter * 1e6
except Exception:
    pass

# Memcopy SOL: copy total_bytes (each element is read + written, so half the elements)
n_elems = max(total_bytes // 8, 1)
src = torch.empty(n_elems, dtype=torch.float32, device="cuda")
dst = torch.empty_like(src)
for _ in range(n_warmup):
    dst.copy_(src)
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(n_iter):
    dst.copy_(src)
torch.cuda.synchronize()
sol_us = (time.perf_counter() - t0) / n_iter * 1e6
sol_bw = total_bytes / (sol_us * 1e-6) / 1e9

result = {
    "kernels": kernel_count,
    "total_bytes": total_bytes,
    "inductor_bytes": inductor_bytes,
    "physical_bytes": physical_bytes,
    "compiled_us": round(compiled_us, 2),
    "cd_us": round(cd_us, 2),
    "graph_us": round(graph_us, 2) if graph_us else None,
    "sol_us": round(sol_us, 2),
    "gap": round(compiled_us / sol_us, 2) if sol_us > 0 else None,
    "cd_gap": round(cd_us / sol_us, 2) if sol_us > 0 else None,
    "graph_gap": round(graph_us / sol_us, 2) if graph_us and sol_us > 0 else None,
    "graph_speedup": round(compiled_us / graph_us, 2) if graph_us and graph_us > 0 else None,
    "cd_speedup": round(compiled_us / cd_us, 2) if cd_us > 0 else None,
    "sol_bw_gbps": round(sol_bw, 1),
}
print("RESULT:" + json.dumps(result))
'''


def run_one(path: str, gpu_id: int, n_warmup: int = 25, n_iter: int = 200) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONPATH"] = os.environ.get("PYTORCH_DIR", "/tmp/pytorch-work")
    try:
        r = subprocess.run(
            [sys.executable, "-c", WORKER_SCRIPT, path, str(n_warmup), str(n_iter)],
            capture_output=True, text=True, timeout=300, env=env,
        )
        for line in r.stdout.splitlines():
            if line.startswith("RESULT:"):
                result = json.loads(line[7:])
                result["file"] = path
                result["error"] = None
                return result
        return {"file": path, "error": (r.stderr or "no output")[-300:]}
    except subprocess.TimeoutExpired:
        return {"file": path, "error": "TIMEOUT"}
    except Exception as e:
        return {"file": path, "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Directory of repro .py files")
    parser.add_argument("--filelist", help="File with paths, one per line")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--output", default="sol_benchmark_results.json")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iter", type=int, default=200)
    args = parser.parse_args()

    # Collect files
    if args.filelist:
        with open(args.filelist) as f:
            files = [l.strip() for l in f if l.strip()]
    elif args.path:
        files = sorted(glob.glob(os.path.join(args.path, "*.py")))
    else:
        parser.error("Provide a directory or --filelist")

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True
            )
            gpu_ids = [int(x.strip()) for x in out.strip().split("\n")]
        except Exception:
            gpu_ids = [0]

    n_gpus = len(gpu_ids)
    print(f"Benchmarking {len(files)} repros across {n_gpus} GPUs: {gpu_ids}", file=sys.stderr)

    # ThreadPoolExecutor + subprocess: each thread grabs a GPU, runs subprocess.
    # Threads share the GPU assignment queue safely.
    import threading
    from concurrent.futures import ThreadPoolExecutor

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
        futures = {
            executor.submit(run_with_gpu, f): f
            for f in files
        }
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            name = os.path.basename(result["file"])
            if result.get("error"):
                print(f"[{i+1}/{len(files)}] FAIL {name}: {result['error'][:80]}", file=sys.stderr)
            else:
                gap = result.get("gap", "?")
                cd_gap = result.get("cd_gap", "?")
                g_gap = result.get("graph_gap", "-")
                us = result.get("compiled_us", "?")
                print(f"[{i+1}/{len(files)}] {name}: {us}us, {gap}x SOL (cd: {cd_gap}x, graph: {g_gap}x), k={result.get('kernels')}", file=sys.stderr)

    elapsed = time.time() - start

    # Sort by gap (worst first)
    ok = [r for r in results if r.get("gap") is not None]
    ok.sort(key=lambda r: -r["gap"])
    errs = [r for r in results if r.get("error")]

    with open(args.output, "w") as f:
        json.dump({"results": ok + errs, "elapsed_s": round(elapsed, 1)}, f, indent=2)

    # Print summary
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"Completed {len(ok)} benchmarks, {len(errs)} errors in {elapsed:.0f}s", file=sys.stderr)
    if ok:
        gaps = [r["gap"] for r in ok]
        cd_gaps = [r["cd_gap"] for r in ok if r.get("cd_gap")]
        g_gaps = [r["graph_gap"] for r in ok if r.get("graph_gap")]
        print(f"Default  SOL gap: median={sorted(gaps)[len(gaps)//2]:.2f}x, "
              f"mean={sum(gaps)/len(gaps):.2f}x, "
              f"max={max(gaps):.2f}x, min={min(gaps):.2f}x", file=sys.stderr)
        if cd_gaps:
            print(f"CoordDes SOL gap: median={sorted(cd_gaps)[len(cd_gaps)//2]:.2f}x, "
                  f"mean={sum(cd_gaps)/len(cd_gaps):.2f}x, "
                  f"max={max(cd_gaps):.2f}x, min={min(cd_gaps):.2f}x", file=sys.stderr)
            cd_wins = sum(1 for r in ok if r.get("cd_speedup", 1) > 1.05)
            print(f"CoordDes wins (>5% faster): {cd_wins}/{len(ok)}", file=sys.stderr)
        if g_gaps:
            print(f"CUDAGraph SOL gap: median={sorted(g_gaps)[len(g_gaps)//2]:.2f}x, "
                  f"mean={sum(g_gaps)/len(g_gaps):.2f}x, "
                  f"max={max(g_gaps):.2f}x, min={min(g_gaps):.2f}x", file=sys.stderr)
            g_wins = sum(1 for r in ok if r.get("graph_speedup") and r["graph_speedup"] > 1.05)
            print(f"CUDAGraph wins (>5% faster): {g_wins}/{len(ok)}", file=sys.stderr)
        print(f"\nTop 20 worst SOL gaps (default | coord descent | cuda graph):", file=sys.stderr)
        for r in ok[:20]:
            name = os.path.basename(r["file"])
            kb = r["total_bytes"] / 1024
            cd = r.get("cd_gap", "?")
            gg = r.get("graph_gap", "-")
            gg_str = f"{gg:5.2f}" if isinstance(gg, (int, float)) else f"{gg:>5s}"
            print(f"  {r['gap']:5.2f}x | {cd:5.2f}x | {gg_str}x  {r['compiled_us']:8.1f}us  {kb:8.1f}KB  k={r['kernels']}  {name}", file=sys.stderr)

    print(f"\nResults written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
