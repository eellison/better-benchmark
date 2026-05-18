"""
Parallel benchmark runner — distributes repros across all available GPUs.

Each GPU gets a persistent worker process that holds the lock for the entire
batch, avoiding per-repro torch startup overhead.

Usage:
    python scripts/bench_parallel.py repros/canonical/
    python scripts/bench_parallel.py repros/canonical/ --device-kind B200 --update-perf
    python scripts/bench_parallel.py repro1.py repro2.py repro3.py
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_lock import gpu_lock_for_kind, discover_gpus, matching_gpus


def find_repros(paths: list[Path]) -> list[Path]:
    """Resolve paths to individual repro.py files."""
    repros = []
    for p in paths:
        if p.is_file() and p.name.endswith(".py"):
            repros.append(p)
        elif p.is_dir():
            repros.extend(sorted(p.rglob("repro.py")))
    return repros


def worker(gpu_idx: str, task_queue: mp.Queue, result_queue: mp.Queue,
           args_dict: dict):
    """Worker process: holds GPU lock, benchmarks repros until queue is empty."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

    import torch
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from triton.testing import do_bench
    import importlib.util
    import math

    def load_and_bench(repro_path: str, all_shapes: bool, no_cd: bool,
                       n_warmup: int, n_rep: int) -> dict:
        spec = importlib.util.spec_from_file_location("repro", repro_path)
        mod = importlib.util.module_from_spec(spec)
        mod.device = torch.device
        mod.inf = math.inf
        mod.nan = math.nan
        spec.loader.exec_module(mod)

        repro_cls = mod.Repro
        make_inputs_fn = getattr(mod, "make_inputs", mod.make_inputs)

        # Determine shapes to run
        if all_shapes and hasattr(mod, "load_shape_configs"):
            configs = mod.load_shape_configs(repro_path)
            if configs:
                shape_names = list(configs.keys())
            else:
                shape_names = [None]
        else:
            shape_names = [None]

        results = {}
        for name in shape_names:
            label = name if name is not None else "default"
            if name is not None:
                from repro_harness import make_inputs_from_config, load_shape_configs
                configs = load_shape_configs(repro_path)
                inputs = make_inputs_from_config(configs[name])
            else:
                inputs = make_inputs_fn()

            instance = repro_cls()
            with torch.no_grad():
                eager_out = instance(*inputs)

            # Count bytes
            total_bytes = 0
            for t in inputs:
                if isinstance(t, torch.Tensor):
                    total_bytes += t.nelement() * t.element_size()
            if isinstance(eager_out, torch.Tensor):
                total_bytes += eager_out.nelement() * eager_out.element_size()
            elif isinstance(eager_out, (tuple, list)):
                for o in eager_out:
                    if isinstance(o, torch.Tensor):
                        total_bytes += o.nelement() * o.element_size()

            # SOL
            copy_elems = max(total_bytes // (2 * 4), 256)
            src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
            dst = torch.empty_like(src)
            sol_us = do_bench(lambda: dst.copy_(src), warmup=n_warmup, rep=n_rep) * 1000
            del src, dst

            # Compiled
            torch._dynamo.reset()
            compiled = torch.compile(instance)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()
            compiled_us = do_bench(lambda: compiled(*inputs), warmup=n_warmup, rep=n_rep) * 1000

            # Coord descent
            cd_us = None
            if not no_cd:
                inductor_config.coordinate_descent_tuning = True
                torch._dynamo.reset()
                compiled_cd = torch.compile(instance)
                with torch.no_grad():
                    for _ in range(3):
                        compiled_cd(*inputs)
                    torch.cuda.synchronize()
                cd_us = do_bench(lambda: compiled_cd(*inputs), warmup=n_warmup, rep=n_rep) * 1000
                inductor_config.coordinate_descent_tuning = False

            results[label] = {
                "compiled_us": compiled_us,
                "coord_descent_us": cd_us,
                "memcopy_sol_us": sol_us,
                "total_bytes": total_bytes,
                "gap_default": compiled_us / sol_us if sol_us > 0 else None,
                "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
            }

        return results

    # Main worker loop
    sys.path.insert(0, str(Path(args_dict["root"])))
    while True:
        try:
            repro_path = task_queue.get_nowait()
        except Exception:
            break

        start = time.time()
        try:
            results = load_and_bench(
                str(repro_path),
                all_shapes=args_dict["all_shapes"],
                no_cd=args_dict["no_cd"],
                n_warmup=args_dict["n_warmup"],
                n_rep=args_dict["n_rep"],
            )
            elapsed = time.time() - start
            result_queue.put({
                "repro": str(repro_path),
                "gpu": gpu_idx,
                "status": "ok",
                "elapsed": elapsed,
                "results": results,
            })
        except Exception as e:
            elapsed = time.time() - start
            result_queue.put({
                "repro": str(repro_path),
                "gpu": gpu_idx,
                "status": "failed",
                "elapsed": elapsed,
                "error": str(e)[:200],
            })


def main():
    parser = argparse.ArgumentParser(description="Parallel GPU benchmark runner")
    parser.add_argument("paths", nargs="*", type=Path,
                        help="repro.py files or directories to benchmark")
    parser.add_argument("--benchmark-set", type=Path, default=None,
                        help="Path to a frozen benchmark set JSON (e.g. benchmarks/v1.json)")
    parser.add_argument("--device-kind", default=None,
                        help="GPU kind to use (e.g. H100, B200). Default: all GPUs")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: one per matching GPU)")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark all shapes from shapes.txt")
    parser.add_argument("--no-cd", action="store_true",
                        help="Skip coordinate descent tuning")
    parser.add_argument("--update-perf", action="store_true",
                        help="Write results to each repro's perf.json")
    parser.add_argument("--hardware", default=None,
                        help="Hardware label for perf.json (auto-detected if not set)")
    parser.add_argument("--n-warmup", type=int, default=25)
    parser.add_argument("--n-rep", type=int, default=200)
    parser.add_argument("--tag", default=None,
                        help="Tag for this run (e.g. 'baseline', 'my_fix'). Used to key results in perf.json for comparison.")
    parser.add_argument("--compare", type=str, nargs=2, metavar=("TAG_A", "TAG_B"),
                        help="Compare two tagged runs from perf.json (no benchmarking, just report)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write all results to a JSON file")
    args = parser.parse_args()

    # Compare mode: just read existing perf.json and diff
    if args.compare:
        paths = args.paths or [Path("repros/canonical")]
        _run_compare(paths, args.compare[0], args.compare[1])
        return

    # Load benchmark set if specified
    benchmark_entries = None
    if args.benchmark_set:
        bset = json.loads(args.benchmark_set.read_text())
        benchmark_entries = bset.get("benchmarks", [])
        canonical_dir = Path("repros/canonical")
        repros = []
        for entry in benchmark_entries:
            repro_path = canonical_dir / entry["repro"] / "repro.py"
            if repro_path.exists():
                repros.append(repro_path)
        # Deduplicate (same repro appears multiple times for different shapes)
        repros = sorted(set(repros))
        # Pass shape info to workers via the all_shapes mechanism
        args.all_shapes = True
        print(f"Benchmark set: {args.benchmark_set.name} "
              f"({len(benchmark_entries)} points, {len(repros)} unique repros)")
    elif args.paths:
        repros = find_repros(args.paths)
    else:
        repros = find_repros([Path("repros/canonical")])

    if not repros:
        print("No repro.py files found.")
        return

    gpus = matching_gpus(args.device_kind)
    n_workers = min(args.max_workers or len(gpus), len(gpus), len(repros))

    print(f"Benchmarking {len(repros)} repros across {n_workers} GPUs")
    gpu_labels = [f"{g['index']}:{g['kind']}" for g in gpus[:n_workers]]
    print(f"  GPUs: {', '.join(gpu_labels)}")
    print()

    # Fill task queue
    task_queue = mp.Queue()
    for r in repros:
        task_queue.put(r)

    result_queue = mp.Queue()
    root = str(Path(__file__).resolve().parents[1])

    args_dict = {
        "root": root,
        "all_shapes": args.all_shapes,
        "no_cd": args.no_cd,
        "n_warmup": args.n_warmup,
        "n_rep": args.n_rep,
    }

    # Spawn workers — each acquires its own GPU lock
    workers = []
    for i in range(n_workers):
        gpu = gpus[i]
        p = mp.Process(
            target=_locked_worker,
            args=(gpu, task_queue, result_queue, args_dict),
        )
        p.start()
        workers.append(p)

    # Collect results
    all_results = {}
    done = 0
    failed = 0
    start_time = time.time()

    while done + failed < len(repros):
        try:
            result = result_queue.get(timeout=600)
        except Exception:
            print("Timed out waiting for results")
            break

        repro_name = Path(result["repro"]).parent.name
        if result["status"] == "ok":
            done += 1
            best_gap = None
            for label, r in result["results"].items():
                gap = r.get("gap_cd") or r.get("gap_default")
                if gap and (best_gap is None or gap > best_gap):
                    best_gap = gap
            gap_str = f"{best_gap:.2f}x" if best_gap else "?"
            print(f"  [{done+failed}/{len(repros)}] OK  gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  gap={gap_str}  {repro_name}")
            all_results[result["repro"]] = result["results"]
        else:
            failed += 1
            print(f"  [{done+failed}/{len(repros)}] FAIL gpu={result['gpu']}  "
                  f"{result['elapsed']:.1f}s  {repro_name}: {result['error'][:80]}")

    # Wait for workers
    for p in workers:
        p.join(timeout=10)

    elapsed_total = time.time() - start_time
    print(f"\nDone: {done} ok, {failed} failed in {elapsed_total:.1f}s "
          f"({elapsed_total/max(done+failed,1):.1f}s/repro effective)")

    # Save perf.json per repro
    if args.update_perf and all_results:
        hardware = args.hardware or _detect_hw()
        tag = args.tag or "latest"
        for repro_path, results in all_results.items():
            perf_path = Path(repro_path).parent / "perf.json"
            if perf_path.exists():
                perf = json.loads(perf_path.read_text())
            else:
                perf = {}
            if hardware not in perf:
                perf[hardware] = {}
            if tag not in perf[hardware]:
                perf[hardware][tag] = {}
            for shape_label, r in results.items():
                entry = {k: v for k, v in r.items()}
                entry["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                perf[hardware][tag][shape_label] = entry
            perf_path.write_text(json.dumps(perf, indent=2))
        print(f"[perf] Updated perf.json for {len(all_results)} repros (hardware={hardware}, tag={tag})")

    # Summary report
    if all_results:
        gaps = []
        for repro_path, results in all_results.items():
            for label, r in results.items():
                gap = r.get("gap_cd") or r.get("gap_default")
                if gap is not None:
                    gaps.append((gap, Path(repro_path).parent.name, label,
                                 r.get("total_bytes", 0)))
        if gaps:
            gaps.sort(reverse=True)
            print(f"\n{'='*70}")
            print(f"Top gaps (worst SOL ratio):")
            for gap, name, label, nbytes in gaps[:15]:
                size_str = f"{nbytes/1024:.0f}KB" if nbytes < 1e6 else f"{nbytes/1e6:.1f}MB"
                print(f"  {gap:5.2f}x  {size_str:>8s}  {name}  [{label}]")
            avg_gap = sum(g for g, *_ in gaps) / len(gaps)
            at_sol = sum(1 for g, *_ in gaps if g <= 1.1)
            print(f"\n  Median gap: {sorted(g for g,*_ in gaps)[len(gaps)//2]:.2f}x")
            print(f"  Mean gap:   {avg_gap:.2f}x")
            print(f"  At SOL (<=1.1x): {at_sol}/{len(gaps)} ({at_sol*100//len(gaps)}%)")

    # Optional JSON output
    if args.output:
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"[output] Wrote {args.output}")


def _locked_worker(gpu: dict, task_queue, result_queue, args_dict):
    """Acquire GPU lock, run persistent worker subprocess, respawn on CUDA error."""
    from gpu_lock import gpu_lock
    import subprocess

    with gpu_lock(gpu["index"], label=f"bench_parallel gpu{gpu['index']}"):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu["index"]

        proc = None

        def _spawn_worker():
            return subprocess.Popen(
                [sys.executable, "-c", _persistent_worker_script(gpu["index"], args_dict)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
                env=env, cwd=args_dict["root"],
            )

        def _kill_worker(p):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except Exception:
                    p.kill()

        while True:
            try:
                repro_path = task_queue.get_nowait()
            except Exception:
                break

            # Ensure we have a live worker
            if proc is None or proc.poll() is not None:
                _kill_worker(proc)
                proc = _spawn_worker()

            start = time.time()
            try:
                proc.stdin.write(str(repro_path) + "\n")
                proc.stdin.flush()

                line = proc.stdout.readline()
                elapsed = time.time() - start

                if not line:
                    # Worker died
                    _kill_worker(proc)
                    proc = None
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": "worker crashed (CUDA error)",
                    })
                    continue

                line = line.strip()
                if line.startswith("{"):
                    results = json.loads(line)
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "ok",
                        "elapsed": elapsed,
                        "results": results,
                    })
                elif "CUDA_ERROR" in line:
                    # Worker will exit, respawn on next iteration
                    _kill_worker(proc)
                    proc = None
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": line[:200],
                    })
                else:
                    result_queue.put({
                        "repro": str(repro_path),
                        "gpu": gpu["index"],
                        "status": "failed",
                        "elapsed": elapsed,
                        "error": line[:200],
                    })
            except Exception as e:
                _kill_worker(proc)
                proc = None
                result_queue.put({
                    "repro": str(repro_path),
                    "gpu": gpu["index"],
                    "status": "failed",
                    "elapsed": time.time() - start,
                    "error": str(e)[:200],
                })

        _kill_worker(proc)


def _persistent_worker_script(gpu_idx: str, args_dict: dict) -> str:
    """Script for a persistent worker that reads repro paths from stdin."""
    return f'''
import sys, json, os, traceback
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"
sys.path.insert(0, "{args_dict["root"]}")

import torch, torch._dynamo
import torch._inductor.config as inductor_config
from triton.testing import do_bench
import importlib.util, math
from repro_harness import load_shape_configs, make_inputs_from_config

def _bench_with_cudagraph(fn, inps, warmup, rep):
    with torch.no_grad():
        for _ in range(3):
            fn(*inps)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*inps)
        torch.cuda.synchronize()
    return do_bench(lambda: g.replay(), warmup=warmup, rep=rep) * 1000

def bench_one(repro_path):
    spec = importlib.util.spec_from_file_location("repro", repro_path)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)

    instance = mod.Repro()

    all_shapes = {args_dict["all_shapes"]}
    configs = load_shape_configs(repro_path)
    if all_shapes and configs:
        shape_items = list(configs.items())
    else:
        shape_items = [(None, None)]

    all_results = {{}}
    for shape_name, shape_config in shape_items:
        if shape_config is not None:
            inputs = make_inputs_from_config(shape_config)
            label = shape_name
        else:
            inputs = mod.make_inputs() if hasattr(mod, "make_inputs") else mod._default_make_inputs()
            label = "default"

        with torch.no_grad():
            eager_out = instance(*inputs)

        total_bytes = sum(t.nelement() * t.element_size() for t in inputs if isinstance(t, torch.Tensor))
        if isinstance(eager_out, torch.Tensor):
            total_bytes += eager_out.nelement() * eager_out.element_size()
        elif isinstance(eager_out, (tuple, list)):
            for o in eager_out:
                if isinstance(o, torch.Tensor):
                    total_bytes += o.nelement() * o.element_size()

        copy_elems = max(total_bytes // 8, 256)
        src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
        dst = torch.empty_like(src)
        sol_us = do_bench(lambda: dst.copy_(src), warmup={args_dict["n_warmup"]}, rep={args_dict["n_rep"]}) * 1000
        del src, dst

        torch._dynamo.reset()
        compiled = torch.compile(instance)
        compiled_us = _bench_with_cudagraph(compiled, inputs, {args_dict["n_warmup"]}, {args_dict["n_rep"]})

        cd_us = None
        if not {args_dict["no_cd"]}:
            inductor_config.coordinate_descent_tuning = True
            torch._dynamo.reset()
            compiled_cd = torch.compile(instance)
            cd_us = _bench_with_cudagraph(compiled_cd, inputs, {args_dict["n_warmup"]}, {args_dict["n_rep"]})
            inductor_config.coordinate_descent_tuning = False

        all_results[label] = {{
            "compiled_us": compiled_us,
            "coord_descent_us": cd_us,
            "memcopy_sol_us": sol_us,
            "total_bytes": total_bytes,
            "gap_default": compiled_us / sol_us if sol_us > 0 else None,
            "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
        }}

    return all_results

# Main loop: read repro paths from stdin, write JSON results to stdout
for line in sys.stdin:
    line = line.strip()
    if not line or line == "EXIT":
        break
    try:
        results = bench_one(line)
        print(json.dumps(results), flush=True)
    except Exception as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            print(f"CUDA_ERROR: {{str(e)[:150]}}", flush=True)
            sys.exit(1)  # die so parent respawns
        print(f"ERROR: {{str(e)[:150]}}", flush=True)
'''


def _worker_script(repro_path: str, gpu_idx: str, args_dict: dict) -> str:
    """Generate a self-contained benchmark script for one repro."""
    return f'''
import sys, json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "{gpu_idx}"
sys.path.insert(0, "{args_dict["root"]}")

import torch, torch._dynamo
import torch._inductor.config as inductor_config
from triton.testing import do_bench
import importlib.util, math

spec = importlib.util.spec_from_file_location("repro", "{repro_path}")
mod = importlib.util.module_from_spec(spec)
mod.device = torch.device
mod.inf = math.inf
mod.nan = math.nan
spec.loader.exec_module(mod)

from repro_harness import load_shape_configs, make_inputs_from_config

instance = mod.Repro()

# Determine which shapes to run
all_shapes = {args_dict["all_shapes"]}
configs = load_shape_configs("{repro_path}")
if all_shapes and configs:
    shape_items = list(configs.items())
else:
    shape_items = [(None, None)]

all_results = {{}}
for shape_name, shape_config in shape_items:
    if shape_config is not None:
        inputs = make_inputs_from_config(shape_config)
        label = shape_name
    else:
        inputs = mod.make_inputs() if hasattr(mod, "make_inputs") else mod._default_make_inputs()
        label = "default"

    with torch.no_grad():
        eager_out = instance(*inputs)

    total_bytes = sum(t.nelement() * t.element_size() for t in inputs if isinstance(t, torch.Tensor))
    if isinstance(eager_out, torch.Tensor):
        total_bytes += eager_out.nelement() * eager_out.element_size()
    elif isinstance(eager_out, (tuple, list)):
        for o in eager_out:
            if isinstance(o, torch.Tensor):
                total_bytes += o.nelement() * o.element_size()

    copy_elems = max(total_bytes // 8, 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_us = do_bench(lambda: dst.copy_(src), warmup={args_dict["n_warmup"]}, rep={args_dict["n_rep"]}) * 1000
    del src, dst

    def _bench_with_cudagraph(fn, inps, warmup, rep):
        with torch.no_grad():
            for _ in range(3):
                fn(*inps)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn(*inps)
            torch.cuda.synchronize()
        return do_bench(lambda: g.replay(), warmup=warmup, rep=rep) * 1000

    torch._dynamo.reset()
    compiled = torch.compile(instance)
    compiled_us = _bench_with_cudagraph(compiled, inputs, {args_dict["n_warmup"]}, {args_dict["n_rep"]})

    cd_us = None
    if not {args_dict["no_cd"]}:
        inductor_config.coordinate_descent_tuning = True
        torch._dynamo.reset()
        compiled_cd = torch.compile(instance)
        cd_us = _bench_with_cudagraph(compiled_cd, inputs, {args_dict["n_warmup"]}, {args_dict["n_rep"]})
        inductor_config.coordinate_descent_tuning = False

    all_results[label] = {{
        "compiled_us": compiled_us,
        "coord_descent_us": cd_us,
        "memcopy_sol_us": sol_us,
        "total_bytes": total_bytes,
        "gap_default": compiled_us / sol_us if sol_us > 0 else None,
        "gap_cd": cd_us / sol_us if (cd_us and sol_us > 0) else None,
    }}

print(json.dumps(all_results))
'''


def _detect_hw() -> str:
    gpus = discover_gpus()
    return gpus[0]["kind"] if gpus else "unknown"


def _run_compare(paths: list[Path], tag_a: str, tag_b: str):
    """Compare two tagged runs across all repros."""
    repros = find_repros(paths)
    if not repros:
        print("No repro.py files found.")
        return

    diffs = []
    for repro in repros:
        perf_path = repro.parent / "perf.json"
        if not perf_path.exists():
            continue
        perf = json.loads(perf_path.read_text())
        name = repro.parent.name

        for hardware, hw_data in perf.items():
            a_data = hw_data.get(tag_a, {})
            b_data = hw_data.get(tag_b, {})

            for shape in set(list(a_data.keys()) + list(b_data.keys())):
                a = a_data.get(shape, {})
                b = b_data.get(shape, {})

                a_gap = a.get("gap_cd") or a.get("gap_default")
                b_gap = b.get("gap_cd") or b.get("gap_default")
                a_us = a.get("coord_descent_us") or a.get("compiled_us")
                b_us = b.get("coord_descent_us") or b.get("compiled_us")

                if a_us and b_us:
                    speedup = a_us / b_us
                    diffs.append({
                        "name": name,
                        "shape": shape,
                        "hardware": hardware,
                        "a_us": a_us,
                        "b_us": b_us,
                        "speedup": speedup,
                        "a_gap": a_gap,
                        "b_gap": b_gap,
                    })

    if not diffs:
        print(f"No comparable results found for tags '{tag_a}' and '{tag_b}'")
        return

    diffs.sort(key=lambda d: d["speedup"])

    print(f"Comparing '{tag_a}' vs '{tag_b}' ({len(diffs)} shapes)")
    print(f"{'='*70}")

    # Regressions (b slower than a)
    regressions = [d for d in diffs if d["speedup"] > 1.05]
    improvements = [d for d in diffs if d["speedup"] < 0.95]
    unchanged = [d for d in diffs if 0.95 <= d["speedup"] <= 1.05]

    if improvements:
        print(f"\nImprovements ({len(improvements)}):")
        for d in improvements[:15]:
            print(f"  {d['speedup']:.2f}x  {d['a_us']:7.1f} -> {d['b_us']:7.1f} us  "
                  f"gap {d['a_gap']:.2f}x -> {d['b_gap']:.2f}x  {d['name']}[{d['shape']}]")

    if regressions:
        print(f"\nRegressions ({len(regressions)}):")
        for d in reversed(regressions[-15:]):
            print(f"  {d['speedup']:.2f}x  {d['a_us']:7.1f} -> {d['b_us']:7.1f} us  "
                  f"gap {d['a_gap']:.2f}x -> {d['b_gap']:.2f}x  {d['name']}[{d['shape']}]")

    print(f"\nSummary: {len(improvements)} improved, {len(regressions)} regressed, "
          f"{len(unchanged)} unchanged")
    geomean = 1.0
    for d in diffs:
        geomean *= d["speedup"]
    geomean = geomean ** (1.0 / len(diffs))
    print(f"Geometric mean speedup ({tag_a} -> {tag_b}): {geomean:.3f}x")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
