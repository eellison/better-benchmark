"""
Benchmark runner for extracted kernel repros.

Usage:
    python scripts/bench.py repros/canonical/var_mean_a7cbd072693b/repro.py
    python scripts/bench.py repros/canonical/  # all in dir
"""
import argparse
import glob
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from gpu_lock import gpu_lock, gpu_lock_for_kind


def _resolve_input_path(path_arg):
    path = Path(path_arg)
    stripped_arg = path_arg.strip()
    stripped = Path(stripped_arg)
    if path.exists():
        return path
    if stripped_arg and path_arg != stripped_arg and stripped.exists():
        print(f"[bench] normalized path whitespace: {path_arg!r} -> {str(stripped)!r}")
        return stripped
    raise FileNotFoundError(
        f"Benchmark path does not exist: {path_arg!r}. "
        "Pass a repro.py file or a directory containing repro.py files."
    )


def current_hardware_label():
    import torch
    if not torch.cuda.is_available():
        return "CPU"
    name = torch.cuda.get_device_name()
    for token in ("B200", "H200", "H100", "A100", "V100"):
        if token in name:
            return token
    return name.replace(" ", "_")


def _randn_with_bool(old_randn):
    import torch

    def randn(*args, **kwargs):
        if kwargs.get("dtype") is torch.bool:
            kwargs = dict(kwargs)
            kwargs.pop("dtype")
            if "size" in kwargs:
                size = kwargs.pop("size")
            elif len(args) == 1 and isinstance(args[0], (list, tuple, torch.Size)):
                size = tuple(args[0])
            else:
                size = args
            return torch.randint(0, 2, size, dtype=torch.bool, **kwargs)
        return old_randn(*args, **kwargs)

    return randn


def make_inputs(mod):
    import torch
    old_randn = torch.randn
    torch.randn = _randn_with_bool(old_randn)
    try:
        return mod.make_inputs()
    finally:
        torch.randn = old_randn


def _count_kernels(mod, inputs):
    import torch
    import torch._dynamo
    from torch._inductor.utils import fresh_inductor_cache
    from torch._inductor.codecache import cache_dir

    torch._dynamo.reset()
    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            compiled(*inputs)
            torch.cuda.synchronize()
        cd = cache_dir()
        py_files = sorted(glob.glob(os.path.join(cd, "**", "*.py"), recursive=True), key=os.path.getmtime)
        for f in reversed(py_files):
            with open(f) as fh:
                content = fh.read()
            if 'def call(' in content and '.run(' in content:
                runs = [l for l in content.split('\n') if '.run(' in l and not l.strip().startswith('#')]
                return len(runs)
    return 0


def load_repro(path):
    import torch
    spec = importlib.util.spec_from_file_location("repro", path)
    mod = importlib.util.module_from_spec(spec)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    spec.loader.exec_module(mod)
    mod.device = torch.device
    mod.inf = math.inf
    mod.nan = math.nan
    return mod.Repro(), make_inputs(mod)


def _cuda_graph_time(fn, inputs, n_warmup=10, n_iter=200):
    import time
    import torch

    with torch.no_grad():
        for _ in range(n_warmup):
            fn(*inputs)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn(*inputs)
        torch.cuda.synchronize()

        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iter):
            g.replay()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / n_iter * 1e6
    return elapsed


def benchmark_one(repro_path, n_warmup=25, n_rep=200, use_cuda_graph=True):
    import torch
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from byte_accounting import count_bytes_effective, count_bytes_naive
    from triton.testing import do_bench

    mod, inputs = load_repro(repro_path)

    with torch.no_grad():
        eager_out = mod(*inputs)

    total_bytes_naive = count_bytes_naive(inputs, eager_out)
    total_bytes = count_bytes_effective(mod, inputs)
    if total_bytes_naive > total_bytes * 1.1:
        print(
            f"\nBytes adjusted: {total_bytes_naive/1e6:.1f} MB naive "
            f"-> {total_bytes/1e6:.1f} MB effective"
        )

    n_kernels = _count_kernels(mod, inputs)

    # SOL: memcopy same total bytes
    copy_elems = max(total_bytes // (2 * 4), 256)
    src = torch.empty(copy_elems, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    sol_ms = do_bench(lambda: dst.copy_(src), warmup=n_warmup, rep=n_rep)
    sol_us = sol_ms * 1000
    del src, dst

    # Compiled (default heuristics)
    torch._dynamo.reset()
    compiled = torch.compile(mod)
    if use_cuda_graph:
        compiled_us = _cuda_graph_time(compiled, inputs, n_warmup, n_rep)
    else:
        with torch.no_grad():
            for _ in range(3):
                compiled(*inputs)
            torch.cuda.synchronize()
        compiled_ms = do_bench(lambda: compiled(*inputs), warmup=n_warmup, rep=n_rep)
        compiled_us = compiled_ms * 1000

    # Compiled with coordinate descent tuning
    inductor_config.coordinate_descent_tuning = True
    torch._dynamo.reset()
    compiled_cd = torch.compile(mod)
    if use_cuda_graph:
        cd_us = _cuda_graph_time(compiled_cd, inputs, n_warmup, n_rep)
    else:
        with torch.no_grad():
            for _ in range(3):
                compiled_cd(*inputs)
            torch.cuda.synchronize()
        cd_ms = do_bench(lambda: compiled_cd(*inputs), warmup=n_warmup, rep=n_rep)
        cd_us = cd_ms * 1000
    inductor_config.coordinate_descent_tuning = False

    print(f"\nKernel data: {total_bytes / 1024:.1f} KB (read+write)")
    print(f"Kernels generated: {n_kernels}")
    print(f"Memcopy SOL (same size): {sol_us:8.1f} us")
    print(f"Compiled (default):      {compiled_us:8.1f} us")
    print(f"Compiled (coord desc):   {cd_us:8.1f} us")
    print(f"Gap (default / SOL):     {compiled_us / sol_us:8.2f}x")
    print(f"Gap (CD / SOL):          {cd_us / sol_us:8.2f}x")

    return {
        "compiled_us": compiled_us,
        "coord_descent_us": cd_us,
        "memcopy_sol_us": sol_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="repro.py file or directory containing regions")
    parser.add_argument("--update-meta", action="store_true", help="Write results to meta.json")
    parser.add_argument("--no-cuda-graph", action="store_true", help="Use do_bench instead of CUDA graph replay")
    parser.add_argument("--gpu", default=None, help="GPU id to lock (default: grab any free GPU)")
    parser.add_argument("--device-kind", default=None, help="GPU kind to lock, e.g. H100 or B200")
    parser.add_argument("--hardware", default=None, help="perf metadata key for --update-meta; defaults to detected GPU")
    parser.add_argument("--no-gpu-lock", action="store_true", help="Disable the per-GPU benchmark lock")
    parser.add_argument("--lock-dir", type=Path, default=None, help="Directory for GPU lock files")
    parser.add_argument("--lock-timeout-s", type=float, default=None, help="Seconds to wait for the GPU lock")
    args = parser.parse_args()

    use_cg = not args.no_cuda_graph
    selected_hardware = args.hardware

    def run():
        nonlocal selected_hardware
        input_path = _resolve_input_path(args.path)
        if input_path.is_file():
            result = benchmark_one(str(input_path), use_cuda_graph=use_cg)
            if args.update_meta:
                meta_path = input_path.parent / "meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    hardware = selected_hardware or current_hardware_label()
                    meta.setdefault("perf", {})[hardware] = {
                        "compiled_us": round(result["compiled_us"], 1),
                        "coord_descent_us": round(result["coord_descent_us"], 1),
                        "memcpy_sol_us": round(result["memcopy_sol_us"], 1),
                        "total_bytes": result["total_bytes"],
                    }
                    meta["num_kernels"] = result["n_kernels"]
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
        else:
            repros = sorted(input_path.glob("**/repro.py"))
            print(f"Found {len(repros)} repros in {input_path}\n")
            if not repros:
                raise FileNotFoundError(
                    f"No repro.py files found under directory: {input_path}"
                )
            for repro in repros:
                rel = os.path.relpath(repro.parent, input_path)
                print(f"--- {rel} ---")
                try:
                    benchmark_one(str(repro), use_cuda_graph=use_cg)
                except Exception as e:
                    print(f"  FAILED: {e}")
                print()

    # Acquire GPU lock BEFORE any torch/CUDA imports happen in benchmark_one().
    # `import torch` alone doesn't initialize CUDA — that only happens on the
    # first .cuda() call — but we set CUDA_VISIBLE_DEVICES here to be safe.
    if args.no_gpu_lock:
        if args.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        run()
    elif args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        with gpu_lock(
            args.gpu,
            lock_dir=args.lock_dir,
            timeout_s=args.lock_timeout_s,
            label=f"bench.py {args.path}",
        ) as lock_path:
            print(f"[gpu-lock] acquired {lock_path}")
            run()
    else:
        with gpu_lock_for_kind(
            args.device_kind,
            lock_dir=args.lock_dir,
            timeout_s=args.lock_timeout_s,
            label=f"bench.py {args.path}",
        ) as device_info:
            os.environ["CUDA_VISIBLE_DEVICES"] = device_info["index"]
            selected_hardware = selected_hardware or device_info["kind"]
            print(
                f"[gpu-lock] acquired {device_info['lock_path']} "
                f"({device_info['kind']} {device_info['name']})"
            )
            run()


if __name__ == "__main__":
    main()
