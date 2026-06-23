"""
Helper: run a single repro file, compile it, time it, print JSON result to stdout.
Usage: python scripts/_single_repro_runner.py <repro_path>
"""
import json
import os
import sys
import time
import traceback

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# gpu_lock.py lives in scripts/, so scripts/ must be on sys.path for the bare
# `from gpu_lock import gpu_lock` below to resolve (repo root alone is not enough).
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import torch
torch.cuda.empty_cache()

repro_path = sys.argv[1]

import importlib.util
spec = importlib.util.spec_from_file_location("repro_mod", repro_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

Repro = mod.Repro
make_inputs = mod.make_inputs

try:
    model = Repro().cuda()
    inputs = make_inputs()
    inputs = [x.cuda() if hasattr(x, 'cuda') else x for x in inputs]

    # Compile
    t0 = time.time()
    compiled = torch.compile(model)
    # Warmup (triggers compilation)
    for _ in range(3):
        out = compiled(*inputs)
    torch.cuda.synchronize()
    compile_time = time.time() - t0

    # CUDA graph capture for timing
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = compiled(*inputs)
    stream.synchronize()

    # Benchmark UNDER THE GPU LOCK. Running one repro per process is NOT enough
    # to get a clean number: a co-tenant kernel from any other process on this
    # GPU inflates the timing window. The lock (advisory flock per GPU) is what
    # actually serializes the timed region. (See tests/test_no_unlocked_timing.py.)
    from gpu_lock import gpu_lock
    from triton.testing import do_bench
    # Lock logical index 0: the caller (rerun_oom_singles.py) already pins this
    # subprocess to one physical GPU via CUDA_VISIBLE_DEVICES, and CVD remaps that
    # device to logical index 0 — the same index the caller's lock dir is keyed on.
    # (We can't auto-discover/re-pin via gpu_lock_for_kind here: torch is imported
    # and CUDA is initialized at module top, so the visible device is already fixed.)
    with gpu_lock(0, label="single_repro_runner"):
        t_us = do_bench(lambda: g.replay(), warmup=25, rep=200, return_mode='min') * 1000
    print(json.dumps({"status": "ok", "time_us": t_us, "compile_time_s": compile_time}))
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    print(json.dumps({"status": "failed", "error": f"{type(e).__name__}: {e}"}))
