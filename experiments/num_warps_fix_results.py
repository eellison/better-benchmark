"""
Experiment: Remove num_warps=1 override for persistent INNER reductions.

The bug: in torch/_inductor/runtime/triton_heuristics.py around line 4113-4114,
_persistent_reduction_configs() forces num_warps=1, min_num_warps=1 for CUDA
INNER reductions with rnumel >= 256. This cripples occupancy on B200.

Fix: Remove the override so num_warps defaults to None (letting
triton_config_reduction compute it from the total numel).

This script measures before/after on two affected repros.
"""
import sys
import os
import time
import json
import textwrap
import importlib
import functools

# Ensure pytorch is on path
sys.path.insert(0, "/tmp/pytorch-work")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from triton.testing import do_bench

# Add repro harness path
sys.path.insert(0, "/tmp/scratch_space/better_benchmark")

# ============================================================================
# THE DIFF (what we'd submit)
# ============================================================================
DIFF_DESCRIPTION = """
--- a/torch/_inductor/runtime/triton_heuristics.py
+++ b/torch/_inductor/runtime/triton_heuristics.py
@@ -4111,7 +4111,8 @@
                     num_warps, min_num_warps, reduction_hint = None, None, None
                 else:
                     x_block = min(1024 // rnumel, 8)
-                    num_warps, min_num_warps = 1, 1
+                    # Let num_warps be auto-computed (was: num_warps=1, min_num_warps=1)
+                    num_warps, min_num_warps = None, None
                 configs = [
                     triton_config_reduction(
                         size_hints,
"""


def apply_monkey_patch():
    """Monkey-patch the persistent reduction config to remove num_warps=1 override."""
    import torch._inductor.runtime.triton_heuristics as th

    # Get the source and find _persistent_reduction_configs
    # We'll patch the module-level function by wrapping it
    original_func = th._persistent_reduction_configs

    @functools.wraps(original_func)
    def patched_persistent_reduction_configs(*args, **kwargs):
        # The function returns a list of configs. We need to intercept
        # the internal logic. Since we can't easily patch inside a function,
        # we'll patch _num_warps to refuse min_num_warps=1 for these cases.
        # Actually, let's just patch the source directly at module level.
        pass

    # Better approach: directly modify the source logic by patching
    # triton_config_reduction to ignore num_warps=1 when called from
    # persistent reduction. But that's fragile.
    #
    # Cleanest approach: reload the module after patching the source in memory.
    # Actually simplest: just patch the file temporarily and reimport.
    # But we were told not to permanently modify. So let's use exec-based patching.

    # The REAL cleanest approach: read the function source, modify it, exec it.
    import inspect
    import textwrap

    src = inspect.getsource(th._persistent_reduction_configs)
    # Replace the problematic line
    patched_src = src.replace(
        "num_warps, min_num_warps = 1, 1",
        "num_warps, min_num_warps = None, None  # PATCHED: remove warp override"
    )

    # We need the function's globals to exec it properly
    fn_globals = th._persistent_reduction_configs.__globals__

    # Dedent and compile
    patched_src = textwrap.dedent(patched_src)
    exec(compile(patched_src, "<patched>", "exec"), fn_globals)

    # The exec puts the function into fn_globals
    th._persistent_reduction_configs = fn_globals["_persistent_reduction_configs"]
    print("[PATCH] Applied: num_warps override removed from _persistent_reduction_configs")


def revert_monkey_patch():
    """Revert by reimporting the module."""
    import torch._inductor.runtime.triton_heuristics as th
    importlib.reload(th)
    print("[PATCH] Reverted: reloaded triton_heuristics module")


def benchmark_single(repro_path, label, n_warmup=25, n_rep=200):
    """Compile and benchmark a single repro. Returns time in microseconds."""
    from torch._inductor.utils import fresh_inductor_cache

    # Import the repro module
    repro_dir = os.path.dirname(repro_path)
    if repro_dir not in sys.path:
        sys.path.insert(0, repro_dir)

    spec = importlib.util.spec_from_file_location(f"repro_mod_{label}", repro_path)
    repro_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repro_mod)

    mod = repro_mod.Repro()
    inputs = repro_mod.make_inputs()

    # Reset everything and use fresh cache to force recompilation
    torch._dynamo.reset()

    with fresh_inductor_cache():
        compiled = torch.compile(mod)
        with torch.no_grad():
            # Warmup compilation
            for _ in range(3):
                compiled(*inputs)
            torch.cuda.synchronize()

            # Benchmark
            ms = do_bench(lambda: compiled(*inputs), warmup=n_warmup, rep=n_rep)

    us = ms * 1000
    print(f"  [{label}] {us:.1f} us")
    return us


def run_experiment():
    """Run before/after measurements on both repros."""
    repros = [
        {
            "path": "/tmp/scratch_space/better_benchmark/repros/canonical/amax_sum_b978f41418bc/repro.py",
            "name": "amax_sum_b978f41418bc (attention+dropout, 7.3x gap)",
        },
        {
            "path": "/tmp/scratch_space/better_benchmark/repros/canonical/amax_sum_35525501b882/repro.py",
            "name": "amax_sum_35525501b882 (softmax+dropout, 3.8x gap)",
        },
    ]

    results = {}

    for repro in repros:
        print(f"\n{'='*70}")
        print(f"Repro: {repro['name']}")
        print(f"{'='*70}")

        # --- BEFORE (baseline, with num_warps=1 override) ---
        print("\n[BEFORE] Running with original num_warps=1 override...")
        torch._dynamo.reset()
        before_us = benchmark_single(repro["path"], "BEFORE")

        # --- AFTER (patched, num_warps=None) ---
        print("\n[AFTER] Applying patch (num_warps=None)...")
        apply_monkey_patch()
        torch._dynamo.reset()
        after_us = benchmark_single(repro["path"], "AFTER")

        # Revert
        revert_monkey_patch()

        speedup = before_us / after_us if after_us > 0 else float('inf')
        print(f"\n  BEFORE: {before_us:.1f} us")
        print(f"  AFTER:  {after_us:.1f} us")
        print(f"  SPEEDUP: {speedup:.2f}x")

        results[repro["name"]] = {
            "before_us": before_us,
            "after_us": after_us,
            "speedup": speedup,
            "repro_path": repro["path"],
        }

    # --- Summary ---
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, r in results.items():
        print(f"  {name}")
        print(f"    Before: {r['before_us']:.1f} us | After: {r['after_us']:.1f} us | Speedup: {r['speedup']:.2f}x")

    print(f"\n{'='*70}")
    print("PROPOSED DIFF")
    print(f"{'='*70}")
    print(DIFF_DESCRIPTION)

    # Save results
    results_path = os.path.abspath(__file__)
    json_path = results_path.replace(".py", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "experiment": "num_warps_fix_persistent_reduction",
            "description": "Remove num_warps=1,min_num_warps=1 override for CUDA persistent INNER reductions",
            "file_patched": "torch/_inductor/runtime/triton_heuristics.py",
            "line": "4114",
            "original": "num_warps, min_num_warps = 1, 1",
            "patched": "num_warps, min_num_warps = None, None",
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
            "results": results,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    return results


if __name__ == "__main__":
    run_experiment()
