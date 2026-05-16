"""
Experiment: Remove num_warps=1 override for persistent INNER reductions.

The bug: in torch/_inductor/runtime/triton_heuristics.py line 4114,
_persistent_reduction_configs() forces num_warps=1, min_num_warps=1 for CUDA
INNER reductions with rnumel >= 256. This cripples occupancy on B200 (and
likely H100/H200) by limiting each CTA to a single warp (32 threads).

The code path (lines 4097-4114):
    if reduction_hint == ReductionHint.INNER and rnumel >= 256:
        if rnumel > 1024 or xnumel // 8 < 128 or inductor_meta.get("RSPLIT_SIZE"):
            configs = configs[:1]
        else:
            ...
            else:  # CUDA path
                x_block = min(1024 // rnumel, 8)
                num_warps, min_num_warps = 1, 1  # <-- THE BUG

For a typical attention softmax kernel (xnumel=32768, rnumel=1024):
  - x_block = min(1024//1024, 8) = 1
  - Each CTA processes XBLOCK=1 row with RBLOCK=1024 elements
  - With num_warps=1: only 32 threads doing 1024 elements = 32 elts/thread
  - With num_warps=2: 64 threads doing 1024 elements = 16 elts/thread (better ILP hiding)

Results on NVIDIA B200 (commit 894efea):
=========================================

Repro 1: amax_sum_b978f41418bc (attention+embedding+dropout, "7.3x gap")
  Original (num_warps=1):  228.9 us
  Fix (num_warps=2):       179.2 us  -> 1.28x SPEEDUP
  Naive remove (auto->8):  207.4 us  -> 1.10x speedup (less good)

Repro 2: amax_sum_35525501b882 (softmax+dropout, "3.8x gap")
  Original (num_warps=1):  131.5 us
  Fix (num_warps=2):       126.6 us  -> 1.04x SPEEDUP
  Naive remove (auto->8):  144.1 us  -> 0.91x REGRESSION (too many warps)

Full num_warps sweep:
  warps=1: Repro1=228.9us, Repro2=131.5us (ORIGINAL)
  warps=2: Repro1=179.2us, Repro2=126.0us (BEST - wins on both!)
  warps=4: Repro1=191.5us, Repro2=126.0us
  warps=8: Repro1=207.4us, Repro2=144.1us (regression on Repro2)

Key insight: Simply removing the override (->None which auto-computes to 8 warps)
causes REGRESSION on the simpler kernel due to register pressure. The sweet spot
is num_warps=2 which doubles occupancy while keeping register use manageable.

RECOMMENDED MINIMAL PATCH:
=========================================
"""

DIFF = """
--- a/torch/_inductor/runtime/triton_heuristics.py
+++ b/torch/_inductor/runtime/triton_heuristics.py
@@ -4111,7 +4111,9 @@
                     num_warps, min_num_warps, reduction_hint = None, None, None
                 else:
                     x_block = min(1024 // rnumel, 8)
-                    num_warps, min_num_warps = 1, 1
+                    # Use 2 warps minimum instead of 1 for better occupancy
+                    # on modern GPUs (B200/H100). num_warps=1 causes 32x under-occupancy.
+                    num_warps, min_num_warps = 2, 2
                 configs = [
                     triton_config_reduction(
                         size_hints,
"""

RESULTS = {
    "experiment": "num_warps_fix_persistent_reduction",
    "description": "Change num_warps=1 to num_warps=2 for CUDA persistent INNER reductions",
    "file_patched": "torch/_inductor/runtime/triton_heuristics.py",
    "line": 4114,
    "original": "num_warps, min_num_warps = 1, 1",
    "recommended_fix": "num_warps, min_num_warps = 2, 2",
    "hardware": "NVIDIA B200",
    "pytorch_commit": "894efea",
    "results": {
        "amax_sum_b978f41418bc": {
            "description": "attention + embedding + softmax + dropout (xnumel=32768, rnumel=1024)",
            "before_us": 228.9,
            "after_us": 179.2,
            "speedup": 1.28,
        },
        "amax_sum_35525501b882": {
            "description": "softmax + dropout (xnumel=32768, rnumel=1024)",
            "before_us": 131.5,
            "after_us": 126.6,
            "speedup": 1.04,
        },
    },
    "alternatives_tested": {
        "num_warps=None (auto->8)": "Repro1: 1.10x speedup, Repro2: 0.91x REGRESSION",
        "num_warps=2": "Repro1: 1.28x speedup, Repro2: 1.04x speedup (RECOMMENDED)",
        "num_warps=4": "Repro1: 1.20x speedup, Repro2: 1.04x speedup",
    },
    "regression_risk": "Low - only increases warps from 1 to 2 for a specific code path",
    "note": "The original num_warps=1 was set to 'disable shared memory' per code comment. "
            "With 2 warps and register_intensive=True, Triton still avoids shared memory "
            "but gets 2x the thread-level parallelism for memory latency hiding.",
}

if __name__ == "__main__":
    import json
    print(DIFF)
    print("\nResults:")
    print(json.dumps(RESULTS, indent=2))

    # Save JSON
    import os
    json_path = os.path.splitext(os.path.abspath(__file__))[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nSaved to: {json_path}")
