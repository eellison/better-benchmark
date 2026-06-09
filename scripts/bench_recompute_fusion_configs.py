"""
Benchmark the 4 RECOMPUTE_FUSION repros with various inductor configs
to understand what helps with the 2-pass re-read pattern.

The 2-pass pattern:
  Pass 1: reads large tensor, computes reduction (sum/var/mean) -> small result
  Pass 2: re-reads same large tensor, uses reduction result in pointwise output

Configs tested:
  1. Default (baseline)
  2. coordinate_descent_tuning = True
  3. cooperative_reductions = True
  4. force_cooperative_reductions = True
  5. combo_kernels + per_subkernel_blocks
  6. multi_kernel = 2 (force persistent)
  7. multi_kernel = 1 (tuned persistent/non-persistent)
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
from triton.testing import do_bench

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPROS = [
    ("sum_sum_6a14a9c9ba88", "InceptionV3 (channels-last BN bwd)"),
    ("sum_sum_a7c7ab7bfcf5", "InceptionV3 (variant)"),
    ("sum_sum_sum_70d71fcb0d68", "ConvNeXtV2 (GRN+GELU bwd)"),
    ("sum_sum_sum_7b24a3457260", "T5 (attention softmax bwd)"),
]

CONFIGS = {
    "default": {},
    "coord_descent": {
        "coordinate_descent_tuning": True,
    },
    "cooperative": {
        "triton.cooperative_reductions": True,
    },
    "force_cooperative": {
        "triton.force_cooperative_reductions": True,
    },
    "combo_kernels": {
        "combo_kernels": True,
        "combo_kernel_per_subkernel_blocks": True,
    },
    "multi_kernel_tuned": {
        "triton.multi_kernel": 1,
    },
    "multi_kernel_persistent": {
        "triton.multi_kernel": 2,
    },
    "cd_plus_cooperative": {
        "coordinate_descent_tuning": True,
        "triton.cooperative_reductions": True,
    },
    "cd_plus_force_cooperative": {
        "coordinate_descent_tuning": True,
        "triton.force_cooperative_reductions": True,
    },
}


def load_repro(name):
    """Load a repro module and return (Repro class, make_inputs fn)."""
    repro_dir = REPO_ROOT / "repros" / "canonical" / name
    repro_path = repro_dir / "repro.py"

    spec = importlib.util.spec_from_file_location(f"repro_{name}", repro_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    return mod.Repro, mod.make_inputs


@contextmanager
def inductor_config_context(config_overrides):
    """Temporarily override inductor config values."""
    old_values = {}
    for key, val in config_overrides.items():
        if "." in key:
            # Handle nested config like "triton.cooperative_reductions"
            parts = key.split(".")
            obj = inductor_config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            old_values[key] = getattr(obj, parts[-1])
            setattr(obj, parts[-1], val)
        else:
            old_values[key] = getattr(inductor_config, key)
            setattr(inductor_config, key, val)
    try:
        yield
    finally:
        for key, val in old_values.items():
            if "." in key:
                parts = key.split(".")
                obj = inductor_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], val)
            else:
                setattr(inductor_config, key, val)


def benchmark_config(repro_cls, make_inputs_fn, config_overrides, n_warmup=25, n_rep=100):
    """Compile and benchmark a repro with given config overrides.

    Returns dict with timing info, or error string on failure.
    """
    torch._dynamo.reset()

    try:
        with inductor_config_context(config_overrides):
            model = repro_cls()
            inputs = make_inputs_fn()

            compiled = torch.compile(model)

            # Warmup (includes compilation)
            with torch.no_grad():
                for _ in range(3):
                    compiled(*inputs)
                torch.cuda.synchronize()

            # CUDAGraph capture for accurate timing
            with torch.no_grad():
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    compiled(*inputs)
                torch.cuda.synchronize()

            # Benchmark
            ms = do_bench(lambda: g.replay(), warmup=n_warmup, rep=n_rep, return_mode="min")
            us = ms * 1000

            return {"us": us, "status": "ok"}
    except Exception as e:
        return {"us": None, "status": "error", "error": str(e)[:200]}


def count_kernels_for_config(repro_cls, make_inputs_fn, config_overrides):
    """Count how many Triton kernels are generated with given config."""
    import glob
    from torch._inductor.utils import fresh_inductor_cache

    torch._dynamo.reset()

    try:
        with inductor_config_context(config_overrides):
            with fresh_inductor_cache():
                model = repro_cls()
                inputs = make_inputs_fn()
                compiled = torch.compile(model)
                with torch.no_grad():
                    compiled(*inputs)
                    torch.cuda.synchronize()

                from torch._inductor.codecache import cache_dir
                cd = cache_dir()
                py_files = sorted(
                    glob.glob(os.path.join(cd, "**", "*.py"), recursive=True),
                    key=os.path.getmtime
                )
                for f in reversed(py_files):
                    with open(f) as fh:
                        content = fh.read()
                    if 'def call(' in content and '.run(' in content:
                        runs = [l for l in content.split('\n')
                                if '.run(' in l and not l.strip().startswith('#')]
                        return len(runs)
        return 0
    except Exception as e:
        return f"error: {str(e)[:100]}"


def analyze_reduction_dims(repro_cls, make_inputs_fn):
    """Analyze the reduction structure by inspecting the graph."""
    torch._dynamo.reset()

    model = repro_cls()
    inputs = make_inputs_fn()

    # Get tensor sizes to understand the reduction dimensions
    tensor_info = []
    for i, inp in enumerate(inputs):
        if isinstance(inp, torch.Tensor):
            tensor_info.append({
                "idx": i,
                "shape": list(inp.shape),
                "dtype": str(inp.dtype),
                "stride": list(inp.stride()),
                "nbytes_MB": inp.nbytes / 1e6,
            })

    return tensor_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repro", type=str, default=None,
                       help="Run only a specific repro (by hash)")
    parser.add_argument("--config", type=str, default=None,
                       help="Run only a specific config")
    parser.add_argument("--count-kernels", action="store_true",
                       help="Also count generated kernels per config")
    parser.add_argument("--n-warmup", type=int, default=25)
    parser.add_argument("--n-rep", type=int, default=100)
    parser.add_argument("--output", type=str, default=None,
                       help="Save results JSON to file")
    parser.add_argument("--no-gpu-lock", action="store_true")
    args = parser.parse_args()

    # Filter repros if requested
    repros_to_run = REPROS
    if args.repro:
        repros_to_run = [(n, d) for n, d in REPROS if args.repro in n]
        if not repros_to_run:
            print(f"No repro matching '{args.repro}'")
            sys.exit(1)

    # Filter configs if requested
    configs_to_run = CONFIGS
    if args.config:
        if args.config not in CONFIGS:
            print(f"Unknown config '{args.config}'. Available: {list(CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGS[args.config]}

    results = {}

    print(f"{'='*80}")
    print(f"RECOMPUTE_FUSION 2-pass re-read pattern: config comparison")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"{'='*80}")

    for repro_name, repro_desc in repros_to_run:
        print(f"\n{'─'*80}")
        print(f"Repro: {repro_name} ({repro_desc})")
        print(f"{'─'*80}")

        try:
            repro_cls, make_inputs_fn = load_repro(repro_name)
        except Exception as e:
            print(f"  ERROR loading repro: {e}")
            continue

        # Analyze input shapes
        tensor_info = analyze_reduction_dims(repro_cls, make_inputs_fn)
        large_tensors = [t for t in tensor_info if t["nbytes_MB"] > 1]
        if large_tensors:
            total_large_mb = sum(t["nbytes_MB"] for t in large_tensors)
            print(f"  Large tensors: {len(large_tensors)}, total {total_large_mb:.1f} MB")
            for t in large_tensors[:3]:
                print(f"    [{t['idx']}] shape={t['shape']} stride={t['stride']} ({t['nbytes_MB']:.1f} MB)")

        repro_results = {}

        for cfg_name, cfg_overrides in configs_to_run.items():
            print(f"\n  Config: {cfg_name}", end="", flush=True)

            result = benchmark_config(
                repro_cls, make_inputs_fn, cfg_overrides,
                n_warmup=args.n_warmup, n_rep=args.n_rep
            )

            if result["status"] == "ok":
                print(f"  -> {result['us']:.1f} us", end="")
            else:
                print(f"  -> ERROR: {result.get('error', 'unknown')[:80]}", end="")

            if args.count_kernels and result["status"] == "ok":
                n_kernels = count_kernels_for_config(repro_cls, make_inputs_fn, cfg_overrides)
                result["n_kernels"] = n_kernels
                print(f"  ({n_kernels} kernels)", end="")

            print()
            repro_results[cfg_name] = result

        # Summary for this repro
        ok_results = {k: v for k, v in repro_results.items() if v["status"] == "ok"}
        if ok_results:
            baseline = ok_results.get("default", {}).get("us")
            if baseline:
                print(f"\n  Summary (vs default={baseline:.1f} us):")
                for cfg_name, r in sorted(ok_results.items(), key=lambda x: x[1]["us"]):
                    speedup = baseline / r["us"]
                    marker = " *BEST*" if r["us"] == min(v["us"] for v in ok_results.values()) else ""
                    print(f"    {cfg_name:30s} {r['us']:8.1f} us  ({speedup:.3f}x){marker}")

        results[repro_name] = repro_results

    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    for repro_name, repro_desc in repros_to_run:
        if repro_name not in results:
            continue
        repro_results = results[repro_name]
        ok_results = {k: v for k, v in repro_results.items() if v["status"] == "ok"}
        if not ok_results:
            continue

        baseline = ok_results.get("default", {}).get("us", 0)
        best_cfg = min(ok_results.items(), key=lambda x: x[1]["us"])
        best_speedup = baseline / best_cfg[1]["us"] if baseline else 0

        print(f"\n  {repro_name} ({repro_desc}):")
        print(f"    Default: {baseline:.1f} us")
        print(f"    Best:    {best_cfg[1]['us']:.1f} us ({best_cfg[0]}, {best_speedup:.3f}x)")

        # Check cooperative specifically
        coop = ok_results.get("force_cooperative", {}).get("us")
        if coop:
            coop_speedup = baseline / coop
            print(f"    Force cooperative: {coop:.1f} us ({coop_speedup:.3f}x)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
