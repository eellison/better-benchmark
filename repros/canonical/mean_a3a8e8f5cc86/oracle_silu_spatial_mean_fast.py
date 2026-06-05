"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete f32 NFNet SiLU producer and 7x7 spatial mean for channels-last `[128,2304,7,7]` in one Triton reduction kernel that writes the final contiguous `[128,2304]` reshape directly, whereas Inductor currently lowers the same pointwise producer plus small spatial mean through its generic fused reduction schedule; Inductor cannot do this today because scheduler/codegen has no guarded small-spatial reduction template that sinks this expensive SiLU producer and emits the final view layout with enough row/channel tiling control; the fix is SCHEDULER_FUSION: add a benchmark-gated pointwise-producer-to-small-spatial-mean schedule for fixed NCHW channels-last tensors that chooses direct output-layout stores."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 128
CHANNELS = 2304
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CHANNELS_LAST_X_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
INV_HW = 1.0 / HW

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
    from torch._inductor.runtime.triton_helpers import math as tl_math

    triton_helpers.set_driver_to_gpu()

    @triton.jit
    def _silu_spatial_mean_kernel(
        x_ptr,
        out_ptr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < 2304
        base_offsets = batch * 112896 + c_offsets

        hw_offsets = tl.arange(0, 64)
        hw_mask = hw_offsets < 49
        x = tl.load(
            x_ptr + base_offsets[:, None] + hw_offsets[None, :] * 2304,
            mask=c_mask[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        silu = x / (tl.exp(-x) + 1.0)
        reduced = tl.sum(tl.where(hw_mask[None, :], silu, 0.0), axis=1) * 0.02040816326530612

        tl.store(out_ptr + batch * 2304 + c_offsets, reduced, mask=c_mask)

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 524288, "r0_": 64},
        reduction_hint=ReductionHint.DEFAULT,
        filename=__file__,
        triton_meta={
            "signature": {
                "out_ptr": "*fp32",
                "x_ptr": "*fp32",
                "xnumel": "i32",
                "r0_numel": "i32",
                "XBLOCK": "constexpr",
            },
            "device": DeviceProperties.create(torch.device("cuda", 0)),
            "constants": {},
            "native_matmul": False,
            "enable_fp_fusion": True,
            "launch_pdl": False,
            "disable_ftz": False,
            "configs": [
                {
                    (0,): [["tt.divisibility", 16]],
                    (1,): [["tt.divisibility", 16]],
                    (2,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "_silu_spatial_mean_persistent_kernel",
            "mutated_arg_names": [],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 1,
            "num_store": 1,
            "num_reduction": 1,
            "autotune_hints": set(),
            "tiling_scores": {"x": 60162048, "r0_": 0},
            "autotune_local_cache": True,
            "autotune_pointwise": True,
            "dynamic_scale_rblock": True,
            "incremental_autotune": False,
            "max_autotune": False,
            "max_autotune_pointwise": False,
            "min_split_scan_rblock": 256,
            "spill_threshold": 16,
            "store_cubin": False,
            "deterministic": False,
            "batch_invariant": False,
            "force_filter_reduction_configs": False,
            "mix_order_reduction_allow_multi_stages": True,
            "dynamic_disable_pipelining": True,
            "are_deterministic_algorithms_enabled": False,
        },
    )
    @triton.jit
    def _silu_spatial_mean_persistent_kernel(
        out_ptr,
        x_ptr,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 294912
        r0_numel = 49
        R0_BLOCK: tl.constexpr = 64
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_mask = r0_index < r0_numel
        x0 = xindex % 2304
        x1 = xindex // 2304
        x3 = xindex
        x = tl.load(
            x_ptr + (x0 + 2304 * r0_index + 112896 * x1),
            r0_mask,
            eviction_policy="evict_first",
            other=0.0,
        )
        silu = x / (tl_math.exp(-x) + 1.0)
        reduced = tl.sum(tl.where(r0_mask, silu, 0.0), 1)[:, None].to(tl.float32)
        tl.store(out_ptr + x3, reduced * 0.02040816326530612, None)


def _require_input_tensor(value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(
            f"{REPRO_ID} input shape {tuple(value.shape)} != "
            f"{(BATCH, CHANNELS, HEIGHT, WIDTH)}"
        )
    if value.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{REPRO_ID} expects CUDA input")
    if tuple(value.stride()) != CHANNELS_LAST_X_STRIDE:
        raise ValueError(
            f"{REPRO_ID} input stride {tuple(value.stride())} != {CHANNELS_LAST_X_STRIDE}"
        )
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, tuple[int, int]]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, view_shape = inputs
    x_t = _require_input_tensor(x)
    output_shape = tuple(int(dim) for dim in view_shape)
    if output_shape != OUTPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} got unexpected view shape parameter: {view_shape!r}")
    return x_t, output_shape


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_silu_spatial_mean_fast.py")

    x, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    _silu_spatial_mean_persistent_kernel.run(
        output,
        x,
        294912,
        49,
        stream=get_raw_stream(x.device.index if x.device.index is not None else 0),
    )
    return output


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2,
                        help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200,
                        help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true",
                        help="Disable auto-detection and skipping of stochastic outputs")
    parser.add_argument("--all-shapes", action="store_true",
                        help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true",
                        help="Print GPU hardware info and exit")
    args = parser.parse_args()

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=not args.no_skip_stochastic,
        )
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
