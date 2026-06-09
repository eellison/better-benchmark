"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete NFNet exact-erf GELU, fixed gamma scale, and 6x6 spatial mean directly into the final contiguous `[128, 3072]` output with the same shape-specialized channels-last persistent reduction that tuned Inductor selects, whereas Inductor already fuses the single-use GELU producer into the spatial mean and avoids full activation materialization for this captured scope; Inductor cannot materially reduce the local work today because the remaining cost is the mandatory channels-last activation read, exact erf evaluation for every spatial element, small fixed reduction, and output store rather than a missed scheduler fusion or algebraic rewrite; the fix is BANDWIDTH_BOUND: record this as a full-scope floor unless a broader norm-template math-codegen or launch-overhead change moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None

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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)

CLASSIFICATION = "BANDWIDTH_BOUND"
BATCH = 128
CHANNELS = 3072
HEIGHT = 6
WIDTH = 6
HW = HEIGHT * WIDTH
INPUT_STRIDE_N = 110592
INPUT_STRIDE_C = 1
INPUT_STRIDE_H = 18432
INPUT_STRIDE_W = 3072
OUTPUT_STRIDE_N = 3072
OUTPUT_STRIDE_C = 1
GELU_INV_SQRT2 = 0.7071067811865476
NFNET_GAMMA = 1.7015043497085571


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    _async_compile = AsyncCompile()
    _gelu_spatial_mean_kernel = _async_compile.triton(
        "oracle_mean_42b19743bdd2_gelu_spatial_mean",
        r"""
import torch
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import DeviceProperties, ReductionHint
from torch._inductor.runtime.triton_helpers import libdevice
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={"x": 524288, "r0_": 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={
        "signature": {
            "out_ptr0": "*fp32",
            "in_ptr0": "*fp32",
            "xnumel": "i32",
            "r0_numel": "i32",
            "XBLOCK": "constexpr",
        },
        "device": DeviceProperties.create(torch.device("cuda")),
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
        "kernel_name": "oracle_mean_42b19743bdd2_gelu_spatial_mean",
        "mutated_arg_names": [],
        "optimize_mem": True,
        "no_x_dim": None,
        "atomic_add_found": False,
        "num_load": 1,
        "num_store": 1,
        "num_reduction": 1,
        "autotune_hints": set(),
        "tiling_scores": {"x": 59768832, "r0_": 0},
        "autotune_local_cache": True,
        "autotune_pointwise": True,
        "autotune_remote_cache": None,
        "force_disable_caches": False,
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
        "coordinate_descent_tuning": True,
        "coordinate_descent_search_radius": 1,
        "coordinate_descent_check_all_directions": False,
    },
)
@triton.jit
def oracle_mean_42b19743bdd2_gelu_spatial_mean(
    out_ptr0,
    in_ptr0,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
):
    xnumel = 393216
    r0_numel = 36
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = xindex % 3072
    x1 = xindex // 3072
    tmp0 = tl.load(
        in_ptr0 + (x0 + 3072 * r0_index + 110592 * x1),
        r0_mask,
        eviction_policy="evict_first",
        other=0.0,
    )
    tmp1 = tmp0 * 0.5
    tmp2 = tmp0 * 0.7071067811865476
    tmp3 = libdevice.erf(tmp2)
    tmp4 = tmp3 + 1.0
    tmp5 = tmp1 * tmp4
    tmp6 = tmp5 * 1.7015043497085571
    tmp7 = tl.sum(tl.where(r0_mask, tmp6, 0.0), 1)[:, None].to(tl.float32)
    tmp8 = tmp7 / 36.0
    tl.store(out_ptr0 + xindex, tmp8, None)
""",
    )
    _async_compile.wait(globals())
    del _async_compile


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_fused_gelu_spatial_mean.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor")
    if tuple(x.shape) != (BATCH, CHANNELS, HEIGHT, WIDTH):
        raise ValueError(
            f"{REPRO_ID} input 0 shape {tuple(x.shape)} != "
            f"{(BATCH, CHANNELS, HEIGHT, WIDTH)}"
        )
    if tuple(x.stride()) != (INPUT_STRIDE_N, INPUT_STRIDE_C, INPUT_STRIDE_H, INPUT_STRIDE_W):
        raise ValueError(
            f"{REPRO_ID} input 0 stride {tuple(x.stride())} != "
            f"{(INPUT_STRIDE_N, INPUT_STRIDE_C, INPUT_STRIDE_H, INPUT_STRIDE_W)}"
        )
    if x.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError(f"{REPRO_ID} expects CUDA input")
    if list(shape_param) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output shape parameter: {shape_param!r}")
    return x


@oracle_impl(hardware="H100", shapes="(T([128, 3072, 6, 6], f32, stride=(110592, 1, 18432, 3072)), S([128, 3072]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, CHANNELS),
        (OUTPUT_STRIDE_N, OUTPUT_STRIDE_C),
        device=x.device,
        dtype=torch.float32,
    )
    _gelu_spatial_mean_kernel.run(
        output,
        x,
        BATCH * CHANNELS,
        HW,
        stream=get_raw_stream(x.get_device()),
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
