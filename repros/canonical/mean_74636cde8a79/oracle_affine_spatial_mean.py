"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete GhostNet batch-norm-style affine followed by spatial mean returned by Repro.forward, including the channel mean subtract, reciprocal sqrt with eps=1e-5, scale, bias, NaN propagation from random negative variance inputs, and final `[512,672,1,1]` keepdim output layout, using the same persistent Triton reduction envelope as tuned Inductor; bench_oracle measures this full-scope oracle at floor versus torch.compile, so this norm-template canonicalization row does not expose a material local scheduler-fusion, algebraic-elimination, split-K, scatter-reduce, or new-pattern gap; the fix is BANDWIDTH_BOUND: record the repro as at_floor unless broader persistent-reduction, launch, or memory-traffic work moves both implementations."""
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


CLASSIFICATION = "BANDWIDTH_BOUND"
EPS = 1.0e-5
DEFAULT_BATCH = 512
DEFAULT_CHANNELS = 672
DEFAULT_HEIGHT = 7
DEFAULT_WIDTH = 7
DEFAULT_HW = DEFAULT_HEIGHT * DEFAULT_WIDTH
DEFAULT_ROWS = DEFAULT_BATCH * DEFAULT_CHANNELS
DEFAULT_CONV_SHAPE = (DEFAULT_BATCH, DEFAULT_CHANNELS, DEFAULT_HEIGHT, DEFAULT_WIDTH)
DEFAULT_CONV_STRIDE = (
    DEFAULT_CHANNELS * DEFAULT_HW,
    DEFAULT_HW,
    DEFAULT_WIDTH,
    1,
)
OUTPUT_SHAPE = (DEFAULT_BATCH, DEFAULT_CHANNELS, 1, 1)
OUTPUT_STRIDE = (DEFAULT_CHANNELS, 1, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.hints import DeviceProperties, ReductionHint

    triton_helpers.set_driver_to_gpu()

    @triton_heuristics.persistent_reduction(
        size_hints={"x": 524288, "r0_": 64},
        reduction_hint=ReductionHint.INNER,
        filename="/tmp/mean_74636cde8a79_oracle_triton.py",
        triton_meta={
            "signature": {
                "in_out_ptr0": "*fp32",
                "in_ptr0": "*fp32",
                "in_ptr1": "*fp32",
                "in_ptr2": "*fp32",
                "in_ptr3": "*fp32",
                "in_ptr4": "*fp32",
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
                    (3,): [["tt.divisibility", 16]],
                    (4,): [["tt.divisibility", 16]],
                    (5,): [["tt.divisibility", 16]],
                    (6,): [["tt.divisibility", 16]],
                }
            ],
        },
        inductor_meta={
            "grid_type": "Grid1D",
            "kernel_name": "triton_per_fused_add_mean_mul_reciprocal_sqrt_sub_unsqueeze_0",
            "mutated_arg_names": ["in_out_ptr0"],
            "optimize_mem": True,
            "no_x_dim": None,
            "atomic_add_found": False,
            "num_load": 5,
            "num_store": 1,
            "num_reduction": 1,
            "autotune_hints": set(),
            "tiling_scores": {"x": 2763264, "r0_": 67436544},
            "backend_hash": "2FF7A49C450C045FF2AC7A730B5A8D4CD6DF40914BC38271CC70D96733C19B52",
            "assert_indirect_indexing": True,
            "autotune_local_cache": False,
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
    def triton_per_fused_add_mean_mul_reciprocal_sqrt_sub_unsqueeze_0(
        in_out_ptr0,
        in_ptr0,
        in_ptr1,
        in_ptr2,
        in_ptr3,
        in_ptr4,
        xnumel,
        r0_numel,
        XBLOCK: tl.constexpr,
    ):
        xnumel = 344064
        r0_numel = 49
        R0_BLOCK: tl.constexpr = 64
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
        r0_index = tl.arange(0, R0_BLOCK)[None, :]
        r0_offset = 0
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        x3 = xindex
        x0 = xindex % 672
        tmp0 = tl.load(in_ptr0 + (r0_2 + 49 * x3), r0_mask, eviction_policy="evict_first", other=0.0)
        tmp1 = tl.load(in_ptr1 + x0, None, eviction_policy="evict_last")
        tmp3 = tl.load(in_ptr2 + x0, None, eviction_policy="evict_last")
        tmp11 = tl.load(in_ptr3 + x0, None, eviction_policy="evict_last")
        tmp13 = tl.load(in_ptr4 + x0, None, eviction_policy="evict_last")
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 1.0e-5, tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1.0, tl.float32)
        tmp8 = tmp7 / tmp6
        tmp9 = tmp8 * tmp7
        tmp10 = tmp2 * tmp9
        tmp12 = tmp10 * tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
        tmp17 = tl.where(r0_mask, tmp15, 0)
        tmp18 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
        tmp19 = tl.full([1, 1], 49.0, tl.float32)
        tmp20 = tmp18 / tmp19
        tl.store(in_out_ptr0 + x3, tmp20, None)

    triton_per_fused_add_mean_mul_reciprocal_sqrt_sub_unsqueeze_0.configs = [
        triton.Config({"XBLOCK": 32}, num_warps=2, num_stages=1)
    ]


def _expect_f32_cuda_1d(name: str, value: Any, size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != (size,):
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(size,)}")
    if tuple(value.stride()) != (1,):
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected (1,)")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, conv, variance, weight, bias = inputs
    if not isinstance(conv, torch.Tensor):
        raise TypeError(f"convolution_66 must be a tensor, got {type(conv)!r}")
    if conv.ndim != 4:
        raise ValueError(f"convolution_66 must be 4D, got shape {tuple(conv.shape)}")
    if tuple(conv.shape) != DEFAULT_CONV_SHAPE:
        raise ValueError(
            f"convolution_66 has shape {tuple(conv.shape)}, expected {DEFAULT_CONV_SHAPE}"
        )
    if tuple(conv.stride()) != DEFAULT_CONV_STRIDE:
        raise ValueError(
            f"convolution_66 has stride {tuple(conv.stride())}, expected {DEFAULT_CONV_STRIDE}"
        )
    if conv.dtype != torch.float32:
        raise TypeError(f"convolution_66 has dtype {conv.dtype}, expected torch.float32")
    if not conv.is_cuda:
        raise RuntimeError("convolution_66 must be a CUDA tensor for this Triton oracle")

    mean_t = _expect_f32_cuda_1d("arg308_1", mean, DEFAULT_CHANNELS)
    variance_t = _expect_f32_cuda_1d("arg309_1", variance, DEFAULT_CHANNELS)
    weight_t = _expect_f32_cuda_1d("arg310_1", weight, DEFAULT_CHANNELS)
    bias_t = _expect_f32_cuda_1d("arg311_1", bias, DEFAULT_CHANNELS)
    tensors = (mean_t, conv, variance_t, weight_t, bias_t)
    if any(t.device != conv.device for t in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, conv, variance_t, weight_t, bias_t


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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_affine_spatial_mean.py")

    mean, conv, variance, weight, bias = _validate_inputs(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=conv.device,
        dtype=torch.float32,
    )
    triton_per_fused_add_mean_mul_reciprocal_sqrt_sub_unsqueeze_0.run(
        output,
        conv,
        mean,
        variance,
        weight,
        bias,
        DEFAULT_ROWS,
        DEFAULT_HW,
        stream=get_raw_stream(conv.device.index if conv.device.index is not None else 0),
    )
    return output


def _check_oracle_with_equal_nan(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Check values for this deterministic repro while requiring matching NaN masks."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False
    if eager.shape != oracle_out.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} "
            f"eager={list(eager.shape)}"
        )
        return False
    if eager.stride() != oracle_out.stride():
        print(
            f"  output 0: SCOPE_MISMATCH stride oracle={list(oracle_out.stride())} "
            f"eager={list(eager.stride())}"
        )
        return False
    if eager.dtype != oracle_out.dtype:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        return False

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_mask_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~eager_nan & ~oracle_nan
    if finite.any():
        finite_eager = eager_f32[finite]
        finite_oracle = oracle_f32[finite]
        max_diff = (finite_eager - finite_oracle).abs().max().item()
        values_ok = torch.allclose(finite_eager, finite_oracle, atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_mask_ok and values_ok
    status = "PASS" if ok else "FAIL"
    print(
        f"  output 0: {status} (shape={list(eager.shape)} dtype={eager.dtype} "
        f"stride={list(eager.stride())} finite_max_diff={max_diff:.2e} "
        f"nan_count={int(eager_nan.sum().item())})"
    )
    if not nan_mask_ok:
        mismatched = torch.count_nonzero(eager_nan != oracle_nan).item()
        print(f"    NaN mask mismatch count={mismatched}")
    return ok


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
        ok = _check_oracle_with_equal_nan(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
