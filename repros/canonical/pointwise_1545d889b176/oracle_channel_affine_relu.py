"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete batchnorm-style channel affine, residual add, and ReLU scope with a channel/NHW-tiled Triton kernel that folds the affine into `x * scale + shifted_bias + residual` and writes the final contiguous NCHW output directly, whereas tuned Inductor already measures at the same CUDAGraph replay floor for this isolated fused pointwise graph; Inductor cannot be assigned a material local scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, or recompute gap here because the required two activation reads, output write, channel parameter reads, one launch, and f32 pointwise math dominate and the custom parameter-hoisting tradeoff does not beat the generic contiguous kernel; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise launch, memory-traffic, or math-codegen work moves both implementations."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle as _harness_check_oracle,
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2048}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 4096}, num_warps=8, num_stages=3),
        ],
        key=["NHW", "HW"],
    )
    @triton.jit
    def _channel_affine_relu_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        output_ptr,
        C: tl.constexpr,
        HW: tl.constexpr,
        NHW: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        c = tl.program_id(0)
        block_m = tl.program_id(1)
        m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = m < NHW

        n = m // HW
        hw = m - n * HW
        offsets = n * C * HW + c * HW + hw

        mean = tl.load(mean_ptr + c).to(tl.float32)
        var = tl.load(var_ptr + c).to(tl.float32)
        weight = tl.load(weight_ptr + c).to(tl.float32)
        bias = tl.load(bias_ptr + c).to(tl.float32)
        scale = weight * tl.rsqrt(var + 1.0e-5)
        shifted_bias = bias - mean * scale

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        value = x * scale + shifted_bias + residual
        value = tl.maximum(value, 0.0, propagate_nan=tl.PropagateNan.ALL)
        tl.store(output_ptr + offsets, value, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_channel_affine_relu.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, residual = inputs
    tensors = (
        ("arg77_1", mean),
        ("convolution_15", x),
        ("arg78_1", var),
        ("arg79_1", weight),
        ("arg80_1", bias),
        ("relu_10", residual),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor)!r}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must be f32, got {tensor.dtype}")

    if x.dim() != 4:
        raise ValueError(f"convolution_15 must be rank-4 NCHW, got shape={tuple(x.shape)}")
    if residual.shape != x.shape:
        raise ValueError(
            f"relu_10 shape must match convolution_15, got {tuple(residual.shape)} "
            f"vs {tuple(x.shape)}"
        )
    if not x.is_contiguous():
        raise ValueError(f"convolution_15 must be contiguous, got stride={tuple(x.stride())}")
    if not residual.is_contiguous():
        raise ValueError(f"relu_10 must be contiguous, got stride={tuple(residual.stride())}")

    n, c, h, w = (int(dim) for dim in x.shape)
    expected_channel_shape = (c,)
    for name, tensor in (("arg77_1", mean), ("arg78_1", var),
                         ("arg79_1", weight), ("arg80_1", bias)):
        if tuple(tensor.shape) != expected_channel_shape:
            raise ValueError(
                f"{name} shape must be {expected_channel_shape}, got {tuple(tensor.shape)}"
            )
        if tuple(tensor.stride()) != (1,):
            raise ValueError(f"{name} must be contiguous 1D, got stride={tuple(tensor.stride())}")

    return mean, x, var, weight, bias, residual, n, c, h * w


@oracle_impl(hardware="H100", shapes="(T([256], f32), T([1024, 256, 8, 8], f32), T([256], f32), T([256], f32), T([256], f32), T([1024, 256, 8, 8], f32))")
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
    mean, x, var, weight, bias, residual, n, c, hw = _validate_inputs(inputs)
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    nhw = n * hw
    grid = lambda meta: (c, triton.cdiv(nhw, meta["BLOCK_M"]))
    _channel_affine_relu_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        output,
        C=c,
        HW=hw,
        NHW=nhw,
    )
    return output


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_nan_equal(
    oracle_forward_fn,
    instance,
    inputs,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    skip_stochastic: bool = True,
) -> bool:
    del skip_stochastic
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward_fn(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={tuple(actual.stride())} "
                f"eager={tuple(expected.stride())}"
            )
            all_pass = False
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        nan_ok = torch.equal(expected_nan, actual_nan)
        finite = ~expected_nan & ~actual_nan
        if finite.any():
            finite_diff = (expected_f32[finite] - actual_f32[finite]).abs()
            max_diff = finite_diff.max().item()
            values_ok = torch.allclose(
                expected_f32[finite],
                actual_f32[finite],
                atol=atol,
                rtol=rtol,
            )
        else:
            max_diff = 0.0
            values_ok = True

        ok = bool(nan_ok and values_ok)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
            f"max_finite_diff={max_diff:.2e})"
        )
        all_pass = all_pass and ok

    return all_pass


check_oracle = _check_oracle_nan_equal
del _harness_check_oracle


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
