"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete two-branch BatchNorm-inference affine, branch add, and ReLU in one stride-aware Triton kernel while hoisting each channel's scalar normalization parameters across an N/H/W tile, whereas Inductor already fuses the full pointwise graph into a single compiled region and measures at the same CUDAGraph replay floor; Inductor cannot get a material local win here through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the required two activation reads, output write, f32 math, and one launch dominate and the custom channel-scalar tiling does not beat the generic pointwise kernel; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth/codegen work moves both implementations."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 4, "BLOCK_HW": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 256}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=8, num_stages=3),
        ],
        key=[
            "C",
            "H",
            "W",
            "X1_STRIDE_C",
            "X1_STRIDE_H",
            "X1_STRIDE_W",
            "OUT_STRIDE_C",
            "OUT_STRIDE_H",
            "OUT_STRIDE_W",
        ],
    )
    @triton.jit
    def _dual_bn_relu_kernel(
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        mean2_ptr,
        x2_ptr,
        var2_ptr,
        weight2_ptr,
        bias2_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        X1_STRIDE_N: tl.constexpr,
        X1_STRIDE_C: tl.constexpr,
        X1_STRIDE_H: tl.constexpr,
        X1_STRIDE_W: tl.constexpr,
        X2_STRIDE_N: tl.constexpr,
        X2_STRIDE_C: tl.constexpr,
        X2_STRIDE_H: tl.constexpr,
        X2_STRIDE_W: tl.constexpr,
        OUT_STRIDE_N: tl.constexpr,
        OUT_STRIDE_C: tl.constexpr,
        OUT_STRIDE_H: tl.constexpr,
        OUT_STRIDE_W: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
        h_offsets = hw_offsets // W
        w_offsets = hw_offsets - h_offsets * W
        hw_size: tl.constexpr = H * W

        c_mask = c_offsets < C
        hw_mask = hw_offsets < hw_size
        mask = c_mask[:, None] & hw_mask[None, :]

        mean1 = tl.load(mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        var1 = tl.load(var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        mean2 = tl.load(mean2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        var2 = tl.load(var2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        weight2 = tl.load(weight2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]
        bias2 = tl.load(bias2_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)[:, None]

        x1_offsets = (
            n * X1_STRIDE_N
            + c_offsets[:, None] * X1_STRIDE_C
            + h_offsets[None, :] * X1_STRIDE_H
            + w_offsets[None, :] * X1_STRIDE_W
        )
        x2_offsets = (
            n * X2_STRIDE_N
            + c_offsets[:, None] * X2_STRIDE_C
            + h_offsets[None, :] * X2_STRIDE_H
            + w_offsets[None, :] * X2_STRIDE_W
        )
        out_offsets = (
            n * OUT_STRIDE_N
            + c_offsets[:, None] * OUT_STRIDE_C
            + h_offsets[None, :] * OUT_STRIDE_H
            + w_offsets[None, :] * OUT_STRIDE_W
        )

        x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + x2_offsets, mask=mask, other=0.0).to(tl.float32)

        inv1 = 1.0 / tl.sqrt(var1 + 1.0e-5)
        inv2 = 1.0 / tl.sqrt(var2 + 1.0e-5)
        branch1 = (x1 - mean1) * inv1
        branch1 = branch1 * weight1 + bias1
        branch2 = (x2 - mean2) * inv2
        branch2 = branch2 * weight2 + bias2
        value = branch1 + branch2
        value = tl.where(value != value, value, tl.maximum(value, 0.0))

        tl.store(out_ptr + out_offsets, value, mask=mask)


def _require_cuda_f32(name, value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")
    if value.device.type != "cuda":
        raise RuntimeError(f"{name} must be a CUDA tensor")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    return value


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    mean1 = _require_cuda_f32("arg88_1", inputs[0])
    x1 = _require_cuda_f32("convolution_14", inputs[1])
    var1 = _require_cuda_f32("arg89_1", inputs[2])
    weight1 = _require_cuda_f32("arg90_1", inputs[3])
    bias1 = _require_cuda_f32("arg91_1", inputs[4])
    mean2 = _require_cuda_f32("arg93_1", inputs[5])
    x2 = _require_cuda_f32("convolution_15", inputs[6])
    var2 = _require_cuda_f32("arg94_1", inputs[7])
    weight2 = _require_cuda_f32("arg95_1", inputs[8])
    bias2 = _require_cuda_f32("arg96_1", inputs[9])

    if x1.dim() != 4:
        raise ValueError(f"convolution_14 must be rank-4 NCHW, got shape={tuple(x1.shape)}")
    if tuple(x2.shape) != tuple(x1.shape):
        raise ValueError(
            f"convolution_15 shape must match convolution_14, got "
            f"{tuple(x2.shape)} vs {tuple(x1.shape)}"
        )
    if x2.device != x1.device:
        raise RuntimeError("convolution_14 and convolution_15 must be on the same CUDA device")

    _, channels, _, _ = (int(dim) for dim in x1.shape)
    expected_channel_shape = (channels,)
    for name, tensor in (
        ("arg88_1", mean1),
        ("arg89_1", var1),
        ("arg90_1", weight1),
        ("arg91_1", bias1),
        ("arg93_1", mean2),
        ("arg94_1", var2),
        ("arg95_1", weight2),
        ("arg96_1", bias2),
    ):
        if tuple(tensor.shape) != expected_channel_shape:
            raise ValueError(f"{name} shape must be {expected_channel_shape}, got {tuple(tensor.shape)}")
        if tuple(tensor.stride()) != (1,):
            raise ValueError(f"{name} must be contiguous 1D, got stride={tuple(tensor.stride())}")
        if tensor.device != x1.device:
            raise RuntimeError(f"{name} must be on the same CUDA device as convolution_14")

    return mean1, x1, var1, weight1, bias1, mean2, x2, var2, weight2, bias2


@oracle_impl(hardware="H100", shapes="(T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32))")
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
    mean1, x1, var1, weight1, bias1, mean2, x2, var2, weight2, bias2 = _validate_inputs(inputs)
    n, c, h, w = (int(dim) for dim in x1.shape)
    output = torch.empty_strided(
        tuple(x1.shape),
        tuple(x1.stride()),
        device=x1.device,
        dtype=x1.dtype,
    )

    x1_stride_n, x1_stride_c, x1_stride_h, x1_stride_w = (int(s) for s in x1.stride())
    x2_stride_n, x2_stride_c, x2_stride_h, x2_stride_w = (int(s) for s in x2.stride())
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = (int(s) for s in output.stride())

    grid = lambda meta: (
        n,
        triton.cdiv(c, meta["BLOCK_C"]),
        triton.cdiv(h * w, meta["BLOCK_HW"]),
    )
    _dual_bn_relu_kernel[grid](
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        mean2,
        x2,
        var2,
        weight2,
        bias2,
        output,
        C=c,
        H=h,
        W=w,
        X1_STRIDE_N=x1_stride_n,
        X1_STRIDE_C=x1_stride_c,
        X1_STRIDE_H=x1_stride_h,
        X1_STRIDE_W=x1_stride_w,
        X2_STRIDE_N=x2_stride_n,
        X2_STRIDE_C=x2_stride_c,
        X2_STRIDE_H=x2_stride_h,
        X2_STRIDE_W=x2_stride_w,
        OUT_STRIDE_N=out_stride_n,
        OUT_STRIDE_C=out_stride_c,
        OUT_STRIDE_H=out_stride_h,
        OUT_STRIDE_W=out_stride_w,
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
