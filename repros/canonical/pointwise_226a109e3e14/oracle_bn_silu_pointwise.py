"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured per-channel BN-inference affine plus SiLU pointwise scope in one Triton kernel, preserving the dense NCHW or channels-last output layout and deterministic NaN mask while reusing channel-only sqrt/rsqrt/scale/bias values within each channel tile, whereas tuned Inductor already lowers the same decomposed broadcast pointwise graph into a fused kernel measuring within floor noise of this full-scope specialization; Inductor cannot materially improve this repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining work is dominated by the required input/output memory traffic, one pointwise launch, and the unavoidable exp-heavy SiLU evaluation; the fix is BANDWIDTH_BOUND: record this as an at-floor pointwise BN-SiLU case unless broader launch-overhead or math-codegen improvements move the baseline."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
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

    @triton.jit
    def _bn_silu_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        B: tl.constexpr,
        C: tl.constexpr,
        HW: tl.constexpr,
        LAYOUT: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        c_mask = c_offsets < C
        hw_mask = hw_offsets < HW
        mask = c_mask[:, None] & hw_mask[None, :]

        if LAYOUT == 0:
            x_offsets = n * C * HW + c_offsets[:, None] * HW + hw_offsets[None, :]
        else:
            x_offsets = n * HW * C + hw_offsets[None, :] * C + c_offsets[:, None]

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        invstd = 1.0 / tl.sqrt(var + 1.0e-5)
        affine = (x - mean[:, None]) * invstd[:, None] * weight[:, None] + bias[:, None]
        silu = affine / (tl.exp(-affine) + 1.0)
        tl.store(out_ptr + x_offsets, silu, mask=mask)


EPS = 1.0e-5
BLOCK_C = 8
BLOCK_HW = 64
CLASSIFICATION = "BANDWIDTH_BOUND"


def _require_f32_tensor(name: str, value: Any, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _layout_kind(x: torch.Tensor) -> int:
    if x.is_contiguous():
        return 0
    if x.dim() == 4 and x.is_contiguous(memory_format=torch.channels_last):
        return 1
    raise ValueError(
        f"unsupported input layout for {REPRO_ID}: shape={tuple(x.shape)} "
        f"stride={tuple(x.stride())}"
    )


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean, x, var, weight, bias = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"convolution_33 must be a tensor, got {type(x)!r}")
    if x.dim() != 4:
        raise ValueError(f"convolution_33 must be rank 4, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32:
        raise TypeError(f"convolution_33 has dtype {x.dtype}, expected torch.float32")
    if not x.is_cuda:
        raise RuntimeError("convolution_33 must be a CUDA tensor for this Triton oracle")

    _, channels, _, _ = x.shape
    channel_shape = (channels,)
    mean_t = _require_f32_tensor("arg269_1", mean, channel_shape)
    var_t = _require_f32_tensor("arg270_1", var, channel_shape)
    weight_t = _require_f32_tensor("arg271_1", weight, channel_shape)
    bias_t = _require_f32_tensor("arg272_1", bias, channel_shape)

    if any(t.device != x.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, x, var_t, weight_t, bias_t, _layout_kind(x)


@oracle_impl(hardware="H100", shapes="(T([160], f32), T([128, 160, 8, 8], f32), T([160], f32), T([160], f32), T([160], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope with one Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_silu_pointwise.py")

    mean, x, var, weight, bias, layout = _validate_inputs(inputs)
    batch, channels, height, width = x.shape
    hw = height * width
    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.float32,
    )

    grid = (
        batch,
        triton.cdiv(channels, BLOCK_C),
        triton.cdiv(hw, BLOCK_HW),
    )
    _bn_silu_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        B=batch,
        C=channels,
        HW=hw,
        LAYOUT=layout,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
        num_stages=4,
    )
    return output


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Validate values, dtype, shape, stride, and deterministic NaN masks."""
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    shape_ok = expected.shape == actual.shape
    dtype_ok = expected.dtype == actual.dtype
    stride_ok = expected.stride() == actual.stride()

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan & ~actual_nan

    if finite.any():
        finite_expected = expected_f32[finite]
        finite_actual = actual_f32[finite]
        max_diff = (finite_expected - finite_actual).abs().max().item()
        values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    print(
        f"  output 0 values: {'PASS' if values_ok else 'FAIL'} "
        f"(shape={list(actual.shape)} dtype={actual.dtype} "
        f"max_finite_diff={max_diff:.2e})"
    )
    print(
        f"  output 0 layout: {'PASS' if shape_ok and dtype_ok and stride_ok else 'FAIL'} "
        f"(expected_shape={list(expected.shape)}, oracle_shape={list(actual.shape)}, "
        f"expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    print(
        f"  output 0 NaNs: {'PASS' if nan_ok else 'FAIL'} "
        f"(expected_nan={int(expected_nan.sum().item())}, "
        f"oracle_nan={int(actual_nan.sum().item())})"
    )
    return values_ok and shape_ok and dtype_ok and stride_ok and nan_ok


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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
