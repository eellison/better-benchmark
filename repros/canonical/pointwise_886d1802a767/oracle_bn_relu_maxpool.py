"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete fp32 per-channel affine normalization, ReLU, and padding-1 3x3 stride-2 low-memory maxpool-with-offsets in one Triton kernel that writes only the final f32 pooled values and int8 offsets, whereas Inductor currently materializes the full normalized ReLU activation before running a separate multi-output pooling stencil; Inductor cannot do this today because scheduler fusion does not sink broadcast affine/ReLU producers through prims low-memory maxpool-with-offsets while preserving padded-window offset, finite tie, and NaN selection semantics; the fix is SCHEDULER_FUSION: allow pointwise affine/ReLU producers to be inlined into low-memory maxpool-with-offsets and emit values plus offsets from the same loop nest."""
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


EPS = 1.0e-5
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
BLOCK_C = 4
BLOCK_OUT = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _relu_preserve_nan(x):
        return tl.where((x > 0.0) | (x != x), x, 0.0)


    @triton.jit
    def _bn_relu_maxpool_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        values_ptr,
        offsets_ptr,
        CHANNELS: tl.constexpr,
        HEIGHT: tl.constexpr,
        WIDTH: tl.constexpr,
        OUT_HEIGHT: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_OUT_: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        out_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_offsets = out_block * BLOCK_OUT_ + tl.arange(0, BLOCK_OUT_)
        out_h = out_offsets // OUT_WIDTH
        out_w = out_offsets - out_h * OUT_WIDTH

        c_mask = c_offsets < CHANNELS
        out_mask = out_offsets < (OUT_HEIGHT * OUT_WIDTH)
        mask = c_mask[:, None] & out_mask[None, :]

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        scale = (1.0 / tl.sqrt(var + 1.0e-5)) * weight

        best = tl.full((BLOCK_C_, BLOCK_OUT_), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C_, BLOCK_OUT_), tl.int32)
        input_base = (n * CHANNELS + c_offsets[:, None]) * (HEIGHT * WIDTH)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh - 1
            valid_h = (in_h >= 0) & (in_h < HEIGHT)
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw - 1
                valid = mask & valid_h[None, :] & (in_w[None, :] >= 0) & (in_w[None, :] < WIDTH)
                x = tl.load(
                    conv_ptr + input_base + in_h[None, :] * WIDTH + in_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                y = _relu_preserve_nan((x - mean[:, None]) * scale[:, None] + bias[:, None])

                take = valid & ((y > best) | (y != y))
                best = tl.where(take, y, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        output_base = (n * CHANNELS + c_offsets[:, None]) * (OUT_HEIGHT * OUT_WIDTH)
        tl.store(values_ptr + output_base + out_offsets[None, :], best, mask=mask)
        tl.store(offsets_ptr + output_base + out_offsets[None, :], best_offset.to(tl.int8), mask=mask)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    conv = _require_f32_tensor("convolution", inputs[1])
    if conv.ndim != 4:
        raise ValueError(f"convolution must be rank 4, got shape {tuple(conv.shape)}")
    if not conv.is_contiguous():
        raise ValueError(f"convolution must be contiguous NCHW, got stride {conv.stride()}")

    batch, channels, height, width = conv.shape
    if batch <= 0 or channels <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"invalid convolution shape {tuple(conv.shape)}")

    mean = _require_f32_tensor("arg2_1", inputs[0], (channels,), (1,))
    var = _require_f32_tensor("arg3_1", inputs[2], (channels,), (1,))
    weight = _require_f32_tensor("arg4_1", inputs[3], (channels,), (1,))
    bias = _require_f32_tensor("arg5_1", inputs[4], (channels,), (1,))

    device = conv.device
    if any(tensor.device != device for tensor in (mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return mean, conv, var, weight, bias


def _torch_oracle(
    mean: torch.Tensor,
    conv: torch.Tensor,
    var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_std = torch.reciprocal(torch.sqrt(var + EPS))
    y = (conv - mean[None, :, None, None]) * inv_std[None, :, None, None]
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    y = torch.relu(y)
    return torch.ops.prims._low_memory_max_pool_with_offsets.default(
        y, [KERNEL_SIZE, KERNEL_SIZE], [STRIDE, STRIDE], [PADDING, PADDING], [1, 1], False
    )


@oracle_impl(hardware="H100", shapes="(T([64], f32), T([1, 64, 512, 512], f32), T([64], f32), T([64], f32), T([64], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full affine normalization, ReLU, and maxpool-with-offsets scope."""
    mean, conv, var, weight, bias = _validate_inputs(inputs)
    batch, channels, height, width = conv.shape
    out_height = (height + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_width = (width + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_shape = (batch, channels, out_height, out_width)
    out_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    if not conv.is_cuda:
        return _torch_oracle(mean, conv, var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    values = torch.empty_strided(out_shape, out_stride, device=conv.device, dtype=torch.float32)
    offsets = torch.empty_strided(out_shape, out_stride, device=conv.device, dtype=torch.int8)

    grid = (
        batch,
        triton.cdiv(channels, BLOCK_C),
        triton.cdiv(out_height * out_width, BLOCK_OUT),
    )
    _bn_relu_maxpool_kernel[grid](
        mean,
        conv,
        var,
        weight,
        bias,
        values,
        offsets,
        CHANNELS=channels,
        HEIGHT=height,
        WIDTH=width,
        OUT_HEIGHT=out_height,
        OUT_WIDTH=out_width,
        BLOCK_C_=BLOCK_C,
        BLOCK_OUT_=BLOCK_OUT,
        num_warps=8,
        num_stages=3,
    )
    return values, offsets


def _max_finite_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).abs()
    finite = diff[torch.isfinite(diff)]
    if finite.numel() == 0:
        return 0.0
    return float(finite.max().item())


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Check full scope while treating deterministic NaNs as equal."""
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    if not isinstance(eager, tuple) or not isinstance(oracle_out, tuple):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tuples")
        return False
    if len(eager) != len(oracle_out):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_out)} outputs, "
            f"eager produces {len(eager)}"
        )
        return False

    all_pass = True
    for index, (expected, actual) in enumerate(zip(eager, oracle_out)):
        shape_ok = expected.shape == actual.shape
        dtype_ok = expected.dtype == actual.dtype
        stride_ok = expected.stride() == actual.stride()
        if expected.is_floating_point():
            expected_f32 = expected.float()
            actual_f32 = actual.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            nan_ok = torch.equal(expected_nan, actual_nan)
            finite = ~(expected_nan | actual_nan)
            if finite.any():
                value_ok = torch.allclose(
                    actual_f32[finite],
                    expected_f32[finite],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                value_ok = True
            ok = shape_ok and dtype_ok and stride_ok and nan_ok and value_ok
            print(
                f"  output {index}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
                f"max_finite_diff={_max_finite_diff(actual, expected):.2e} "
                f"nan_count={int(expected_nan.sum().item())})"
            )
        else:
            value_ok = torch.equal(actual, expected)
            mismatch_count = int((actual != expected).sum().item()) if shape_ok else -1
            ok = shape_ok and dtype_ok and stride_ok and value_ok
            print(
                f"  output {index}: {'PASS' if ok else 'FAIL'} "
                f"(exact, shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()} mismatch_count={mismatch_count})"
            )
        all_pass = all_pass and ok
    return all_pass


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
        ok = _check_oracle_nan_equal(
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
