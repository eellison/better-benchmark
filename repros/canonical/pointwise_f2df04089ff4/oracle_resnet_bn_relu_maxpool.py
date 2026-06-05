"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNet inference BN affine in fp32, fp16 cast, ReLU, and 3x3 stride-2 padding-1 low-memory maxpool-with-offsets in one Triton kernel that returns both final fp16 pooled values and int8 offsets, whereas Inductor currently materializes the full fp16 ReLU activation before running a separate multi-output pooling stencil; Inductor cannot do this today because scheduler fusion does not sink a broadcast pointwise BN/ReLU producer through prims low-memory maxpool-with-offsets while preserving padded-window offset semantics; the fix is SCHEDULER_FUSION: allow affine/ReLU producers to be inlined into low-memory maxpool-with-offsets and emit values plus offsets from the same loop nest."""
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
    def _relu_after_fp16_cast(x):
        x_f16 = x.to(tl.float16)
        return tl.where(x_f16 <= 0.0, 0.0, x_f16.to(tl.float32))


    @triton.jit
    def oracle_kernel(
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
        """Fuse BN affine, fp16 ReLU, and padded maxpool-with-offsets."""
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        out_block = tl.program_id(2)

        c_offsets = c_block * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        out_offsets = out_block * BLOCK_OUT_ + tl.arange(0, BLOCK_OUT_)
        out_h = out_offsets // OUT_WIDTH
        out_w = out_offsets - out_h * OUT_WIDTH

        c_mask = c_offsets < CHANNELS
        out_mask = out_offsets < (OUT_HEIGHT * OUT_WIDTH)

        mean = tl.load(mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=c_mask, other=1.0).to(tl.float32)
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
                valid_w = (in_w >= 0) & (in_w < WIDTH)
                valid = c_mask[:, None] & out_mask[None, :] & valid_h[None, :] & valid_w[None, :]

                x = tl.load(
                    conv_ptr + input_base + in_h[None, :] * WIDTH + in_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)
                y = _relu_after_fp16_cast((x - mean[:, None]) * scale[:, None] + bias[:, None])

                better = valid & ((y > best) | (y != y))
                best = tl.where(better, y, best)
                best_offset = tl.where(better, kh * 3 + kw, best_offset)

        output_base = (n * CHANNELS + c_offsets[:, None]) * (OUT_HEIGHT * OUT_WIDTH)
        store_mask = c_mask[:, None] & out_mask[None, :]
        tl.store(values_ptr + output_base + out_offsets[None, :], best, mask=store_mask)
        tl.store(offsets_ptr + output_base + out_offsets[None, :], best_offset.to(tl.int8), mask=store_mask)


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype is not torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_resnet_bn_relu_maxpool.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    conv = inputs[1]
    if not isinstance(conv, torch.Tensor):
        raise TypeError(f"convolution must be a tensor, got {type(conv)!r}")
    if conv.ndim != 4:
        raise ValueError(f"convolution must be rank 4, got shape {tuple(conv.shape)}")
    batch, channels, height, width = conv.shape
    conv_stride = (channels * height * width, height * width, width, 1)

    mean = _require_f16_tensor("arg2_1", inputs[0], (channels,), (1,))
    conv = _require_f16_tensor("convolution", conv, (batch, channels, height, width), conv_stride)
    var = _require_f16_tensor("arg3_1", inputs[2], (channels,), (1,))
    weight = _require_f16_tensor("arg4_1", inputs[3], (channels,), (1,))
    bias = _require_f16_tensor("arg5_1", inputs[4], (channels,), (1,))

    if any(tensor.device != conv.device for tensor in (mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid spatial shape {(height, width)}")
    return mean, conv, var, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward BN/ReLU/maxpool-with-offsets scope."""
    mean, conv, var, weight, bias = _validate_inputs(inputs)
    batch, channels, height, width = conv.shape
    out_height = (height + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_width = (width + 2 * PADDING - KERNEL_SIZE) // STRIDE + 1
    out_shape = (batch, channels, out_height, out_width)
    out_stride = (channels * out_height * out_width, out_height * out_width, out_width, 1)

    values = torch.empty_strided(out_shape, out_stride, device=conv.device, dtype=torch.float16)
    offsets = torch.empty_strided(out_shape, out_stride, device=conv.device, dtype=torch.int8)

    grid = (
        batch,
        triton.cdiv(channels, BLOCK_C),
        triton.cdiv(out_height * out_width, BLOCK_OUT),
    )
    oracle_kernel[grid](
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
            value_ok = torch.allclose(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=True)
            nan_ok = torch.equal(torch.isnan(actual.float()), torch.isnan(expected.float()))
            max_diff = _max_finite_diff(actual, expected)
            ok = shape_ok and dtype_ok and stride_ok and value_ok and nan_ok
            print(
                f"  output {index}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
                f"max_finite_diff={max_diff:.2e} nan_count={int(torch.isnan(expected.float()).sum().item())})"
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
