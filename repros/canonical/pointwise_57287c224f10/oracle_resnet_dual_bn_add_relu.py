"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full ResNet152 inference dual BN-affine residual epilogue in one channel-tiled Triton kernel, including fp32 mean/variance normalization for both branches, fp32 gamma/beta affine, the required per-branch fp16 rounding points, fp16 branch add, ReLU, and final contiguous fp16 [32,2048,7,7] output, whereas Inductor emits a single generic fused pointwise kernel for the same scope but does not form a channel-tiled BN epilogue that reuses per-channel scalar normalization work across the 7x7 spatial block; Inductor cannot do this today because its pointwise scheduler/codegen lacks a dual-branch BN-inference affine template or loop-invariant scalar hoisting strategy for NCHW per-channel parameters with required fp16 cast boundaries; the fix is NEW_PATTERN: add a channel-tiled dual BN-affine add/ReLU epilogue that computes each branch's per-channel invstd/affine scalars once per row tile while preserving the fp16 rounding and add semantics."""
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
    get_shape_key,
    has_stochastic_ops,
)

BATCH = 32
CHANNELS = 2048
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ROWS = BATCH * CHANNELS
EPS = 1.0e-5
OUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
BLOCK_ROWS = 8
BLOCK_HW = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def oracle_kernel(
        mean0_ptr,
        conv0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        conv1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        out_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        row_mask = row_offsets < rows
        hw_mask = hw_offsets < hw
        mask = row_mask[:, None] & hw_mask[None, :]

        channel_offsets = row_offsets % channels
        element_offsets = row_offsets[:, None] * hw + hw_offsets[None, :]

        mean0 = tl.load(mean0_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)

        mean1 = tl.load(mean1_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)

        conv0 = tl.load(conv0_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32)
        conv1 = tl.load(conv1_ptr + element_offsets, mask=mask, other=0.0).to(tl.float32)

        branch0 = (conv0 - mean0[:, None]) * tl.rsqrt(var0[:, None] + eps) * weight0[:, None] + bias0[:, None]
        branch1 = (conv1 - mean1[:, None]) * tl.rsqrt(var1[:, None] + eps) * weight1[:, None] + bias1[:, None]

        branch0_fp16 = branch0.to(tl.float16)
        branch1_fp16 = branch1.to(tl.float16)
        added_fp16 = (branch0_fp16 + branch1_fp16).to(tl.float16)
        relu = tl.where(added_fp16 != added_fp16, added_fp16, tl.maximum(added_fp16, 0.0))
        tl.store(out_ptr + element_offsets, relu, mask=mask)


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
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_resnet_dual_bn_add_relu.py")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    names = (
        "arg737_1",
        "convolution_147",
        "arg738_1",
        "arg739_1",
        "arg740_1",
        "arg742_1",
        "convolution_148",
        "arg743_1",
        "arg744_1",
        "arg745_1",
    )
    channel_shape = (CHANNELS,)
    channel_stride = (1,)

    tensors = (
        _require_f16_tensor(names[0], inputs[0], channel_shape, channel_stride),
        _require_f16_tensor(names[1], inputs[1], OUT_SHAPE, OUT_STRIDE),
        _require_f16_tensor(names[2], inputs[2], channel_shape, channel_stride),
        _require_f16_tensor(names[3], inputs[3], channel_shape, channel_stride),
        _require_f16_tensor(names[4], inputs[4], channel_shape, channel_stride),
        _require_f16_tensor(names[5], inputs[5], channel_shape, channel_stride),
        _require_f16_tensor(names[6], inputs[6], OUT_SHAPE, OUT_STRIDE),
        _require_f16_tensor(names[7], inputs[7], channel_shape, channel_stride),
        _require_f16_tensor(names[8], inputs[8], channel_shape, channel_stride),
        _require_f16_tensor(names[9], inputs[9], channel_shape, channel_stride),
    )

    device = tensors[1].device
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return tensors


@oracle_impl(hardware="H100", shapes="(T([2048], f16), T([32, 2048, 7, 7], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([2048], f16), T([32, 2048, 7, 7], f16), T([2048], f16), T([2048], f16), T([2048], f16))")
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
    (
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)

    output = torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=conv0.device, dtype=torch.float16)
    oracle_kernel[(triton.cdiv(ROWS, BLOCK_ROWS),)](
        mean0,
        conv0,
        var0,
        weight0,
        bias0,
        mean1,
        conv1,
        var1,
        weight1,
        bias1,
        output,
        rows=ROWS,
        channels=CHANNELS,
        hw=HW,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
        num_warps=8,
        num_stages=3,
    )
    return output


def _check_oracle_nan_equal(
    instance: torch.nn.Module,
    inputs: list[Any],
    *,
    atol: float,
    rtol: float,
) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(eager, torch.Tensor) or not isinstance(oracle_out, torch.Tensor):
        print("  SCOPE_MISMATCH: eager and oracle outputs must both be tensors")
        return False
    if tuple(oracle_out.shape) != OUT_SHAPE or tuple(eager.shape) != OUT_SHAPE:
        print(f"  output 0: SCOPE_MISMATCH shape oracle={list(oracle_out.shape)} eager={list(eager.shape)}")
        return False
    if oracle_out.dtype is not torch.float16 or eager.dtype is not torch.float16:
        print(f"  output 0: SCOPE_MISMATCH dtype oracle={oracle_out.dtype} eager={eager.dtype}")
        return False
    if tuple(oracle_out.stride()) != OUT_STRIDE or tuple(eager.stride()) != OUT_STRIDE:
        print(f"  output 0: SCOPE_MISMATCH stride oracle={oracle_out.stride()} eager={eager.stride()}")
        return False

    eager_f32 = eager.float()
    oracle_f32 = oracle_out.float()
    eager_nan = torch.isnan(eager_f32)
    oracle_nan = torch.isnan(oracle_f32)
    nan_ok = torch.equal(eager_nan, oracle_nan)
    finite = ~(eager_nan | oracle_nan)
    if finite.any():
        max_diff = (eager_f32[finite] - oracle_f32[finite]).abs().max().item()
        values_ok = torch.allclose(eager_f32[finite], oracle_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(eager.shape)} dtype={eager.dtype} stride={eager.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(eager_nan.sum().item())})"
    )
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
        ok = _check_oracle_nan_equal(instance, inputs, atol=args.atol, rtol=args.rtol)
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
