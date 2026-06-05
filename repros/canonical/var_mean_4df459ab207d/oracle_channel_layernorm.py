"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ConvNeXtV2 channel-LayerNorm scope in one Triton kernel, including the NCHW input add, per-[N,H,W] population var_mean over 320 channels, eps=1e-6 affine epilogue, direct final NCHW output stores, and the live rsqrt/320 side output, but tiles 16 adjacent spatial rows together so the physically strided channel loads/stores become coalesced row groups; Inductor does not reach this today because its norm-template canonicalization treats the permute-to-NHWC LayerNorm as a generic row reduction over a non-inner channel dimension and does not select a spatial-row grouped lowering that writes the final NCHW layout directly; the fix is NEW_PATTERN: add a ConvNeXt-style NCHW channel-LayerNorm template or layout-aware tiling rule that groups neighboring spatial rows, keeps the add/statistics/affine/side-output scope fused, and sinks the inverse permute into the store plan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

EPS = 1.0e-6


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _channel_layernorm_kernel(
        x0_ptr,
        x1_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        inv_out_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        cols = tl.arange(0, BLOCK_C)[None, :]
        row_mask = row_offsets < rows
        col_mask = cols < channels
        mask = row_mask & col_mask

        n_index = row_offsets // hw
        spatial_index = row_offsets - n_index * hw
        nchw_offsets = n_index * channels * hw + cols * hw + spatial_index

        x0 = tl.load(x0_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + nchw_offsets, mask=mask, other=0.0).to(tl.float32)
        x = x0 + x1
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1)[:, None] / channels
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / channels
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + nchw_offsets, y, mask=mask)
        tl.store(inv_out_ptr + row_offsets, invstd / channels, mask=row_mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]:
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    x0, x1, weight, bias = inputs
    tensor_inputs = (x0, x1, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("all repro inputs must be tensors")

    if x0.ndim != 4:
        raise ValueError(f"input 0 must be rank 4, got shape {tuple(x0.shape)}")
    n, channels, h, w = (int(dim) for dim in x0.shape)
    expected_shapes = ((n, channels, h, w), (n, channels, h, w), (channels,), (channels,))

    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    if (n, channels, h, w) != (128, 320, 14, 14):
        raise ValueError(f"oracle is specialized for (128, 320, 14, 14), got {(n, channels, h, w)}")

    return x0, x1, weight, bias, n, channels, h, w


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with one Triton LayerNorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same two outputs: contiguous float32[N,C,H,W] and float32[N,H,W,1] invstd/C.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_channel_layernorm.py")

    x0, x1, weight, bias, n, channels, h, w = _validate_inputs(inputs)
    out = torch.empty_strided(
        (n, channels, h, w),
        (channels * h * w, h * w, w, 1),
        device=x0.device,
        dtype=x0.dtype,
    )
    inv_out = torch.empty_strided(
        (n, h, w, 1),
        (h * w, w, 1, 1),
        device=x0.device,
        dtype=x0.dtype,
    )

    rows = n * h * w
    xblock = 16
    _channel_layernorm_kernel[(triton.cdiv(rows, xblock),)](
        x0,
        x1,
        weight,
        bias,
        out,
        inv_out,
        rows=rows,
        channels=channels,
        hw=h * w,
        eps=EPS,
        BLOCK_C=triton.next_power_of_2(channels),
        XBLOCK=xblock,
        num_warps=8,
    )
    return (out, inv_out)


def _check_layout_and_diff(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if any(isinstance(value, torch.Tensor) and value.is_cuda for value in actual):
            torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        max_diff = (expected_tensor - actual_tensor).abs().max().item()
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        ok = ok and layout_ok
        print(
            f"  output {index}: shape={tuple(actual_tensor.shape)} "
            f"dtype={actual_tensor.dtype} stride={actual_tensor.stride()} "
            f"max_diff={max_diff:.2e} layout={'PASS' if layout_ok else 'FAIL'}"
        )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
        ok = _check_layout_and_diff(instance, inputs) and ok
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
