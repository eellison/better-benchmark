"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full dual RegNet BN-inference affine, add, ReLU, and spatial mean scope for the two fp16 [32,2240,7,7] inputs in one reduction kernel using a folded rsqrt affine form and stores the final fp16 [32,2240] output, whereas Inductor emits a fused reduction but repeats sqrt followed by reciprocal/divide for both batch-invariant variance terms in every batch-channel row; Inductor cannot do this today because its algebraic simplifier and reduction codegen do not canonicalize the sqrt-plus-reciprocal BN inference pattern to a reusable rsqrt affine expression inside reduction inputs; the fix is ALGEBRAIC_ELIMINATION: canonicalize inference BN variance normalization to rsqrt-based scale/shift arithmetic before lowering the downstream fused reduction."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "ALGEBRAIC_ELIMINATION"
ACTIONABLE = True
TRUE_FLOOR = True

BATCH = 32
CHANNELS = 2240
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_ROWS = 8
BLOCK_HW = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _dual_bn_add_relu_spatial_mean_direct_kernel(
        mean0_ptr,
        x0_ptr,
        var0_ptr,
        weight0_ptr,
        bias0_ptr,
        mean1_ptr,
        x1_ptr,
        var1_ptr,
        weight1_ptr,
        bias1_ptr,
        out_ptr,
        total_rows,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        channel_offsets = row_offsets - (row_offsets // 2240) * 2240
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_rows = row_offsets < total_rows
        valid_hw = hw_offsets < 49
        valid = valid_rows[:, None] & valid_hw[None, :]

        x_offsets = row_offsets[:, None] * 49 + hw_offsets[None, :]
        x0 = tl.load(x0_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)

        mean0 = tl.load(mean0_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var0 = tl.load(var0_ptr + channel_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        var1 = tl.load(var1_ptr + channel_offsets, mask=valid_rows, other=1.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel_offsets, mask=valid_rows, other=0.0).to(tl.float32)

        y0 = (x0 - mean0[:, None]) * tl.rsqrt(var0[:, None] + eps) * weight0[:, None] + bias0[:, None]
        y1 = (x1 - mean1[:, None]) * tl.rsqrt(var1[:, None] + eps) * weight1[:, None] + bias1[:, None]
        summed = y0 + y1
        zero = tl.full((BLOCK_ROWS_, BLOCK_HW_), 0.0, tl.float32)
        relu = tl.where(summed < zero, zero, summed)
        reduced = tl.sum(tl.where(valid, relu, 0.0), axis=1) * (1.0 / 49.0)

        tl.store(out_ptr + row_offsets, reduced, mask=valid_rows)


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
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1, view_shape = inputs
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    image_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    image_stride = (CHANNELS * HW, HW, WIDTH, 1)

    mean0_t = _require_f16_tensor("arg378_1", mean0, vector_shape, vector_stride)
    x0_t = _require_f16_tensor("convolution_98", x0, image_shape, image_stride)
    var0_t = _require_f16_tensor("arg379_1", var0, vector_shape, vector_stride)
    weight0_t = _require_f16_tensor("arg380_1", weight0, vector_shape, vector_stride)
    bias0_t = _require_f16_tensor("arg381_1", bias0, vector_shape, vector_stride)
    mean1_t = _require_f16_tensor("arg383_1", mean1, vector_shape, vector_stride)
    x1_t = _require_f16_tensor("convolution_99", x1, image_shape, image_stride)
    var1_t = _require_f16_tensor("arg384_1", var1, vector_shape, vector_stride)
    weight1_t = _require_f16_tensor("arg385_1", weight1, vector_shape, vector_stride)
    bias1_t = _require_f16_tensor("arg386_1", bias1, vector_shape, vector_stride)

    if list(view_shape) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")

    device = x0_t.device
    if any(
        t.device != device
        for t in (mean0_t, var0_t, weight0_t, bias0_t, mean1_t, x1_t, var1_t, weight1_t, bias1_t)
    ):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean0_t, x0_t, var0_t, weight0_t, bias0_t, mean1_t, x1_t, var1_t, weight1_t, bias1_t


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_bn_relu_spatial_mean.py")

    mean0, x0, var0, weight0, bias0, mean1, x1, var1, weight1, bias1 = _validate_inputs(inputs)
    output = torch.empty_strided((BATCH, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float16)
    _dual_bn_add_relu_spatial_mean_direct_kernel[(triton.cdiv(BATCH * CHANNELS, BLOCK_ROWS),)](
        mean0,
        x0,
        var0,
        weight0,
        bias0,
        mean1,
        x1,
        var1,
        weight1,
        bias1,
        output,
        BATCH * CHANNELS,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
        num_warps=1,
        num_stages=3,
    )
    return output


def _run_check(instance: torch.nn.Module, inputs: list[Any], *, atol: float, rtol: float) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False
    if expected.shape != actual.shape:
        print(
            f"  output 0: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
            f"eager={list(expected.shape)}"
        )
        return False
    if expected.dtype != actual.dtype:
        print(f"  output 0: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
    if expected.stride() != actual.stride():
        print(
            f"  output 0: SCOPE_MISMATCH stride oracle={actual.stride()} "
            f"eager={expected.stride()}"
        )
        return False

    expected_f32 = expected.float()
    actual_f32 = actual.float()
    expected_nan = torch.isnan(expected_f32)
    actual_nan = torch.isnan(actual_f32)
    nan_mask_ok = torch.equal(expected_nan, actual_nan)
    finite = ~expected_nan
    if finite.any():
        max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
        values_ok = torch.allclose(expected_f32[finite], actual_f32[finite], atol=atol, rtol=rtol)
    else:
        max_diff = 0.0
        values_ok = True

    ok = nan_mask_ok and values_ok
    print(
        f"  output 0: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
        f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
