"""
Oracle for var_mean_3711a161734e

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured residual-add LayerNorm in one shape-specialized Triton row kernel, preserving the `[512, 768]` to `[1, 512, 768]` view, fp16 residual add before fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-5 affine epilogue, fp16 cast, and the three final `[512, 768]` views aliasing one base buffer, whereas tuned Inductor already lowers this fixed hidden-size normalization region to a comparable fused norm kernel; Inductor cannot materially improve it through a local scheduler or algebraic rewrite because the remaining cost is dominated by mandatory residual/input/affine memory traffic, one row reduction, rsqrt latency, output stores, and launch overhead; the fix is BANDWIDTH_BOUND: record the full-scope at-floor result and only revisit for broad normalization-template or launch-overhead improvements.
"""
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
    oracle_impl,
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

ROWS = 512
HIDDEN = 768
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = 1024
EPS = 1.0e-5


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _residual_layernorm_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        # aten.add on two fp16 tensors rounds before the explicit fp32 convert.
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y.to(tl.float16), mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        addmm_49,
        residual,
        weight,
        bias,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs

    tensor_inputs = (addmm_49, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first four repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, RESIDUAL_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    shape0_tuple = _shape_tuple(shape0)
    shape1_tuple = _shape_tuple(shape1)
    shape2_tuple = _shape_tuple(shape2)
    shape3_tuple = _shape_tuple(shape3)
    if shape0_tuple != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected addmm view shape parameter: {shape0!r}")
    for index, shape_tuple in enumerate((shape1_tuple, shape2_tuple, shape3_tuple), start=1):
        if shape_tuple != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output shape{index} parameter: {shape_tuple!r}")

    return addmm_49, residual, weight, bias, shape1_tuple, shape2_tuple, shape3_tuple


def oracle_residual_layernorm(
    addmm_49: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    shape1: tuple[int, int],
    shape2: tuple[int, int],
    shape3: tuple[int, int],
    *,
    num_warps: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the full Repro.forward residual-add LayerNorm output."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_layernorm.py")

    base = torch.empty_strided(
        RESIDUAL_SHAPE,
        BASE_STRIDE,
        device=addmm_49.device,
        dtype=torch.float16,
    )
    _residual_layernorm_kernel[(ROWS,)](
        addmm_49,
        residual,
        weight,
        bias,
        base,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )

    return base.view(shape1), base.view(shape2), base.view(shape3)


@oracle_impl(hardware="H100", shapes="(T([512, 768], f16), T([1, 512, 768], f16), T([768], f16), T([768], f16), S([1, 512, 768]), S([512, 768]), S([512, 768]), S([512, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the complete Repro.forward residual-add LayerNorm computation.

    SCOPE INVARIANT: accepts the same 8 inputs as Repro.forward() and returns
    the same three fp16 `[512, 768]` views with contiguous `(768, 1)` stride
    aliasing one `[1, 512, 768]` storage allocation.
    """
    addmm_49, residual, weight, bias, shape1, shape2, shape3 = _validate_inputs(inputs)
    return oracle_residual_layernorm(addmm_49, residual, weight, bias, shape1, shape2, shape3)


def _check_layout_and_alias(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = True
    for index, (expected_tensor, actual_tensor) in enumerate(zip(expected, actual)):
        layout_ok = (
            tuple(actual_tensor.shape) == tuple(expected_tensor.shape)
            and actual_tensor.stride() == expected_tensor.stride()
            and actual_tensor.dtype == expected_tensor.dtype
        )
        print(
            f"  output {index} layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()})"
        )
        ok = ok and layout_ok

    expected_storage_ptrs = {tensor.untyped_storage().data_ptr() for tensor in expected}
    actual_storage_ptrs = {tensor.untyped_storage().data_ptr() for tensor in actual}
    alias_ok = len(expected_storage_ptrs) == 1 and len(actual_storage_ptrs) == 1
    print(
        f"  output aliasing: {'PASS' if alias_ok else 'FAIL'} "
        f"(expected_same_storage={len(expected_storage_ptrs) == 1} "
        f"oracle_same_storage={len(actual_storage_ptrs) == 1})"
    )
    return ok and alias_ok


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
        ok = _check_layout_and_alias(instance, inputs) and ok
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
