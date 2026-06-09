"""
Oracle for var_mean_24e1d446bfed

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured fp32 hidden-size-768 LayerNorm in one shape-specialized Triton row kernel, preserving the input reshape semantics, `var_mean(..., correction=0, keepdim=True)`, eps=1e-6 affine epilogue, final `[128, 768]` reshape, and contiguous output stride, whereas Inductor already fuses this small fixed-K normalization region into its generated norm/reduction code for the captured shape; Inductor cannot materially improve it today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the required row input read, affine vector reads, output write, and per-row reduction/rsqrt latency dominate the one-kernel workload; the fix is BANDWIDTH_BOUND: close this repro as an at-floor norm-template case unless future broader launch-overhead or norm-template constant-factor work moves the whole family.
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

ROWS = 128
HIDDEN = 768
RESHAPED_INPUT = (ROWS, 1, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-6
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias
        tl.store(output_ptr + offsets, y, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_49, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm_49, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first three repro inputs must be tensors")

    expected_shapes = (OUTPUT_SHAPE, (HIDDEN,), (HIDDEN,))
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(
                f"input {index} shape {tuple(value.shape)} != {expected_shape}"
            )
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    if tuple(int(dim) for dim in shape0) != RESHAPED_INPUT:
        raise ValueError(f"unexpected first shape parameter: {shape0!r}")
    if tuple(int(dim) for dim in shape1) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected second shape parameter: {shape1!r}")

    return addmm_49, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 768], f32), T([768], f32), T([768], f32), S([128, 1, 768]), S([128, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with a Triton LayerNorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous float32[128, 768] output tensor. The two reshapes in
    Repro.forward are view-only for this shape, so the kernel reads and writes
    the equivalent flattened row-major layout directly.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm.py")

    addmm_49, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=addmm_49.device,
        dtype=addmm_49.dtype,
    )
    _layernorm_kernel[(ROWS,)](
        addmm_49,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_H=1024,
        num_warps=8,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
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
        ok = _check_layout(instance, inputs) and ok
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
