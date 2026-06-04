"""
Oracle for var_mean_any_e86b9ea56684

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete residual-add, fp32 hidden-size-768 layernorm, affine epilogue, fp16 cast, and final all-finite scalar predicate with Triton row flags instead of materializing the normalized `[1, 512, 768]` tensor or boolean intermediates, whereas tuned Inductor already measures within noise of this direct full-scope floor for the captured shape; Inductor cannot materially improve this today because the remaining work is one small fixed-K normalization pass plus a tiny predicate reduction dominated by launch overhead, affine memory traffic, and reduction/rsqrt latency rather than by an actionable missed fusion or algebraic rewrite; the fix is BANDWIDTH_BOUND: record the repro as at-floor/closed unless a broader launch-overhead reduction applies across surrounding graph regions.
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
BATCH = 1
EPS = 1.0e-5
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _layernorm_row_finite_kernel(
        addmm_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        row_flags_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        x = (addmm + residual).to(tl.float16).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=0) / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y_f16 = (centered * invstd * weight + bias).to(tl.float16)

        finite = (y_f16 == y_f16) & (tl.abs(y_f16) != float("inf"))
        row_ok = tl.sum(tl.where(mask & finite, 1, 0), axis=0) == hidden
        tl.store(row_flags_ptr + row, row_ok.to(tl.int8))

    @triton.jit
    def _all_rows_finite_kernel(
        row_flags_ptr,
        out_ptr,
        rows: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < rows
        flags = tl.load(row_flags_ptr + offsets, mask=mask, other=1).to(tl.int32)
        all_ok = tl.min(flags, axis=0) != 0
        tl.store(out_ptr, all_ok)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_5, residual, weight, bias, output_shape = inputs
    tensor_inputs = (addmm_5, residual, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first four repro inputs must be tensors")

    expected_shapes = ((ROWS, HIDDEN), (BATCH, ROWS, HIDDEN), (HIDDEN,), (HIDDEN,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    if list(output_shape) != [BATCH, ROWS, HIDDEN]:
        raise ValueError(f"unexpected shape parameter: {output_shape!r}")

    return addmm_5, residual, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with Triton finite reductions.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single CUDA bool scalar. The normalized fp16 values are computed for
    the finite predicate but are not materialized because Repro only returns
    their all-finite result.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm_finite.py")

    addmm_5, residual, weight, bias = _validate_inputs(inputs)
    row_flags = torch.empty((ROWS,), device=addmm_5.device, dtype=torch.int8)
    out = torch.empty((), device=addmm_5.device, dtype=torch.bool)

    _layernorm_row_finite_kernel[(ROWS,)](
        addmm_5,
        residual,
        weight,
        bias,
        row_flags,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
        num_warps=4,
    )
    _all_rows_finite_kernel[(1,)](
        row_flags,
        out,
        rows=ROWS,
        BLOCK_M=triton.next_power_of_2(ROWS),
        num_warps=1,
    )
    return out


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
