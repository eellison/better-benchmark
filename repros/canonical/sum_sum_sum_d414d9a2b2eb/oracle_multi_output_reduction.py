"""
Oracle for sum_sum_sum_d414d9a2b2eb

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full
four-output layernorm-backward-style region with a Triton row kernel that keeps
the dependent row sums in registers, writes the complete permuted side backing
tensor, and accumulates all three column reductions, whereas Inductor currently
schedules the shared row reductions, column reductions, and dependent side-output
epilogue as generic multi-kernel reduction/pointwise work. Inductor cannot do
this today because its scheduler has no template for chaining row reductions into
a transposed side-output epilogue while also producing sibling column reductions
from the same producer. The required change is SCHEDULER_FUSION: add a
multi-output chained-reduction template that keeps the row summaries as
first-class intermediates and fuses the dependent column/transpose epilogue for
the complete return tuple.

Measurement note: this implementation is full-scope and uses Triton kernels for
the measured work, but it is diagnosis-only rather than a true floor on the local
run because the requested combo-kernel Inductor config is slightly faster.
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
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


ROWS = 128
HIDDEN = 768
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_outputs_kernel(
        out0_ptr,
        out1_ptr,
        out3_ptr,
        n_cols: tl.constexpr,
        block_n: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
        mask = offsets < n_cols
        zeros = tl.zeros((block_n,), dtype=tl.float32)
        tl.store(out0_ptr + offsets, zeros, mask=mask)
        tl.store(out1_ptr + offsets, zeros, mask=mask)
        tl.store(out3_ptr + offsets, zeros, mask=mask)

    @triton.jit
    def _row_full_atomic_kernel(
        mm_ptr,
        weight_ptr,
        arg250_ptr,
        mean_ptr,
        scale_ptr,
        select_ptr,
        out0_ptr,
        out1_ptr,
        side_base_ptr,
        out3_ptr,
        n_cols: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_n)
        mask = cols < n_cols
        offsets = row * n_cols + cols

        x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        a250 = tl.load(arg250_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        select = tl.load(select_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)

        yhat = (a250 - mean) * scale
        mul = x * weight
        row_sum = tl.sum(mul, axis=0)
        row_dot = tl.sum(mul * yhat, axis=0)
        grad = (scale / n_cols) * (mul * n_cols - row_sum - yhat * row_dot)
        side = select + grad

        tl.store(side_base_ptr + offsets, side, mask=mask)
        tl.atomic_add(out0_ptr + cols, x * yhat, sem="relaxed", mask=mask)
        tl.atomic_add(out1_ptr + cols, x, sem="relaxed", mask=mask)
        tl.atomic_add(out3_ptr + cols, side, sem="relaxed", mask=mask)

def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"expected 10 inputs, got {len(inputs)}")

    mm_2, arg80_1, arg250_1, arg251_1, arg252_1, select_scatter, shape0, shape1, shape2, shape3 = inputs
    tensors = (mm_2, arg80_1, arg250_1, arg251_1, arg252_1, select_scatter)
    if not all(isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first six repro inputs must be tensors")
    if tuple(mm_2.shape) != (ROWS, HIDDEN):
        raise ValueError(f"mm_2 shape {tuple(mm_2.shape)} != {(ROWS, HIDDEN)}")
    if tuple(arg80_1.shape) != (HIDDEN,):
        raise ValueError(f"arg80_1 shape {tuple(arg80_1.shape)} != {(HIDDEN,)}")
    if tuple(arg250_1.shape) != (ROWS, HIDDEN):
        raise ValueError(f"arg250_1 shape {tuple(arg250_1.shape)} != {(ROWS, HIDDEN)}")
    if tuple(arg251_1.shape) != (ROWS, 1, 1) or tuple(arg252_1.shape) != (ROWS, 1, 1):
        raise ValueError("arg251_1 and arg252_1 must both have shape (128, 1, 1)")
    if tuple(select_scatter.shape) != (ROWS, 1, HIDDEN):
        raise ValueError(f"select_scatter shape {tuple(select_scatter.shape)} != {(ROWS, 1, HIDDEN)}")
    if any(x.dtype != torch.float32 for x in tensors):
        raise TypeError("all tensor inputs must be torch.float32")
    if any(not x.is_cuda for x in tensors):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if any(not x.is_contiguous() for x in tensors):
        raise ValueError("all tensor inputs must be contiguous")
    if list(shape0) != [ROWS, 1, HIDDEN] or list(shape1) != [ROWS, 1, HIDDEN]:
        raise ValueError(f"unexpected leading view shape params: {shape0!r}, {shape1!r}")
    if list(shape2) != [ROWS, HIDDEN] or list(shape3) != [HIDDEN]:
        raise ValueError(f"unexpected trailing view shape params: {shape2!r}, {shape3!r}")

    return tensors


def oracle_forward(inputs):
    """Run the full Repro.forward computation with shape-specialized Triton kernels.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same four outputs: two [768] reductions, the [768, 128] permuted side tensor,
    and the final [768] side reduction.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_2, arg80_1, arg250_1, arg251_1, arg252_1, select_scatter = _validate_inputs(inputs)
    out0 = torch.empty((HIDDEN,), device=mm_2.device, dtype=torch.float32)
    out1 = torch.empty((HIDDEN,), device=mm_2.device, dtype=torch.float32)
    side_base = torch.empty((ROWS, HIDDEN), device=mm_2.device, dtype=torch.float32)
    out3 = torch.empty((HIDDEN,), device=mm_2.device, dtype=torch.float32)

    _zero_outputs_kernel[(triton.cdiv(HIDDEN, 256),)](
        out0,
        out1,
        out3,
        n_cols=HIDDEN,
        block_n=256,
        num_warps=4,
    )
    _row_full_atomic_kernel[(ROWS,)](
        mm_2,
        arg80_1,
        arg250_1,
        arg251_1,
        arg252_1,
        select_scatter,
        out0,
        out1,
        side_base,
        out3,
        n_cols=HIDDEN,
        block_n=1024,
        num_warps=8,
    )
    return out0, out1, side_base.permute(1, 0), out3


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
