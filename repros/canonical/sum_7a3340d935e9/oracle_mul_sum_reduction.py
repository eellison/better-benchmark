"""Gap diagnosis (classification: BANDWIDTH_BOUND): this full-scope oracle computes `sum(arg303_1 * arg230_1, dim=[0, 1])` for the exact Repro.forward output by reading both contiguous f32[B, S, H] inputs once, multiplying in registers, writing per-hidden partial sums for row chunks, and finalizing the returned contiguous f32[H] vector. Inductor already does the important thing here: it fuses the pointwise multiply into a two-stage reduction and does not materialize the full product tensor, so the oracle differs only in hand-written Triton spelling of the same required input-read, reduction, small partial-buffer, and output-store work. Inductor cannot be assigned a material local optimization gap today because there are no sibling reductions, dependent epilogues, scatter intermediates, or recomputable consumers to remove; the required Inductor classification is BANDWIDTH_BOUND: keep the existing fused reduction path unless a broader reduction-template improvement moves this whole simple product-reduction family."""
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

SPLIT_ROWS = 128
SPLIT_X = 128
FINAL_X = 16


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _mul_sum_split_kernel(
        x_ptr,
        y_ptr,
        partials_ptr,
        rows_: tl.constexpr,
        cols_: tl.constexpr,
        xnumel_: tl.constexpr,
        GROUP_ROWS: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        x_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        r_offsets = tl.arange(0, RBLOCK)[None, :]
        cols = x_offsets % cols_
        group = x_offsets // cols_
        rows = group * GROUP_ROWS + r_offsets
        mask = (x_offsets < xnumel_) & (rows < rows_)
        offsets = rows * cols_ + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partial = tl.sum(x * y, axis=1)[:, None]
        tl.store(partials_ptr + x_offsets, partial, mask=x_offsets < xnumel_)

    @triton.jit
    def _finalize_partials_kernel(
        partials_ptr,
        out_ptr,
        num_row_blocks_: tl.constexpr,
        cols_: tl.constexpr,
        RBLOCK: tl.constexpr,
        XBLOCK: tl.constexpr,
    ):
        cols = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        row_blocks = tl.arange(0, RBLOCK)[None, :]
        offsets = row_blocks * cols_ + cols
        mask = (cols < cols_) & (row_blocks < num_row_blocks_)
        partials = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = tl.sum(partials, axis=1)[:, None]
        tl.store(out_ptr + cols, out, mask=cols < cols_)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    x, y = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError(f"expected tensor inputs, got {type(x)!r}, {type(y)!r}")
    if x.shape != y.shape:
        raise ValueError(f"input shapes must match, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 inputs, got shape={tuple(x.shape)}")
    if x.dtype is not torch.float32 or y.dtype is not torch.float32:
        raise TypeError(f"expected float32 inputs, got {x.dtype} and {y.dtype}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("oracle_mul_sum_reduction.py expects CUDA inputs")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError(f"expected contiguous inputs, got strides {x.stride()} and {y.stride()}")
    return x, y


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full `mul -> sum([0, 1])` repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mul_sum_reduction.py")

    x, y = _validate_inputs(inputs)
    rows = x.shape[0] * x.shape[1]
    cols = x.shape[2]
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=torch.float32)
    num_row_blocks = triton.cdiv(rows, SPLIT_ROWS)
    partial_xnumel = num_row_blocks * cols
    partials = torch.empty_strided(
        (num_row_blocks, cols),
        (cols, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _mul_sum_split_kernel[(triton.cdiv(partial_xnumel, SPLIT_X),)](
        x,
        y,
        partials,
        rows_=rows,
        cols_=cols,
        xnumel_=partial_xnumel,
        GROUP_ROWS=SPLIT_ROWS,
        XBLOCK=SPLIT_X,
        RBLOCK=SPLIT_ROWS,
        num_warps=4,
        num_stages=1,
    )

    final_rblock = triton.next_power_of_2(num_row_blocks)
    _finalize_partials_kernel[(triton.cdiv(cols, FINAL_X),)](
        partials,
        out,
        num_row_blocks_=num_row_blocks,
        cols_=cols,
        RBLOCK=final_rblock,
        XBLOCK=FINAL_X,
        num_warps=4,
        num_stages=3,
    )
    return out


def main() -> None:
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
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
