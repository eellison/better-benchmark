"""
Oracle for sum_sum_sum_dbc10f22635c.

Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the
complete Repro.forward scope with one Triton pass over the original inputs,
including the two row-local reductions that feed the final per-channel reduction
and the full/scatter/expand/div/sum tail folded into a direct accumulation over
the pre-expanded values, whereas Inductor lowers the decomposed graph as generic
pointwise, sibling reductions, and the scatter/expand reduction tail; Inductor
cannot do this today because it does not prove that this dense as_strided_scatter
plus zero-stride expand divided by 49 and reduced over the expanded dimensions is
equivalent to reducing the source values directly while also fusing the dependent
row reductions with the sibling channel reductions; the fix is
ALGEBRAIC_ELIMINATION: add shape/stride-aware elimination of the dense
scatter-expand-reduce tail and lower the remaining dependent multi-output
reduction as one fused accumulator kernel.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_sum_sum_sum_kernel(
        mm_ptr,
        arg79_ptr,
        arg216_ptr,
        arg218_ptr,
        partials_ptr,
        B: tl.constexpr,
        C: tl.constexpr,
        NUM_ROW_BLOCKS: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        rows = row_block * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
        c_offsets = tl.arange(0, BLOCK_C)
        row_mask = rows < B
        c_mask = c_offsets < C
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = rows[:, None] * C + c_offsets[None, :]

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg216 = tl.load(arg216_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        arg79 = tl.load(arg79_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weighted = mm * arg79[None, :]

        row_weighted_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_weighted_arg216_sum = tl.sum(
            tl.where(mask, weighted * arg216, 0.0),
            axis=1,
        )

        arg218 = tl.load(arg218_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        tail = arg218[:, None] * (
            (weighted * 640.0 - row_weighted_sum[:, None])
            - arg216 * row_weighted_arg216_sum[:, None]
        )

        partial0 = tl.sum(tl.where(mask, mm * arg216, 0.0), axis=0)
        partial1 = tl.sum(tl.where(mask, mm, 0.0), axis=0)
        partial2 = tl.sum(tl.where(mask, tail, 0.0), axis=0)

        partial_offsets = row_block * C + c_offsets
        plane_stride = NUM_ROW_BLOCKS * C
        tl.store(partials_ptr + partial_offsets, partial0, mask=c_mask)
        tl.store(partials_ptr + plane_stride + partial_offsets, partial1, mask=c_mask)
        tl.store(partials_ptr + 2 * plane_stride + partial_offsets, partial2, mask=c_mask)

    @triton.jit
    def _final_sum_sum_sum_kernel(
        partials_ptr,
        out_ptr,
        C: tl.constexpr,
        NUM_ROW_BLOCKS: tl.constexpr,
        PARTIAL_BLOCK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_blocks = tl.arange(0, PARTIAL_BLOCK)
        c_offsets = tl.arange(0, BLOCK_C)
        row_mask = row_blocks < NUM_ROW_BLOCKS
        c_mask = c_offsets < C
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row_blocks[:, None] * C + c_offsets[None, :]
        plane_stride = NUM_ROW_BLOCKS * C

        partial0 = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        partial1 = tl.load(
            partials_ptr + plane_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        partial2 = tl.load(
            partials_ptr + 2 * plane_stride + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        tl.store(out_ptr + c_offsets, tl.sum(partial0, axis=0), mask=c_mask)
        tl.store(out_ptr + C + c_offsets, tl.sum(partial1, axis=0), mask=c_mask)
        tl.store(out_ptr + 2 * C + c_offsets, tl.sum(partial2, axis=0), mask=c_mask)


def _validate_inputs(
    inputs: tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"expected 6 Repro inputs, got {len(inputs)}")

    mm, arg79_1, arg216_1, arg218_1, shape_param_0, shape_param_1 = inputs
    if not all(isinstance(x, torch.Tensor) for x in (mm, arg79_1, arg216_1, arg218_1)):
        raise TypeError("expected the first four Repro inputs to be tensors")
    if mm.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if mm.dtype != torch.float32 or arg79_1.dtype != torch.float32:
        raise TypeError("expected mm and arg79_1 to be float32")
    if arg216_1.dtype != torch.float32 or arg218_1.dtype != torch.float32:
        raise TypeError("expected arg216_1 and arg218_1 to be float32")

    b, c = mm.shape
    expected_shape_0 = [b, c, 1, 1]
    expected_shape_1 = [b, c, 7, 7]
    if list(shape_param_0) != expected_shape_0:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if list(shape_param_1) != expected_shape_1:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if tuple(arg79_1.shape) != (c,):
        raise ValueError(f"unexpected arg79_1 shape: {tuple(arg79_1.shape)}")
    if tuple(arg216_1.shape) != (b, 1, 1, c):
        raise ValueError(f"unexpected arg216_1 shape: {tuple(arg216_1.shape)}")
    if tuple(arg218_1.shape) != (b, 1, 1, 1):
        raise ValueError(f"unexpected arg218_1 shape: {tuple(arg218_1.shape)}")
    if not mm.is_contiguous() or not arg79_1.is_contiguous():
        raise ValueError("oracle expects contiguous mm and arg79_1 inputs")
    if not arg216_1.is_contiguous() or not arg218_1.is_contiguous():
        raise ValueError("oracle expects contiguous arg216_1 and arg218_1 inputs")
    return mm, arg79_1, arg216_1, arg218_1, b, c


def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same three f32[640] outputs. The third output is not derived from the first
    two sibling channel sums; it is accumulated directly after computing the
    row-local reductions required by the original pointwise expression.
    """
    mm, arg79_1, arg216_1, arg218_1, b, c = _validate_inputs(tuple(inputs))

    row_block = 16
    num_row_blocks = triton.cdiv(b, row_block)
    block_c = triton.next_power_of_2(c)
    partial_block = triton.next_power_of_2(num_row_blocks)

    partials = torch.empty((3, num_row_blocks, c), device=mm.device, dtype=torch.float32)
    out = torch.empty((3, c), device=mm.device, dtype=torch.float32)

    _partial_sum_sum_sum_kernel[(num_row_blocks,)](
        mm,
        arg79_1,
        arg216_1,
        arg218_1,
        partials,
        B=b,
        C=c,
        NUM_ROW_BLOCKS=num_row_blocks,
        ROW_BLOCK=row_block,
        BLOCK_C=block_c,
        num_warps=8,
    )
    _final_sum_sum_sum_kernel[(1,)](
        partials,
        out,
        C=c,
        NUM_ROW_BLOCKS=num_row_blocks,
        PARTIAL_BLOCK=partial_block,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return out[0], out[1], out[2]


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
                        "WARNING: oracle is slower than compile for "
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
