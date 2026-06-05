"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GPT-Neo layer-norm-backward scope returned by `Repro.forward`, including the two sibling column reductions over the matmul view, the row-local hidden-dimension reductions used by the layer-norm backward formula, the materialized `[2048, 4096]` transposed side output, and the downstream column reduction over that side output; whereas Inductor schedules the row reductions, sibling reductions, add epilogue, transpose view materialization, and final side-output reduction as separate generic pieces over materialized intermediates. Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that shares row-local scalars while producing a required materialized side output and compatible column accumulators. The fix is COOPERATIVE_SPLIT_K: add a dependent multi-output reduction plan that row-tiles the hidden reduction, writes structured side outputs, and finalizes all column partials together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps CPU-only syntax checks usable.
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

BATCH = 32
SEQ = 128
ROWS = BATCH * SEQ
CHANNELS = 2048
ROW_SPLIT = 16
FINALIZE_BLOCK_C = 32
FINALIZE_BLOCK_ROWS = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_block_full_scope_kernel(
        x_ptr,
        weight_ptr,
        rhs_ptr,
        gate_ptr,
        residual_ptr,
        out_base_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_out_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        ROW_SPLIT_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        col_mask = cols < CHANNELS_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

        acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
        acc_out = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for local_row in tl.range(0, ROW_SPLIT_):
            row = row_block * ROW_SPLIT_ + local_row
            row_mask = row < ROWS_
            mask = row_mask & col_mask
            offsets = row * CHANNELS_ + cols

            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

            weighted = x * weight
            row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
            row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=0)
            out = residual + gate * (
                weighted * CHANNELS_ - row_sum - rhs * row_dot
            )

            tl.store(out_base_ptr + offsets, out, mask=mask)
            acc_x_rhs += tl.where(mask, x * rhs, 0.0)
            acc_x += tl.where(mask, x, 0.0)
            acc_out += tl.where(mask, out, 0.0)

        partial_offsets = row_block * CHANNELS_ + cols
        tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=col_mask)
        tl.store(partial_x_ptr + partial_offsets, acc_x, mask=col_mask)
        tl.store(partial_out_ptr + partial_offsets, acc_out, mask=col_mask)


    @triton.jit
    def _finalize_column_partials_kernel(
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_out_ptr,
        out_x_rhs_ptr,
        out_x_ptr,
        out_sum_out_ptr,
        NUM_ROW_BLOCKS: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_ROW_BLOCKS: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
        col_mask = cols < CHANNELS_
        block_mask = blocks < NUM_ROW_BLOCKS
        mask = block_mask[:, None] & col_mask[None, :]
        offsets = blocks[:, None] * CHANNELS_ + cols[None, :]

        x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )

        tl.store(out_x_rhs_ptr + cols, tl.sum(x_rhs, axis=0), mask=col_mask)
        tl.store(out_x_ptr + cols, tl.sum(x, axis=0), mask=col_mask)
        tl.store(out_sum_out_ptr + cols, tl.sum(out, axis=0), mask=col_mask)


def _oracle_torch(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_280,
        arg8_1,
        arg227_1,
        arg538_1,
        add_184,
        shape0,
        shape1,
        shape2,
    ) = inputs

    x = mm_280.view(shape0)
    weighted = x * arg8_1
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * arg227_1).sum(dim=2, keepdim=True)
    grad = arg538_1 * (weighted * CHANNELS - row_sum - arg227_1 * row_dot)

    out_x_rhs = (x * arg227_1).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))
    out_base = (add_184 + grad).view(shape1)
    out_sum = out_base.sum(dim=0, keepdim=True).view(shape2)
    return out_x_rhs, out_x, out_base.permute(1, 0), out_sum


def _oracle_triton(*inputs: object) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("triton is not available")

    (
        mm_280,
        arg8_1,
        arg227_1,
        arg538_1,
        add_184,
        *_shape_params,
    ) = inputs

    if mm_280.device.type != "cuda":
        return _oracle_torch(*inputs)

    assert mm_280.shape == (ROWS, CHANNELS)
    assert arg8_1.shape == (CHANNELS,)
    assert arg227_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg538_1.shape == (BATCH, SEQ, 1)
    assert add_184.shape == (BATCH, SEQ, CHANNELS)

    rhs = arg227_1.reshape(ROWS, CHANNELS)
    gate = arg538_1.reshape(ROWS)
    residual = add_184.reshape(ROWS, CHANNELS)
    device = mm_280.device
    num_row_blocks = triton.cdiv(ROWS, ROW_SPLIT)

    partial_x_rhs = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    partial_x = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    partial_out = torch.empty(
        (num_row_blocks, CHANNELS), device=device, dtype=torch.float32
    )
    out_base = torch.empty((ROWS, CHANNELS), device=device, dtype=torch.float32)

    _row_block_full_scope_kernel[(num_row_blocks,)](
        mm_280,
        arg8_1,
        rhs,
        gate,
        residual,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        ROW_SPLIT_=ROW_SPLIT,
        BLOCK_C=CHANNELS,
        num_warps=8,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_sum_out = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _finalize_column_partials_kernel[(triton.cdiv(CHANNELS, FINALIZE_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=FINALIZE_BLOCK_ROWS,
        BLOCK_C=FINALIZE_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_sum_out


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    return _oracle_triton(*inputs)


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
