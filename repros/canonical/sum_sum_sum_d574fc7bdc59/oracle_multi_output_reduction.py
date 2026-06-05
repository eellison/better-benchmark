"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle consumes the same original inputs as repro.py, computes the row-local normalization backward summaries, accumulates the two sibling `[384]` column reductions, and writes the returned full `[8, 1500, 384]` residual-add tensor, whereas Inductor currently schedules the sibling column reductions and the dependent full-tensor epilogue as ordinary producer/consumer work with extra passes over the same `mm_6 + mm_9 + mm_10` and normalized-input expressions; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K template that coordinates row-parallel partial accumulators with a live materialized pointwise side output; the fix is COOPERATIVE_SPLIT_K: add scheduler/codegen support for split row tiles with partial reduction coordination when a pointwise producer also has live materialized side outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_d574fc7bdc59"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

B = 8
T = 1500
C = 384
ROWS = B * T
NUMEL = ROWS * C
INV_C = 1.0 / C



@triton.jit
def _summary_reduce_kernel(
    mm6_ptr,
    mm9_ptr,
    mm10_ptr,
    norm_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_col0_ptr,
    partial_col1_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_col0 = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_col1 = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = (
            tl.load(mm6_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm9_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm10_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        )
        mean = tl.load(mean_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        norm_input = tl.load(norm_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = (norm_input - mean[:, None]) * invstd[:, None]
        weighted = x * weight[None, :]

        tl.store(row_sum_ptr + row, tl.sum(tl.where(mask, weighted, 0.0), axis=1), mask=row_mask)
        tl.store(
            row_dot_ptr + row,
            tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1),
            mask=row_mask,
        )
        acc_col0 += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_col1 += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_col0_ptr + partial_offsets, acc_col0, mask=c_mask)
    tl.store(partial_col1_ptr + partial_offsets, acc_col1, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_col0_ptr,
    partial_col1_ptr,
    out_col0_ptr,
    out_col1_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    col0 = tl.load(partial_col0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    col1 = tl.load(partial_col1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_col0_ptr + c, tl.sum(col0, axis=0), mask=c < C_)
    tl.store(out_col1_ptr + c, tl.sum(col1, axis=0), mask=c < C_)


@triton.jit
def _epilogue_kernel(
    mm6_ptr,
    mm9_ptr,
    mm10_ptr,
    norm_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    add_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_
    row = offsets // C_
    c = offsets - row * C_

    x = (
        tl.load(mm6_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm9_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(mm10_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    )
    norm_input = tl.load(norm_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + row, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    add_value = tl.load(add_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + row, mask=active, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + row, mask=active, other=0.0).to(tl.float32)

    rhs = (norm_input - mean) * invstd
    weighted = x * weight
    out = add_value + invstd * INV_C_ * (weighted * C_ - row_sum - rhs * row_dot)
    tl.store(out_ptr + offsets, out, mask=active)


def oracle_fused(
    mm_6: torch.Tensor,
    mm_9: torch.Tensor,
    mm_10: torch.Tensor,
    arg1_1: torch.Tensor,
    arg9_1: torch.Tensor,
    arg10_1: torch.Tensor,
    arg0_1: torch.Tensor,
    add_3: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_6.shape == (ROWS, C)
    assert mm_9.shape == (ROWS, C)
    assert mm_10.shape == (ROWS, C)
    assert arg1_1.shape == (B, T, C)
    assert arg9_1.shape == (B, T, 1)
    assert arg10_1.shape == (B, T, 1)
    assert arg0_1.shape == (C,)
    assert add_3.shape == (B, T, C)
    assert mm_6.is_contiguous()
    assert mm_9.is_contiguous()
    assert mm_10.is_contiguous()
    assert arg1_1.is_contiguous()
    assert add_3.is_contiguous()

    row_split = 64
    xblock = 2
    num_row_blocks = triton.cdiv(ROWS, row_split)
    row_sum = torch.empty((ROWS,), device=mm_6.device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=mm_6.device, dtype=torch.float32)
    partial_col0 = torch.empty((num_row_blocks, C), device=mm_6.device, dtype=torch.float32)
    partial_col1 = torch.empty((num_row_blocks, C), device=mm_6.device, dtype=torch.float32)

    _summary_reduce_kernel[(num_row_blocks,)](
        mm_6,
        mm_9,
        mm_10,
        arg1_1,
        arg9_1,
        arg10_1,
        arg0_1,
        row_sum,
        row_dot,
        partial_col0,
        partial_col1,
        ROWS_=ROWS,
        C_=C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=512,
        num_warps=8,
    )

    out_col0 = torch.empty((C,), device=mm_6.device, dtype=torch.float32)
    out_col1 = torch.empty((C,), device=mm_6.device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(C, 16),)](
        partial_col0,
        partial_col1,
        out_col0,
        out_col1,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=16,
        num_warps=8,
    )

    out = torch.empty((B, T, C), device=mm_6.device, dtype=torch.float32)
    block_elems = 256
    _epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm_6,
        mm_9,
        mm_10,
        arg1_1,
        arg9_1,
        arg10_1,
        arg0_1,
        add_3,
        row_sum,
        row_dot,
        out,
        NUMEL_=NUMEL,
        C_=C,
        INV_C_=INV_C,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out_col0, out_col1, out


def oracle_forward(inputs):
    return oracle_fused(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
