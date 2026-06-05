"""
Full-scope oracle for sum_sum_sum_f0377fc40fe2 (DeiT tiny LN backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle covers the
same full repro.py scope, including the mm view, affine multiply, row-local
layernorm backward reductions, position-add/center/rsqrt producer, residual add,
and all five returned reductions with matching output strides. It differs from
Inductor by streaming the shared `[128, 197, 192]` producers through a split-row
Triton kernel that computes the per-row layernorm scalars, sibling column
partials, and token-sum atomics together instead of materializing the full
intermediate.
Inductor cannot express this today because the final outputs mix reductions
over C, batch, token, and patch axes with a dependent row reduction, so the
scheduler emits multiple ordinary producer/consumer reductions instead of one
coordinated multi-output reduction; the fix is COOPERATIVE_SPLIT_K support for
dependent multi-output reductions with row-tile partials and recompute in the
consumer reduction.
"""
from __future__ import annotations

import argparse
import importlib.util
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



REPRO_ID = "sum_sum_sum_f0377fc40fe2"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
TOKENS = 197
CHANNELS = 192
ROWS = BATCH * TOKENS
PATCH_TOKENS = TOKENS - 1
INV_C = 1.0 / CHANNELS



def _row_summary_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < CHANNELS_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_norm = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        token = row % TOKENS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * CHANNELS_ + c[None, :]
        pos_offsets = token[:, None] * CHANNELS_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        pos = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        norm = (source + pos - mean[:, None]) * rsqrt[:, None]
        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)

        tl.store(row_sum_ptr + row, row_sum, mask=row_mask)
        tl.store(row_dot_ptr + row, row_dot, mask=row_mask)
        acc_x_norm += tl.sum(tl.where(mask, x * norm, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * CHANNELS_ + c
    tl.store(partial_x_norm_ptr + partial_offsets, acc_x_norm, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < CHANNELS_)
    offsets = block[:, None] * CHANNELS_ + c[None, :]

    x_norm = tl.load(partial_x_norm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c_mask = c < CHANNELS_
    tl.store(out_x_norm_ptr + c, tl.sum(x_norm, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)


@triton.jit
def _row_summary_atomic_token_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    residual_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    token_sum_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_C_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < CHANNELS_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_norm = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        token = row % TOKENS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * CHANNELS_ + c[None, :]
        pos_offsets = token[:, None] * CHANNELS_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        pos = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        rsqrt = tl.load(rsqrt_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        norm = (source + pos - mean[:, None]) * rsqrt[:, None]
        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
        ln_grad = (
            rsqrt[:, None]
            * INV_C_
            * (weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None])
        )

        token_offsets = token[:, None] * CHANNELS_ + c[None, :]
        tl.atomic_add(token_sum_ptr + token_offsets, residual + ln_grad, sem="relaxed", mask=mask)
        acc_x_norm += tl.sum(tl.where(mask, x * norm, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * CHANNELS_ + c
    tl.store(partial_x_norm_ptr + partial_offsets, acc_x_norm, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)


@triton.jit
def _batch_token_reduce_kernel(
    x_ptr,
    weight_ptr,
    source_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    residual_ptr,
    row_sum_ptr,
    row_dot_ptr,
    token_sum_ptr,
    cls_sum_ptr,
    BATCH_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_id = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    b = tl.arange(0, BLOCK_B)
    c_mask = c < CHANNELS_
    b_mask = b < BATCH_
    row = b * TOKENS_ + token_id
    mask = b_mask[:, None] & c_mask[None, :]
    offsets = row[:, None] * CHANNELS_ + c[None, :]
    pos_offsets = token_id * CHANNELS_ + c

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + pos_offsets, mask=c_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + row, mask=b_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    norm = (source + pos[None, :] - mean[:, None]) * rsqrt[:, None]
    ln_grad = (
        rsqrt[:, None]
        * INV_C_
        * (x * weight[None, :] * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None])
    )
    value = residual + ln_grad
    reduced = tl.sum(tl.where(mask, value, 0.0), axis=0)

    out_offsets = token_id * CHANNELS_ + c
    tl.store(token_sum_ptr + out_offsets, reduced, mask=c_mask)
    if token_id == 0:
        tl.store(cls_sum_ptr + c, reduced, mask=c_mask)


@triton.jit
def _patch_token_sum_kernel(
    token_sum_ptr,
    patch_sum_ptr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    token = 1 + tl.arange(0, BLOCK_TOKENS)
    mask = (token[:, None] < TOKENS_) & (c[None, :] < CHANNELS_)
    offsets = token[:, None] * CHANNELS_ + c[None, :]
    values = tl.load(token_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(patch_sum_ptr + c, tl.sum(values, axis=0), mask=c < CHANNELS_)


@triton.jit
def _finalize_columns_and_patch_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    token_sum_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    patch_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    partial_mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < CHANNELS_)
    partial_offsets = block[:, None] * CHANNELS_ + c[None, :]

    x_norm = tl.load(partial_x_norm_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + partial_offsets, mask=partial_mask, other=0.0).to(tl.float32)
    c_mask = c < CHANNELS_
    tl.store(out_x_norm_ptr + c, tl.sum(x_norm, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)

    token = 1 + tl.arange(0, BLOCK_TOKENS)
    token_mask = (token[:, None] < TOKENS_) & c_mask[None, :]
    token_offsets = token[:, None] * CHANNELS_ + c[None, :]
    token_values = tl.load(token_sum_ptr + token_offsets, mask=token_mask, other=0.0).to(tl.float32)
    tl.store(patch_sum_ptr + c, tl.sum(token_values, axis=0), mask=c_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_96: torch.Tensor,
    arg3_1: torch.Tensor,
    arg77_1: torch.Tensor,
    arg2_1: torch.Tensor,
    arg78_1: torch.Tensor,
    arg79_1: torch.Tensor,
    add_46: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_96.shape == (ROWS, CHANNELS)
    assert arg3_1.shape == (CHANNELS,)
    assert arg77_1.shape == (BATCH, TOKENS, CHANNELS)
    assert arg2_1.shape == (1, TOKENS, CHANNELS)
    assert arg78_1.shape == (BATCH, TOKENS, 1)
    assert arg79_1.shape == (BATCH, TOKENS, 1)
    assert add_46.shape == (BATCH, TOKENS, CHANNELS)

    x = mm_96.contiguous()
    weight = arg3_1.contiguous()
    source = arg77_1.contiguous().reshape(ROWS, CHANNELS)
    pos = arg2_1.contiguous().reshape(TOKENS, CHANNELS)
    mean = arg78_1.contiguous().reshape(ROWS)
    rsqrt = arg79_1.contiguous().reshape(ROWS)
    residual = add_46.contiguous().reshape(ROWS, CHANNELS)
    device = x.device

    row_split = 8
    xblock = 1
    block_c = 256
    num_row_blocks = triton.cdiv(ROWS, row_split)
    partials = torch.empty((2, num_row_blocks, CHANNELS), device=device, dtype=torch.float32)
    partial_x_norm = partials[0]
    partial_x = partials[1]
    token_sum = torch.empty((1, TOKENS, CHANNELS), device=device, dtype=torch.float32)
    token_sum.zero_()

    _row_summary_atomic_token_kernel[(num_row_blocks,)](
        x,
        weight,
        source,
        pos,
        mean,
        rsqrt,
        residual,
        partial_x_norm,
        partial_x,
        token_sum,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        INV_C_=INV_C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=4,
    )

    vector_outputs = torch.empty((3, CHANNELS), device=device, dtype=torch.float32)
    out_x_norm = vector_outputs[0]
    out_x = vector_outputs[1]
    patch_sum = vector_outputs[2]
    finalize_block_c = 8
    _finalize_columns_and_patch_kernel[(triton.cdiv(CHANNELS, finalize_block_c),)](
        partial_x_norm,
        partial_x,
        token_sum,
        out_x_norm,
        out_x,
        patch_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_TOKENS=triton.next_power_of_2(PATCH_TOKENS),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    cls_sum = torch.as_strided(token_sum, (1, 1, CHANNELS), (CHANNELS, CHANNELS, 1))

    return out_x_norm, out_x, token_sum, cls_sum, patch_sum


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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
