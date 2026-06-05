"""
Full-scope Triton oracle for sum_f160d1f03c1f (BEiT QKV concat/transpose sum).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle differs from
Inductor by treating the cat -> view -> permute -> clone -> view pipeline as one
QKV layout materialization that also accumulates the two live side reductions
over the materialized `[25216, 2304]` matrix. Inductor cannot do this today
because the scheduler sees the cat/permute/clone producer, the returned
transpose view, and the sum/slice consumers as separate graph regions instead
of a single materializing multi-output reduction with side accumulators. The fix
class is SCHEDULER_FUSION: form a template that can write the required
materialized transpose backing storage while sharing the same loads for the
reduction side outputs.
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



REPRO_ID = "sum_f160d1f03c1f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 128
SEQ = 197
HEADS = 12
HEAD_DIM = 64
QKV = 3
QKV_CHUNK = HEADS * HEAD_DIM
COLS = QKV * QKV_CHUNK
ROWS = BATCH * SEQ

INPUT_STRIDE_N = SEQ * HEADS * HEAD_DIM
INPUT_STRIDE_S = HEADS * HEAD_DIM


@triton.jit
def _materialize_and_reduce_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    partial_q_ptr,
    partial_v_ptr,
    ROWS_: tl.constexpr,
    COLS_: tl.constexpr,
    SEQ_: tl.constexpr,
    QKV_CHUNK_: tl.constexpr,
    INPUT_STRIDE_N_: tl.constexpr,
    INPUT_STRIDE_S_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)

    rows = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < ROWS_
    col_mask = cols < COLS_
    mask = row_mask[:, None] & col_mask[None, :]

    n = rows // SEQ_
    s = rows - n * SEQ_
    qkv_col = cols % QKV_CHUNK_
    input_offsets = (
        n[:, None] * INPUT_STRIDE_N_
        + s[:, None] * INPUT_STRIDE_S_
        + qkv_col[None, :]
    )

    q_mask = mask & (cols[None, :] < QKV_CHUNK_)
    k_mask = mask & ((cols[None, :] >= QKV_CHUNK_) & (cols[None, :] < 2 * QKV_CHUNK_))
    v_mask = mask & (cols[None, :] >= 2 * QKV_CHUNK_)

    values = tl.load(q_ptr + input_offsets, mask=q_mask, other=0.0).to(tl.float32)
    values += tl.load(k_ptr + input_offsets, mask=k_mask, other=0.0).to(tl.float32)
    values += tl.load(v_ptr + input_offsets, mask=v_mask, other=0.0).to(tl.float32)

    out_offsets = rows[:, None] * COLS_ + cols[None, :]
    tl.store(out_ptr + out_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    q_cols = cols
    v_cols = cols - 2 * QKV_CHUNK_
    tl.store(
        partial_q_ptr + row_block * QKV_CHUNK_ + q_cols,
        partial,
        mask=col_mask & (q_cols < QKV_CHUNK_),
    )
    tl.store(
        partial_v_ptr + row_block * QKV_CHUNK_ + v_cols,
        partial,
        mask=col_mask & (v_cols >= 0) & (v_cols < QKV_CHUNK_),
    )


@triton.jit
def _finalize_qv_sums_kernel(
    partial_q_ptr,
    partial_v_ptr,
    sum_q_ptr,
    sum_v_ptr,
    NUM_ROW_BLOCKS_: tl.constexpr,
    QKV_CHUNK_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    blocks = tl.arange(0, BLOCK_ROW_BLOCKS)
    col_mask = cols < QKV_CHUNK_
    block_mask = blocks < NUM_ROW_BLOCKS_
    mask = block_mask[:, None] & col_mask[None, :]
    offsets = blocks[:, None] * QKV_CHUNK_ + cols[None, :]

    q_partials = tl.load(partial_q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    v_partials = tl.load(partial_v_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(sum_q_ptr + cols, tl.sum(q_partials, axis=0), mask=col_mask)
    tl.store(sum_v_ptr + cols, tl.sum(v_partials, axis=0), mask=col_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    getitem_44: torch.Tensor,
    getitem_45: torch.Tensor,
    getitem_46: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
    _shape_param_2: object,
    _shape_param_3: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert getitem_44.shape == (BATCH, HEADS, SEQ, HEAD_DIM)
    assert getitem_45.shape == (BATCH, HEADS, SEQ, HEAD_DIM)
    assert getitem_46.shape == (BATCH, HEADS, SEQ, HEAD_DIM)
    assert getitem_44.stride() == (INPUT_STRIDE_N, HEAD_DIM, INPUT_STRIDE_S, 1)
    assert getitem_45.stride() == (INPUT_STRIDE_N, HEAD_DIM, INPUT_STRIDE_S, 1)
    assert getitem_46.stride() == (INPUT_STRIDE_N, HEAD_DIM, INPUT_STRIDE_S, 1)

    out = torch.empty_strided(
        (COLS, ROWS),
        (1, COLS),
        device=getitem_44.device,
        dtype=torch.float32,
    )

    block_m = 128
    block_n = 64
    num_row_blocks = triton.cdiv(ROWS, block_m)
    num_col_blocks = triton.cdiv(COLS, block_n)
    partial_q = torch.empty((num_row_blocks, QKV_CHUNK), device=getitem_44.device, dtype=torch.float32)
    partial_v = torch.empty((num_row_blocks, QKV_CHUNK), device=getitem_44.device, dtype=torch.float32)

    _materialize_and_reduce_kernel[(num_row_blocks, num_col_blocks)](
        getitem_44,
        getitem_45,
        getitem_46,
        out,
        partial_q,
        partial_v,
        ROWS_=ROWS,
        COLS_=COLS,
        SEQ_=SEQ,
        QKV_CHUNK_=QKV_CHUNK,
        INPUT_STRIDE_N_=INPUT_STRIDE_N,
        INPUT_STRIDE_S_=INPUT_STRIDE_S,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=8,
    )

    sum_q = torch.empty((QKV_CHUNK,), device=getitem_44.device, dtype=torch.float32)
    sum_v = torch.empty((QKV_CHUNK,), device=getitem_44.device, dtype=torch.float32)
    finalize_block_c = 16
    _finalize_qv_sums_kernel[(triton.cdiv(QKV_CHUNK, finalize_block_c),)](
        partial_q,
        partial_v,
        sum_q,
        sum_v,
        NUM_ROW_BLOCKS_=num_row_blocks,
        QKV_CHUNK_=QKV_CHUNK,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    return out, sum_q, sum_v


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
