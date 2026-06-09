"""
Oracle for sum_sum_sum_88c9224fa05a (FNet layer-norm backward plus embedding scatters).

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the layer-norm-backward hidden-row reductions, the two sibling column reductions, and the three returned position/segment/vocabulary embedding-gradient scatter-add outputs from the same Triton producer, so it does not materialize the full `[32, 512, 768]` layer-norm input-gradient tensor and reread it through separate scatter paths; Inductor can fuse pieces of the pointwise/reduction graph, but today its scheduler cannot represent one row-reduction producer that simultaneously feeds independent column reductions and multiple side-effect scatter-add stores with different output layouts, so it splits the graph around the materialized gradient and the embedding scatters; the Inductor fix is scheduler/codegen support for this fused multi-output reduction/scatter template, allowing reductions with shared inputs and dependent scatter stores to be emitted as one producer plus a small column-sum finalizer.
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_sum_sum_88c9224fa05a"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 32
SEQ = 512
HIDDEN = 768
HIDDEN_BLOCK = 1024
ROWS = BATCH * SEQ
POSITION_ROWS = 512
SEGMENT_ROWS = 4
VOCAB_ROWS = 32000
ROW_BLOCK = 16
ROW_BLOCKS = triton.cdiv(ROWS, ROW_BLOCK)


def _fused_ln_embedding_scatter_kernel(
    mm52_ptr,
    gamma_ptr,
    xhat_ptr,
    scale_ptr,
    pos_idx_ptr,
    full_ptr,
    segment_idx_ptr,
    word_idx_ptr,
    partial_sum_vx_ptr,
    partial_sum_v_ptr,
    pos_out_ptr,
    segment_out_ptr,
    word_out_ptr,
    ROWS_: tl.constexpr,
    SEQ_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BATCH_: tl.constexpr,
    POSITION_ROWS_: tl.constexpr,
    SEGMENT_ROWS_: tl.constexpr,
    VOCAB_ROWS_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    block = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    h_mask = h < HIDDEN_
    gamma = tl.load(gamma_ptr + h, mask=h_mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    acc_sum_vx = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc_sum_v = tl.zeros([BLOCK_H], dtype=tl.float32)

    for i in tl.static_range(0, ROW_BLOCK_):
        row = block * ROW_BLOCK_ + i
        token = row % SEQ_
        row_mask = row < ROWS_
        offsets = row * HIDDEN_ + h
        elem_mask = row_mask & h_mask

        view_value = tl.load(mm52_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        xhat = tl.load(xhat_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
        y = view_value * gamma
        row_sum = tl.sum(y, axis=0)
        row_dot = tl.sum(y * xhat, axis=0)
        scale = tl.load(scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)
        grad = scale * (y * HIDDEN_ - row_sum - xhat * row_dot)

        pos_raw = tl.load(pos_idx_ptr + token, mask=row_mask, other=0).to(tl.int64)
        pos_wrapped = tl.where(pos_raw < 0, pos_raw + POSITION_ROWS_, pos_raw)
        pos_valid = row_mask & (pos_wrapped >= 0) & (pos_wrapped < POSITION_ROWS_) & h_mask
        pos_value = tl.where(pos_raw == -1, full_value / BATCH_, grad)
        tl.atomic_add(
            pos_out_ptr + pos_wrapped * HIDDEN_ + h,
            pos_value,
            sem="relaxed",
            mask=pos_valid,
        )

        segment_raw = tl.load(segment_idx_ptr + token, mask=row_mask, other=0).to(tl.int64)
        segment_wrapped = tl.where(segment_raw < 0, segment_raw + SEGMENT_ROWS_, segment_raw)
        segment_valid = row_mask & (segment_wrapped >= 0) & (segment_wrapped < SEGMENT_ROWS_) & h_mask
        segment_value = tl.where(segment_raw == -1, full_value, grad)
        tl.atomic_add(
            segment_out_ptr + segment_wrapped * HIDDEN_ + h,
            segment_value,
            sem="relaxed",
            mask=segment_valid,
        )

        word_raw = tl.load(word_idx_ptr + row, mask=row_mask, other=0).to(tl.int64)
        word_wrapped = tl.where(word_raw < 0, word_raw + VOCAB_ROWS_, word_raw)
        word_valid = row_mask & (word_wrapped >= 0) & (word_wrapped < VOCAB_ROWS_) & h_mask
        word_value = tl.where(word_raw == 3, full_value, grad)
        tl.atomic_add(
            word_out_ptr + word_wrapped * HIDDEN_ + h,
            word_value,
            sem="relaxed",
            mask=word_valid,
        )

        acc_sum_vx += tl.where(elem_mask, view_value * xhat, 0.0)
        acc_sum_v += tl.where(elem_mask, view_value, 0.0)

    partial_offsets = block * HIDDEN_ + h
    tl.store(partial_sum_vx_ptr + partial_offsets, acc_sum_vx, mask=h_mask)
    tl.store(partial_sum_v_ptr + partial_offsets, acc_sum_v, mask=h_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_sum_vx_ptr,
    partial_sum_v_ptr,
    out_sum_vx_ptr,
    out_sum_v_ptr,
    ROW_BLOCKS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    h = tl.program_id(0) * BLOCK_H + tl.arange(0, BLOCK_H)
    n = tl.arange(0, BLOCK_N)
    mask = (n[:, None] < ROW_BLOCKS_) & (h[None, :] < HIDDEN_)
    offsets = n[:, None] * HIDDEN_ + h[None, :]

    sum_vx = tl.load(partial_sum_vx_ptr + offsets, mask=mask, other=0.0)
    sum_v = tl.load(partial_sum_v_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_sum_vx_ptr + h, tl.sum(sum_vx, axis=0), mask=h < HIDDEN_)
    tl.store(out_sum_v_ptr + h, tl.sum(sum_v, axis=0), mask=h < HIDDEN_)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_52,
        arg4_1,
        arg57_1,
        arg165_1,
        arg2_1,
        full_1,
        arg1_1,
        arg0_1,
        mm_1,
        *_shape_params,
    ) = inputs

    return (
        mm_52.contiguous(),
        arg4_1.contiguous(),
        arg57_1.contiguous(),
        arg165_1.contiguous(),
        arg2_1.contiguous(),
        full_1.contiguous(),
        arg1_1.contiguous(),
        arg0_1.contiguous(),
        mm_1.contiguous(),
    )


def oracle_fused(
    mm_52: torch.Tensor,
    arg4_1: torch.Tensor,
    arg57_1: torch.Tensor,
    arg165_1: torch.Tensor,
    arg2_1: torch.Tensor,
    full_1: torch.Tensor,
    arg1_1: torch.Tensor,
    arg0_1: torch.Tensor,
    mm_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_52.shape == (ROWS, HIDDEN)
    assert arg4_1.shape == (HIDDEN,)
    assert arg57_1.shape == (BATCH, SEQ, HIDDEN)
    assert arg165_1.shape == (BATCH, SEQ, 1)
    assert arg2_1.shape == (1, SEQ)
    assert arg1_1.shape == (1, SEQ)
    assert arg0_1.shape == (BATCH, SEQ)
    assert mm_1.shape == (VOCAB_ROWS, HIDDEN)

    device = mm_52.device
    partial_sum_vx = torch.empty((ROW_BLOCKS, HIDDEN), device=device, dtype=torch.float32)
    partial_sum_v = torch.empty((ROW_BLOCKS, HIDDEN), device=device, dtype=torch.float32)
    pos_out = torch.empty((POSITION_ROWS, HIDDEN), device=device, dtype=torch.float32)
    segment_out = torch.empty((SEGMENT_ROWS, HIDDEN), device=device, dtype=torch.float32)
    word_out = mm_1.clone()
    pos_out.zero_()
    segment_out.zero_()

    _fused_ln_embedding_scatter_kernel[(ROW_BLOCKS,)](
        mm_52,
        arg4_1,
        arg57_1,
        arg165_1,
        arg2_1,
        full_1,
        arg1_1,
        arg0_1,
        partial_sum_vx,
        partial_sum_v,
        pos_out,
        segment_out,
        word_out,
        ROWS_=ROWS,
        SEQ_=SEQ,
        HIDDEN_=HIDDEN,
        BATCH_=BATCH,
        POSITION_ROWS_=POSITION_ROWS,
        SEGMENT_ROWS_=SEGMENT_ROWS,
        VOCAB_ROWS_=VOCAB_ROWS,
        ROW_BLOCK_=ROW_BLOCK,
        BLOCK_H=HIDDEN_BLOCK,
        num_warps=8,
    )

    sum_vx = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    sum_v = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    block_h = 16
    _finalize_column_sums_kernel[(triton.cdiv(HIDDEN, block_h),)](
        partial_sum_vx,
        partial_sum_v,
        sum_vx,
        sum_v,
        ROW_BLOCKS_=ROW_BLOCKS,
        HIDDEN_=HIDDEN,
        BLOCK_N=ROW_BLOCKS,
        BLOCK_H=block_h,
        num_warps=8,
    )

    return sum_vx, sum_v, pos_out, segment_out, word_out


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([16384, 768], f32), T([768], f32), T([32, 512, 768], f32), T([32, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([1, 512], i64, gen=Index(4)), T([32, 512], i64, gen=Index(32000)), T([32000, 768], f32), S([32, 512, 768]), S([32, 512]))")
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
