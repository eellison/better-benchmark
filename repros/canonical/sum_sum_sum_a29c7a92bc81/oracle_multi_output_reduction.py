"""
Canonical oracle for sum_sum_sum_a29c7a92bc81 (Swin patch-expand backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this full-scope oracle
differs from Inductor by making the final patch-expand/drop-path producer also
cooperatively reduce the returned `[401408, 128]` buffer into the sibling
`[128]` output, while separate kernels compute the two shared `[512]` column
reductions and the per-row dual reduction needed by the producer. Inductor
cannot do this today because the final `sum(dim=0)` has a tiny output and a
large reduction extent, but is attached to a required materialized
transpose/reshape side output, so the scheduler keeps the side output and
reduction as ordinary producer/consumer work instead of emitting one split-K
producer with partial coordination. The fix is COOPERATIVE_SPLIT_K support for
reductions with materialized side outputs, so a pointwise producer can write
the returned buffer and accumulate the sibling reduction across row tiles
without rereading the buffer.
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



REPRO_ID = "sum_sum_sum_a29c7a92bc81"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 128
H = 28
W = 28
C = 512
K = 128
ROWS = N * H * W
OUT_H = H * 2
OUT_W = W * 2
OUT_ROWS_PER_N = OUT_H * OUT_W
OUT_ROWS = N * OUT_ROWS_PER_N
DROPPATH_SCALE = 1.0 / 0.9956521736457944


def _summary_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
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

    acc_sum_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_sum_x = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        xw = x * weight[None, :]

        tl.store(row_sum_ptr + row, tl.sum(tl.where(mask, xw, 0.0), axis=1), mask=row_mask)
        tl.store(row_dot_ptr + row, tl.sum(tl.where(mask, xw * rhs, 0.0), axis=1), mask=row_mask)

        acc_sum_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_sum_x += tl.sum(tl.where(mask, x, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_sum_x_rhs_ptr + partial_offsets, acc_sum_x_rhs, mask=c_mask)
    tl.store(partial_sum_x_ptr + partial_offsets, acc_sum_x, mask=c_mask)


@triton.jit
def _finalize_column_partials_kernel(
    partial_sum_x_rhs_ptr,
    partial_sum_x_ptr,
    out_sum_x_rhs_ptr,
    out_sum_x_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    sum_x_rhs = tl.load(partial_sum_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(partial_sum_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_x_rhs_ptr + c, tl.sum(sum_x_rhs, axis=0), mask=c < C_)
    tl.store(out_sum_x_ptr + c, tl.sum(sum_x, axis=0), mask=c < C_)


@triton.jit
def _patch_expand_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    gate_ptr,
    keep_ptr,
    row_sum_ptr,
    row_dot_ptr,
    out_base_ptr,
    partial_out_sum_ptr,
    OUT_ROWS_: tl.constexpr,
    OUT_ROWS_PER_N_: tl.constexpr,
    OUT_W_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    C_: tl.constexpr,
    K_: tl.constexpr,
    DROPPATH_SCALE_: tl.constexpr,
    BLOCK_OUT_ROWS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    out_row = tl.program_id(0) * BLOCK_OUT_ROWS + tl.arange(0, BLOCK_OUT_ROWS)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)

    row_mask = out_row < OUT_ROWS_
    k_mask = k < K_
    mask = row_mask[:, None] & k_mask[None, :]

    n = out_row // OUT_ROWS_PER_N_
    inner = out_row - n * OUT_ROWS_PER_N_
    out_h = inner // OUT_W_
    out_w = inner - out_h * OUT_W_
    src_h = out_h // 2
    src_w = out_w // 2
    split_b = out_h - src_h * 2
    split_a = out_w - src_w * 2

    src_row = n * (H_ * W_) + src_h * W_ + src_w
    c = split_a[:, None] * 256 + split_b[:, None] * 128 + k[None, :]
    src_offsets = src_row[:, None] * C_ + c

    x = tl.load(x_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + src_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)
    keep = tl.load(keep_ptr + n, mask=row_mask, other=0).to(tl.float32) * DROPPATH_SCALE_
    row_sum = tl.load(row_sum_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + src_row, mask=row_mask, other=0.0).to(tl.float32)

    value = gate[:, None] * (x * weight * 512.0 - row_sum[:, None] - rhs * row_dot[:, None])
    value = value * keep[:, None]

    out_offsets = out_row[:, None] * K_ + k[None, :]
    tl.store(out_base_ptr + out_offsets, value, mask=mask)

    partial = tl.sum(tl.where(mask, value, 0.0), axis=0)
    partial_offsets = tl.program_id(0) * K_ + k
    tl.store(partial_out_sum_ptr + partial_offsets, partial, mask=k_mask)


@triton.jit
def _finalize_out_sum_kernel(
    partial_out_sum_ptr,
    out_sum_ptr,
    NUM_OUT_ROW_BLOCKS: tl.constexpr,
    K_: tl.constexpr,
    BLOCK_OUT_ROW_BLOCKS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    k = tl.program_id(0) * BLOCK_K + tl.arange(0, BLOCK_K)
    block = tl.arange(0, BLOCK_OUT_ROW_BLOCKS)
    mask = (block[:, None] < NUM_OUT_ROW_BLOCKS) & (k[None, :] < K_)
    offsets = block[:, None] * K_ + k[None, :]

    partial = tl.load(partial_out_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + k, tl.sum(partial, axis=0), mask=k < K_)


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_186,
        primals_35,
        mul_20,
        div_118,
        lt_1,
        *_shape_params,
    ) = inputs

    return (
        mm_186.contiguous(),
        primals_35.contiguous(),
        mul_20.contiguous(),
        div_118.reshape(ROWS).contiguous(),
        lt_1.reshape(N).contiguous(),
    )


def oracle_fused(
    mm_186: torch.Tensor,
    primals_35: torch.Tensor,
    mul_20: torch.Tensor,
    div_118_flat: torch.Tensor,
    lt_1_flat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_186.shape == (ROWS, C)
    assert primals_35.shape == (C,)
    assert mul_20.shape == (N, H, W, C)
    assert div_118_flat.shape == (ROWS,)
    assert lt_1_flat.shape == (N,)

    rhs = mul_20.reshape(ROWS, C)
    device = mm_186.device

    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_split = 128
    xblock = 2
    num_summary_blocks = triton.cdiv(ROWS, row_split)
    partial_sum_x_rhs = torch.empty((num_summary_blocks, C), device=device, dtype=torch.float32)
    partial_sum_x = torch.empty((num_summary_blocks, C), device=device, dtype=torch.float32)
    _summary_reduce_kernel[(num_summary_blocks,)](
        mm_186,
        primals_35,
        rhs,
        row_sum,
        row_dot,
        partial_sum_x_rhs,
        partial_sum_x,
        ROWS_=ROWS,
        C_=C,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=512,
        num_warps=1,
    )

    sum_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    sum_x = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_partials_kernel[(triton.cdiv(C, 16),)](
        partial_sum_x_rhs,
        partial_sum_x,
        sum_x_rhs,
        sum_x,
        NUM_ROW_BLOCKS=num_summary_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=1024,
        BLOCK_C=16,
        num_warps=8,
    )

    out_base = torch.empty((OUT_ROWS, K), device=device, dtype=torch.float32)
    block_out_rows = 64
    block_k = 64
    num_out_row_blocks = triton.cdiv(OUT_ROWS, block_out_rows)
    partial_out_sum = torch.empty((num_out_row_blocks, K), device=device, dtype=torch.float32)
    _patch_expand_store_and_reduce_kernel[
        (num_out_row_blocks, triton.cdiv(K, block_k))
    ](
        mm_186,
        primals_35,
        rhs,
        div_118_flat,
        lt_1_flat,
        row_sum,
        row_dot,
        out_base,
        partial_out_sum,
        OUT_ROWS_=OUT_ROWS,
        OUT_ROWS_PER_N_=OUT_ROWS_PER_N,
        OUT_W_=OUT_W,
        H_=H,
        W_=W,
        C_=C,
        K_=K,
        DROPPATH_SCALE_=DROPPATH_SCALE,
        BLOCK_OUT_ROWS=block_out_rows,
        BLOCK_K=block_k,
        num_warps=4,
    )
    out_sum = torch.empty((K,), device=device, dtype=torch.float32)
    _finalize_out_sum_kernel[(triton.cdiv(K, 8),)](
        partial_out_sum,
        out_sum,
        NUM_OUT_ROW_BLOCKS=num_out_row_blocks,
        K_=K,
        BLOCK_OUT_ROW_BLOCKS=8192,
        BLOCK_K=8,
        num_warps=8,
    )

    return sum_x_rhs, sum_x, out_base.permute(1, 0), out_sum


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


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
