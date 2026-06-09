"""
Full-scope oracle for sum_sum_sum_4de1559f24a7 (MegatronBert LN/dropout backward).

Gap diagnosis (classification: COOPERATIVE_SPLIT_K): the timed oracle covers
the same scope as `repro.py`, including the three matmul-gradient reshapes, the
shared add, affine multiply, per-row layernorm-backward reductions, dropout mask
scaling, the returned `[1024,8192]` permuted materialized buffer, and all three
returned `[1024]` column reductions. It differs from Inductor by streaming each
row tile once through a Triton producer that computes the dependent row scalars,
writes the side output, and records partials for the sibling column sums instead
of materializing/re-reading the row-local intermediates across separate template
regions. Inductor cannot express this today because the scheduler has to mix a
row reduction, downstream column reductions, and a required materialized side
output; the fix is COOPERATIVE_SPLIT_K support for dependent multi-output
reductions with partial buffers or equivalent atomic coordination.
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



REPRO_ID = "sum_sum_sum_4de1559f24a7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ = 512
C = 1024
ROWS = BATCH * SEQ
DROPOUT_SCALE = 1.1111111111111112



def _full_scope_partial_kernel(
    mm0_ptr,
    mm1_ptr,
    mm2_ptr,
    weight_ptr,
    rhs_ptr,
    row_scale_ptr,
    residual_ptr,
    keep_ptr,
    out_base_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    DROPOUT_SCALE_: tl.constexpr,
    ROW_SPLIT: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    c = tl.arange(0, BLOCK_C)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_x = tl.zeros([BLOCK_C], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_C], dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT, XBLOCK):
        row = pid * ROW_SPLIT + start + tl.arange(0, XBLOCK)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x = x0 + x1 + x2
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        keep = tl.load(keep_ptr + offsets, mask=mask, other=0).to(tl.float32)
        row_scale = tl.load(row_scale_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        out = residual + row_scale[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        out = out * keep * DROPOUT_SCALE_

        tl.store(out_base_ptr + offsets, out, mask=mask)
        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_out += tl.sum(tl.where(mask, out, 0.0), axis=0)

    partial_offsets = pid * C_ + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=c_mask)


@triton.jit
def _finalize_partials_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_sum_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_ROW_BLOCKS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    block = tl.arange(0, BLOCK_ROW_BLOCKS)
    mask = (block[:, None] < NUM_ROW_BLOCKS) & (c[None, :] < C_)
    offsets = block[:, None] * C_ + c[None, :]

    x_rhs = tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c_mask = c < C_
    tl.store(out_x_rhs_ptr + c, tl.sum(x_rhs, axis=0), mask=c_mask)
    tl.store(out_x_ptr + c, tl.sum(x, axis=0), mask=c_mask)
    tl.store(out_sum_ptr + c, tl.sum(out, axis=0), mask=c_mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm_274: torch.Tensor,
    mm_276: torch.Tensor,
    mm_278: torch.Tensor,
    arg11_1: torch.Tensor,
    arg216_1: torch.Tensor,
    arg641_1: torch.Tensor,
    add_137: torch.Tensor,
    arg215_1: torch.Tensor,
    *_shape_params: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert mm_274.shape == (ROWS, C)
    assert mm_276.shape == (ROWS, C)
    assert mm_278.shape == (ROWS, C)
    assert arg11_1.shape == (C,)
    assert arg216_1.shape == (BATCH, SEQ, C)
    assert arg641_1.shape == (BATCH, SEQ, 1)
    assert add_137.shape == (BATCH, SEQ, C)
    assert arg215_1.shape == (BATCH, SEQ, C)

    x0 = mm_274.contiguous()
    x1 = mm_276.contiguous()
    x2 = mm_278.contiguous()
    weight = arg11_1.contiguous()
    rhs = arg216_1.contiguous().reshape(ROWS, C)
    row_scale = arg641_1.contiguous().reshape(ROWS)
    residual = add_137.contiguous().reshape(ROWS, C)
    keep = arg215_1.contiguous().reshape(ROWS, C)

    row_split = 16
    xblock = 1
    block_c = 1024
    num_row_blocks = triton.cdiv(ROWS, row_split)

    partials = torch.empty((3, num_row_blocks, C), device=x0.device, dtype=torch.float32)
    partial_x_rhs = partials[0]
    partial_x = partials[1]
    partial_out = partials[2]
    out_base = torch.empty((ROWS, C), device=x0.device, dtype=torch.float32)

    _full_scope_partial_kernel[(num_row_blocks,)](
        x0,
        x1,
        x2,
        weight,
        rhs,
        row_scale,
        residual,
        keep,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        C_=C,
        DROPOUT_SCALE_=DROPOUT_SCALE,
        ROW_SPLIT=row_split,
        XBLOCK=xblock,
        BLOCK_C=block_c,
        num_warps=8,
    )

    vector_outputs = torch.empty((3, C), device=x0.device, dtype=torch.float32)
    out_x_rhs = vector_outputs[0]
    out_x = vector_outputs[1]
    out_sum = vector_outputs[2]
    finalize_block_c = 16
    _finalize_partials_kernel[(triton.cdiv(C, finalize_block_c),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_sum,
        NUM_ROW_BLOCKS=num_row_blocks,
        C_=C,
        BLOCK_ROW_BLOCKS=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=finalize_block_c,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_base.t(), out_sum


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


@oracle_impl(hardware="H100", shapes="(T([8192, 1024], f32), T([8192, 1024], f32), T([8192, 1024], f32), T([1024], f32), T([16, 512, 1024], f32), T([16, 512, 1], f32), T([16, 512, 1024], f32), T([16, 512, 1024], b8), S([16, 512, 1024]), S([16, 512, 1024]), S([16, 512, 1024]), S([8192, 1024]), S([1024]))")
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
