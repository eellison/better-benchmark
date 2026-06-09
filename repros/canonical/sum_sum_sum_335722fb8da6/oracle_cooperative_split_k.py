"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Qwen structured two-head pool/RoPE/RMSNorm-backward return tuple by gathering and reducing the paired input heads inside a row-tiled producer, writing the returned `[1024, 2048]` transposed side output, and cooperatively accumulating the sibling `[128]` column reduction from the same full-scope producer, whereas Inductor schedules the structured pool, slice-scatter rotation, row reduction, transpose clone, and column reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output template that combines a structured gather-reduce producer, row-local reductions, a layout-changing side output, and sibling column accumulation; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile compatible row and column reductions together, recompute cheap structured pool values in the tile, and finalize shared partial accumulators while fusing the transpose side store."""
from __future__ import annotations

import argparse
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
HEADS = 8
PAIR = 2
DIM = 128
HALF_DIM = 64
ROWS = BATCH * SEQ * HEADS
OUT_ROWS = BATCH * SEQ
OUT_COLS = HEADS * DIM
INV_DIM = 1.0 / DIM

BLOCK_M = 16
BLOCK_D = 128
FINAL_BLOCK_M = 2048
FINAL_BLOCK_D = 16


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _producer_kernel(
    getitem_ptr,
    cos_ptr,
    full_ptr,
    sin_ptr,
    weight_ptr,
    normed_ptr,
    rstd_ptr,
    partial_ptr,
    out_t_ptr,
    ROWS_: tl.constexpr,
    SEQ_: tl.constexpr,
    HEADS_: tl.constexpr,
    PAIR_: tl.constexpr,
    DIM_: tl.constexpr,
    HALF_DIM_: tl.constexpr,
    OUT_COLS_: tl.constexpr,
    INV_DIM_: tl.constexpr,
    BLOCK_M_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M_ + tl.arange(0, BLOCK_M_)
    d = tl.arange(0, BLOCK_D_)
    mask = row[:, None] < ROWS_

    b = row // (SEQ_ * HEADS_)
    rem = row - b * (SEQ_ * HEADS_)
    s = rem // HEADS_
    h = rem - s * HEADS_

    in_base0 = ((b[:, None] * (HEADS_ * PAIR_) + (h[:, None] * PAIR_)) * SEQ_ + s[:, None]) * DIM_
    in_base1 = in_base0 + SEQ_ * DIM_
    elem_offsets = d[None, :]

    a0 = tl.load(getitem_ptr + in_base0 + elem_offsets, mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(getitem_ptr + in_base1 + elem_offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = (a0 + a1).to(tl.bfloat16).to(tl.float32)

    trig_offsets = s[:, None] * DIM_ + elem_offsets
    cos = tl.load(cos_ptr + trig_offsets, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + trig_offsets, mask=mask, other=0.0).to(tl.float32)
    full = tl.load(full_ptr + ((b[:, None] * HEADS_ + h[:, None]) * SEQ_ + s[:, None]) * DIM_ + elem_offsets, mask=mask, other=0.0).to(tl.float32)

    other_d = tl.where(d < HALF_DIM_, d + HALF_DIM_, d - HALF_DIM_)
    rotate_src = tl.load(
        getitem_ptr + in_base0 + other_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32) + tl.load(
        getitem_ptr + in_base1 + other_d[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    rotate_src = rotate_src.to(tl.bfloat16).to(tl.float32)
    rotate_cos = tl.load(cos_ptr + s[:, None] * DIM_ + other_d[None, :], mask=mask, other=0.0).to(tl.float32)
    rotate_src = (rotate_src * rotate_cos).to(tl.bfloat16).to(tl.float32)
    rotated = tl.where(d[None, :] < HALF_DIM_, rotate_src, -rotate_src)

    rotated_add = (full + rotated).to(tl.bfloat16).to(tl.float32)
    sin_term = (pooled * sin).to(tl.bfloat16).to(tl.float32)
    permuted = (rotated_add + sin_term).to(tl.bfloat16).to(tl.float32)

    weight = tl.load(weight_ptr + d, mask=d < DIM_, other=0.0).to(tl.float32)
    grad_bf = (permuted * weight[None, :]).to(tl.bfloat16).to(tl.float32)

    normed = tl.load(normed_ptr + row[:, None] * DIM_ + elem_offsets, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.load(rstd_ptr + row, mask=row < ROWS_, other=0.0).to(tl.float32)
    row_sum = tl.sum(tl.where(mask, grad_bf * normed, 0.0), axis=1)
    rstd_cubed = rstd * rstd * rstd
    mul_scalar = row_sum * -0.5
    mul7 = mul_scalar * rstd_cubed
    div = mul7 * INV_DIM_
    mul_scalar1 = normed * 2.0
    out = grad_bf * rstd[:, None] + div[:, None] * mul_scalar1

    out_row = b[:, None] * SEQ_ + s[:, None]
    out_col = h[:, None] * DIM_ + elem_offsets
    tl.store(out_t_ptr + out_col + out_row * OUT_COLS_, out.to(tl.bfloat16), mask=mask)

    normed_scaled_bf = (normed * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
    partial_terms = (permuted * normed_scaled_bf).to(tl.bfloat16).to(tl.float32)
    partial = tl.sum(tl.where(mask, partial_terms, 0.0), axis=0)
    tl.store(partial_ptr + tl.program_id(0) * DIM_ + d, partial, mask=d < DIM_)


@triton.jit
def _finalize_kernel(
    partial_ptr,
    out0_ptr,
    NUM_ROW_TILES: tl.constexpr,
    DIM_: tl.constexpr,
    BLOCK_M_: tl.constexpr,
    BLOCK_D_: tl.constexpr,
):
    d = tl.program_id(0) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
    tile = tl.arange(0, BLOCK_M_)
    mask = (tile[:, None] < NUM_ROW_TILES) & (d[None, :] < DIM_)
    vals = tl.load(partial_ptr + tile[:, None] * DIM_ + d[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(out0_ptr + d, tl.sum(vals, axis=0).to(tl.bfloat16), mask=d < DIM_)


@oracle_impl(hardware="H100", shapes="(T([4, 16, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([4, 8, 512, 128], bf16), T([1, 1, 512, 128], bf16), T([128], bf16), T([2048, 1024], bf16), T([4, 512, 8, 1], f32), S([4, 8, 2, 512, 128]), S([4, 512, 1024]), S([4, 512, -1, 128]), S([128]), S([4, 512, 8, 128]), S([4, 512, 1024]), S([2048, 1024]))")
def oracle_forward(inputs):
    """Run the full-scope Triton oracle for Repro.forward()."""
    (
        getitem_82,
        unsqueeze_6,
        full_2,
        unsqueeze_7,
        arg7_1,
        arg317_1,
        arg318_1,
        *_shape_params,
    ) = inputs

    if getitem_82.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    num_row_tiles = triton.cdiv(ROWS, BLOCK_M)
    partial = torch.empty((num_row_tiles, DIM), device=getitem_82.device, dtype=torch.float32)
    out0 = torch.empty((DIM,), device=getitem_82.device, dtype=torch.bfloat16)
    out1 = torch.empty_strided(
        (OUT_COLS, OUT_ROWS),
        (1, OUT_COLS),
        device=getitem_82.device,
        dtype=torch.bfloat16,
    )

    _producer_kernel[(num_row_tiles,)](
        getitem_82,
        unsqueeze_6,
        full_2,
        unsqueeze_7,
        arg7_1,
        arg317_1,
        arg318_1,
        partial,
        out1,
        ROWS_=ROWS,
        SEQ_=SEQ,
        HEADS_=HEADS,
        PAIR_=PAIR,
        DIM_=DIM,
        HALF_DIM_=HALF_DIM,
        OUT_COLS_=OUT_COLS,
        INV_DIM_=INV_DIM,
        BLOCK_M_=BLOCK_M,
        BLOCK_D_=BLOCK_D,
        num_warps=4,
    )
    _finalize_kernel[(triton.cdiv(DIM, FINAL_BLOCK_D),)](
        partial,
        out0,
        NUM_ROW_TILES=num_row_tiles,
        DIM_=DIM,
        BLOCK_M_=FINAL_BLOCK_M,
        BLOCK_D_=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out0, out1


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-1,
                        help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=5e-1,
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
