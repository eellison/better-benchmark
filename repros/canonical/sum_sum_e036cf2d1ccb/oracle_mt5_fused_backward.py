"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full MT5 dropout/layer-norm-backward return tuple by row-tiling the `[4096, 512]` producer, sharing the row-local reduction needed by the transposed `[512, 4096]` gradient output, and emitting partial column sums for the returned `[512]` vector from the same element pass, whereas Inductor currently schedules the row reduction, dependent pointwise epilogue, transposed side-output write, and sibling column reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, required layout-changing stores, and compatible column accumulators in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer that fuses the layer-norm-backward epilogue, writes the side output in target stride, emits column partials, and finalizes them together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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


REPRO_ID = "sum_sum_e036cf2d1ccb"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

M = 4096
D = 512
SCALE = 1.1111111111111112
INV_D = 1.0 / D

ROW_TILE = 4
BLOCK_D = 512
FINAL_BLOCK_D = 8
FINAL_BLOCK_TILES = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _row_tile_kernel(
        mm_ptr,
        mask1_ptr,
        weight_ptr,
        arg_ptr,
        row_scale_ptr,
        mask2_ptr,
        partial_col_ptr,
        out_transposed_ptr,
        M_: tl.constexpr,
        D_: tl.constexpr,
        SCALE_: tl.constexpr,
        INV_D_: tl.constexpr,
        ROW_TILE_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        tile_m = tl.program_id(0)
        cols = tl.arange(0, BLOCK_D_)
        col_mask = cols < D_
        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        partial_col = tl.zeros((BLOCK_D_,), dtype=tl.float32)

        for row_i in tl.range(0, ROW_TILE_):
            row = tile_m * ROW_TILE_ + row_i
            mask = (row < M_) & col_mask
            offsets = row * D_ + cols

            x = tl.load(mm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            mask1 = tl.load(mask1_ptr + offsets, mask=mask, other=0).to(tl.float32)
            mask2 = tl.load(mask2_ptr + offsets, mask=mask, other=0).to(tl.float32)
            arg = tl.load(arg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            row_scale = tl.load(row_scale_ptr + row, mask=row < M_, other=0.0).to(tl.float32)

            masked_x = x * mask1 * SCALE_
            weighted = masked_x * weight
            partial_col += tl.where(mask, masked_x * arg * row_scale, 0.0)
            row_sum = tl.sum(tl.where(mask, weighted * arg, 0.0), axis=0)

            row_scale_sq = row_scale * row_scale
            row_scale_cu = row_scale_sq * row_scale
            correction = ((row_sum * -0.5) * row_scale_cu * INV_D_) * (arg * 2.0)
            direct = weighted * row_scale
            out = (direct + correction) * (mask2 * SCALE_)
            tl.store(out_transposed_ptr + offsets, out, mask=mask)

        tl.store(partial_col_ptr + tile_m * D_ + cols, partial_col, mask=col_mask)


    @triton.jit
    def _finalize_column_sum_kernel(
        partial_col_ptr,
        out_col_ptr,
        NUM_ROW_TILES_: tl.constexpr,
        D_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_D_: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_D_ + tl.arange(0, BLOCK_D_)
        col_mask = cols < D_
        tiles = tl.arange(0, BLOCK_TILES_)
        acc = tl.zeros((BLOCK_D_,), dtype=tl.float32)

        for tile_start in range(0, NUM_ROW_TILES_, BLOCK_TILES_):
            tile_ids = tile_start + tiles
            mask = (tile_ids[:, None] < NUM_ROW_TILES_) & col_mask[None, :]
            offsets = tile_ids[:, None] * D_ + cols[None, :]
            values = tl.load(partial_col_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            acc += tl.sum(values, axis=0)

        tl.store(out_col_ptr + cols, acc, mask=col_mask)


def oracle_triton(
    mm_1: torch.Tensor,
    arg498_1: torch.Tensor,
    arg188_1: torch.Tensor,
    arg496_1: torch.Tensor,
    arg497_1: torch.Tensor,
    arg495_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm_1.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert mm_1.shape == (M, D)
    assert arg498_1.shape == (32, 128, D)
    assert arg188_1.shape == (D,)
    assert arg496_1.shape == (32, 128, D)
    assert arg497_1.shape == (32, 128, 1)
    assert arg495_1.shape == (32, 128, D)
    assert mm_1.is_contiguous()
    assert arg498_1.is_contiguous()
    assert arg188_1.is_contiguous()
    assert arg496_1.is_contiguous()
    assert arg497_1.is_contiguous()
    assert arg495_1.is_contiguous()

    device = mm_1.device
    num_row_tiles = triton.cdiv(M, ROW_TILE)
    partial_col = torch.empty((num_row_tiles, D), device=device, dtype=torch.float32)
    out_col = torch.empty((D,), device=device, dtype=torch.float32)
    out_transposed = torch.empty_strided((D, M), (1, D), device=device, dtype=torch.float32)

    _row_tile_kernel[(num_row_tiles,)](
        mm_1,
        arg498_1,
        arg188_1,
        arg496_1,
        arg497_1,
        arg495_1,
        partial_col,
        out_transposed,
        M_=M,
        D_=D,
        SCALE_=SCALE,
        INV_D_=INV_D,
        ROW_TILE_=ROW_TILE,
        BLOCK_D_=BLOCK_D,
        num_warps=4,
    )

    _finalize_column_sum_kernel[(triton.cdiv(D, FINAL_BLOCK_D),)](
        partial_col,
        out_col,
        NUM_ROW_TILES_=num_row_tiles,
        D_=D,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_D_=FINAL_BLOCK_D,
        num_warps=8,
    )

    return out_col, out_transposed


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward()."""
    (
        mm_1,
        arg498_1,
        arg188_1,
        arg496_1,
        arg497_1,
        arg495_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    return oracle_triton(mm_1, arg498_1, arg188_1, arg496_1, arg497_1, arg495_1)


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
