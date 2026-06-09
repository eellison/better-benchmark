"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Swin layer-norm-backward plus window-partition return tuple from Repro.forward by row-tiling the [128,3136,128] producer, sharing each row's hidden-dimension reductions, writing the returned [128,401408] transposed side output in window-partition layout, and cooperatively accumulating the two input column sums plus the side-output column sum, whereas Inductor currently schedules the row reductions, dependent layer-norm-backward arithmetic, window-partition clone/permute layout, transposed side output, and sibling column reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps row-local scalars, a layout-changing side-output store, and multiple compatible column accumulators in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: add a row-tiled producer mode that fuses layer-norm-backward epilogues with target-layout side stores and finalizes all compatible channel partials together."""
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


REPRO_ID = "sum_sum_sum_e7a56f82f536"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_c225485b"

BATCH = 128
HEIGHT = 56
WIDTH = 56
WINDOW = 7
WINDOWS_PER_SIDE = 8
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
C = 128
INV_C = 1.0 / C

TILE_M = 8
TILE_C = 128
FINAL_BLOCK_C = 16
FINAL_BLOCK_TILES = 512


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    return (value,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_194,
        arg8_1,
        arg188_1,
        arg555_1,
        view_775,
        *_shape_params,
    ) = inputs

    return (
        mm_194.reshape(ROWS, C).contiguous(),
        arg8_1.contiguous(),
        arg188_1.reshape(ROWS, C).contiguous(),
        arg555_1.reshape(ROWS).contiguous(),
        view_775.reshape(ROWS, C).contiguous(),
    )


@triton.jit
def _original_row_to_window_row(
    row,
    TOKENS_: tl.constexpr,
    WIDTH_: tl.constexpr,
    WINDOW_: tl.constexpr,
    WINDOWS_PER_SIDE_: tl.constexpr,
):
    batch = row // TOKENS_
    token = row - batch * TOKENS_
    h = token // WIDTH_
    w = token - h * WIDTH_
    h_block = h // WINDOW_
    w_block = w // WINDOW_
    h_inner = h - h_block * WINDOW_
    w_inner = w - w_block * WINDOW_
    window_id = (batch * WINDOWS_PER_SIDE_ + h_block) * WINDOWS_PER_SIDE_ + w_block
    return window_id * (WINDOW_ * WINDOW_) + h_inner * WINDOW_ + w_inner


@triton.jit
def _row_tile_store_and_reduce_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_side_ptr,
    side_transposed_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    WIDTH_: tl.constexpr,
    WINDOW_: tl.constexpr,
    WINDOWS_PER_SIDE_: tl.constexpr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile_m = tl.program_id(0)
    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    row_mask = rows < ROWS_
    col_mask = cols < C_
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * C_ + cols[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    grad = scale[:, None] * (weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None])
    side = residual + grad

    window_rows = _original_row_to_window_row(
        rows,
        TOKENS_=TOKENS_,
        WIDTH_=WIDTH_,
        WINDOW_=WINDOW_,
        WINDOWS_PER_SIDE_=WINDOWS_PER_SIDE_,
    )
    side_offsets = window_rows[:, None] * C_ + cols[None, :]
    tl.store(side_transposed_ptr + side_offsets, side, mask=mask)

    partial_offsets = tile_m * C_ + cols
    tl.store(
        partial_x_rhs_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * rhs, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=col_mask,
    )
    tl.store(
        partial_side_ptr + partial_offsets,
        tl.sum(tl.where(mask, side, 0.0), axis=0),
        mask=col_mask,
    )


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_side_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_side_ptr,
    NUM_ROW_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < C_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_side = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & col_mask[None, :]
        offsets = tiles[:, None] * C_ + cols[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_side += tl.sum(
            tl.load(partial_side_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + cols, acc_x_rhs, mask=col_mask)
    tl.store(out_x_ptr + cols, acc_x, mask=col_mask)
    tl.store(out_side_ptr + cols, acc_side, mask=col_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (ROWS, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (ROWS, C)
    assert scale_m.shape == (ROWS,)
    assert residual_mc.shape == (ROWS, C)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()

    device = x_mc.device
    num_row_tiles = triton.cdiv(ROWS, TILE_M)
    partial_x_rhs = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    partial_side = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    side_transposed = torch.empty_strided(
        (C, ROWS),
        (1, C),
        device=device,
        dtype=torch.float32,
    )

    _row_tile_store_and_reduce_kernel[(num_row_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        partial_x_rhs,
        partial_x,
        partial_side,
        side_transposed,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOWS_PER_SIDE_=WINDOWS_PER_SIDE,
        C_=C,
        INV_C_=INV_C,
        BLOCK_M=TILE_M,
        BLOCK_C=TILE_C,
        num_warps=4,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_side = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_side,
        out_x_rhs,
        out_x,
        out_side,
        NUM_ROW_TILES=num_row_tiles,
        C_=C,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
    )

    return out_x_rhs, out_x, side_transposed, out_side


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


@oracle_impl(hardware="H100", shapes="(T([401408, 128], f32), T([128], f32), T([128, 3136, 128], f32), T([128, 3136, 1], f32), T([128, 3136, 128], f32), S([128, 3136, 128]), S([128, 56, 56, 128]), S([128, 8, 7, 8, 7, 128]), S([8192, 7, 7, 128]), S([8192, 49, 128]), S([401408, 128]), S([128]))")
def oracle_forward(inputs):
    """Thin wrapper for oracle_harness compatibility."""
    return oracle_full(*inputs)


def main() -> None:
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
