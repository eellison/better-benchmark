"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete Swin MLP layernorm-backward/window-reverse return tuple by row-tiling the `[401408, 128]` producer, computing the row-local 128-wide layernorm-backward reductions once, writing the returned non-contiguous `[128, 401408]` transpose side output in the structured window-reverse layout, and cooperatively accumulating the two pre-window `[128]` column reductions plus the final `[128]` reduction over the side output, whereas Inductor currently schedules the sibling column sums, row reductions, dependent pointwise epilogue, clone/permute/reshape window layout work, transpose view, and downstream reduction as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output reduction template that coordinates row-local reductions, structured window-layout side stores, and multiple dependent column accumulators in one producer/finalizer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible column reductions across row/window tiles, keep row-local layernorm summaries in registers, and fuse the window-reverse transpose store with finalization of all column sums."""
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


REPRO_ID = "sum_sum_sum_e1f39649886f"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_c225485b"

BATCH = 128
HEIGHT = 56
WIDTH = 56
WINDOW = 7
WINDOWS_H = HEIGHT // WINDOW
WINDOWS_W = WIDTH // WINDOW
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
C = 128

ROW_SPLIT = 16
XBLOCK = 1
TILE_C = 128
FINAL_BLOCK_C = 16
FINAL_BLOCK_TILES = 256
KERNEL_WARPS = 4


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_197,
        primals_14,
        mul_5,
        div_121,
        view_1383,
        *_shape_params,
    ) = inputs

    return (
        mm_197.reshape(ROWS, C).contiguous(),
        primals_14.contiguous(),
        mul_5.reshape(ROWS, C).contiguous(),
        div_121.reshape(ROWS).contiguous(),
        view_1383.reshape(ROWS, C).contiguous(),
    )


@triton.jit
def _swin_ln_window_partial_kernel(
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
    C_: tl.constexpr,
    TOKENS_: tl.constexpr,
    WIDTH_: tl.constexpr,
    WINDOW_: tl.constexpr,
    WINDOWS_H_: tl.constexpr,
    WINDOWS_W_: tl.constexpr,
    ROW_SPLIT_: tl.constexpr,
    XBLOCK_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    tile = tl.program_id(0)
    c = tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    acc_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_out = tl.zeros((BLOCK_C_,), dtype=tl.float32)

    for start in tl.range(0, ROW_SPLIT_, XBLOCK_):
        row = tile * ROW_SPLIT_ + start + tl.arange(0, XBLOCK_)
        row_mask = row < ROWS_
        mask = row_mask[:, None] & c_mask[None, :]
        offsets = row[:, None] * C_ + c[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + row, mask=row_mask, other=0.0).to(tl.float32)

        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
        out = residual + gate[:, None] * (
            weighted * C_ - row_sum[:, None] - rhs * row_dot[:, None]
        )

        batch = row // TOKENS_
        token = row - batch * TOKENS_
        h = token // WIDTH_
        w = token - h * WIDTH_
        hwin = h // WINDOW_
        wh = h - hwin * WINDOW_
        wwin = w // WINDOW_
        ww = w - wwin * WINDOW_
        final_row = (
            (((batch * WINDOWS_H_ + hwin) * WINDOWS_W_ + wwin) * WINDOW_ + wh)
            * WINDOW_
            + ww
        )
        tl.store(out_base_ptr + final_row[:, None] * C_ + c[None, :], out, mask=mask)

        acc_x_rhs += tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        acc_x += tl.sum(tl.where(mask, x, 0.0), axis=0)
        acc_out += tl.sum(tl.where(mask, out, 0.0), axis=0)

    partial_offsets = tile * C_ + c
    tl.store(partial_x_rhs_ptr + partial_offsets, acc_x_rhs, mask=c_mask)
    tl.store(partial_x_ptr + partial_offsets, acc_x, mask=c_mask)
    tl.store(partial_out_ptr + partial_offsets, acc_out, mask=c_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_out_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_final_ptr,
    NUM_ROW_TILES: tl.constexpr,
    C_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    c_mask = c < C_
    tile_offsets = tl.arange(0, BLOCK_TILES_)

    acc_x_rhs = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    acc_out = tl.zeros((BLOCK_C_,), dtype=tl.float32)
    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES_):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & c_mask[None, :]
        offsets = tiles[:, None] * C_ + c[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_out += tl.sum(
            tl.load(partial_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + c, acc_x_rhs, mask=c_mask)
    tl.store(out_x_ptr + c, acc_x, mask=c_mask)
    tl.store(out_final_ptr + c, acc_out, mask=c_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    gate_m: torch.Tensor,
    residual_mc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (ROWS, C)
    assert weight_c.shape == (C,)
    assert rhs_mc.shape == (ROWS, C)
    assert gate_m.shape == (ROWS,)
    assert residual_mc.shape == (ROWS, C)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert gate_m.is_contiguous()
    assert residual_mc.is_contiguous()

    device = x_mc.device
    num_row_tiles = triton.cdiv(ROWS, ROW_SPLIT)
    partial_x_rhs = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    partial_x = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    partial_out = torch.empty((num_row_tiles, C), device=device, dtype=torch.float32)
    out_base = torch.empty((ROWS, C), device=device, dtype=torch.float32)

    _swin_ln_window_partial_kernel[(num_row_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        gate_m,
        residual_mc,
        out_base,
        partial_x_rhs,
        partial_x,
        partial_out,
        ROWS_=ROWS,
        C_=C,
        TOKENS_=TOKENS,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOWS_H_=WINDOWS_H,
        WINDOWS_W_=WINDOWS_W,
        ROW_SPLIT_=ROW_SPLIT,
        XBLOCK_=XBLOCK,
        BLOCK_C_=TILE_C,
        num_warps=KERNEL_WARPS,
    )

    out_x_rhs = torch.empty((C,), device=device, dtype=torch.float32)
    out_x = torch.empty((C,), device=device, dtype=torch.float32)
    out_final = torch.empty((C,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_x_rhs,
        partial_x,
        partial_out,
        out_x_rhs,
        out_x,
        out_final,
        NUM_ROW_TILES=num_row_tiles,
        C_=C,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=KERNEL_WARPS,
    )

    return out_x_rhs, out_x, out_base.permute(1, 0), out_final


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


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
