"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Swin shifted-window layer-norm-backward/drop-path return tuple by row-tiling the `[25088, 512]` producer, cooperatively accumulating the two original `[512]` column reductions, mapping the arbitrary `fmod` advanced-index/window-reverse layout for the returned non-contiguous `[512, 25088]` transposed side output, and finalizing the post-index `[512]` column reduction, whereas Inductor currently schedules the hidden-dimension row reductions, stochastic-depth epilogue, two advanced-index gathers, window layout change, transposed side-output store, and sibling `sum([0, 1])`/`sum([0])` reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reduction scalars, arbitrary gather/window layout remapping, a dependent full-tensor side store, and multiple compatible column accumulators in one producer/finalizer pair; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the shifted-window gather and transpose store into the producer, and finalize all sibling column partials together."""
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


REPRO_ID = "sum_sum_sum_580d823755af"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_d4777b22"

BATCH = 128
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS_W = WIDTH // WINDOW
WINDOWS_PER_IMAGE = (HEIGHT // WINDOW) * (WIDTH // WINDOW)
WINDOW_AREA = WINDOW * WINDOW
HW = HEIGHT * WIDTH
ROWS = BATCH * HW
CHANNELS = 512
KEEP_PROB = 0.9782608672976494

TILE_ROWS = 8
TILE_SELECTED = 4
TILE_CHANNELS = 512
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 256


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
        value.cuda() if isinstance(value, torch.Tensor) and value.device.type != "cuda" else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    model = module.Repro().to(first_tensor.device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _build_window_maps(
    fmod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    fmod_values = [int(value) for value in fmod.detach().cpu().tolist()]
    if len(fmod_values) != HEIGHT:
        raise ValueError(f"expected fmod shape [{HEIGHT}], got {list(fmod.shape)}")
    if any(value < 0 or value >= HEIGHT for value in fmod_values):
        raise ValueError(f"fmod values must be in [0, {HEIGHT})")

    out_locs_by_src = [[] for _ in range(HW)]
    for indexed_h in range(HEIGHT):
        src_h = fmod_values[indexed_h]
        block_h = indexed_h // WINDOW
        inner_h = indexed_h % WINDOW
        for indexed_w in range(WIDTH):
            src_w = fmod_values[indexed_w]
            block_w = indexed_w // WINDOW
            inner_w = indexed_w % WINDOW
            src_local = src_h * WIDTH + src_w
            out_local = (
                ((block_h * WINDOW_BLOCKS_W + block_w) * WINDOW + inner_h)
                * WINDOW
                + inner_w
            )
            out_locs_by_src[src_local].append(out_local)

    starts = [0]
    selected_src_locs = []
    for src_local, out_locs in enumerate(out_locs_by_src):
        if out_locs:
            selected_src_locs.append(src_local)
        starts.append(starts[-1] + len(out_locs))

    flat_out_locs = [out_local for out_locs in out_locs_by_src for out_local in out_locs]
    if len(flat_out_locs) != HW:
        raise AssertionError("window index map must cover each local output row once")

    max_occ = max((len(out_locs) for out_locs in out_locs_by_src), default=1)
    selected_tile = 1 if max_occ > 32 else min(TILE_SELECTED, len(selected_src_locs))
    if selected_tile <= 0:
        selected_tile = 1

    device = fmod.device
    return (
        torch.tensor(starts, device=device, dtype=torch.int64),
        torch.tensor(flat_out_locs, device=device, dtype=torch.int64),
        torch.tensor(selected_src_locs, device=device, dtype=torch.int64),
        max_occ,
        selected_tile,
    )


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_150,
        arg47_1,
        arg248_1,
        arg530_1,
        view_600,
        arg247_1,
        fmod,
        *_shape_params,
    ) = inputs

    starts, out_locs, selected_src_locs, max_occ, selected_tile = _build_window_maps(fmod)
    return (
        mm_150.reshape(ROWS, CHANNELS).contiguous(),
        arg47_1.contiguous(),
        arg248_1.reshape(ROWS, CHANNELS).contiguous(),
        arg530_1.reshape(ROWS).contiguous(),
        view_600.reshape(ROWS, CHANNELS).contiguous(),
        arg247_1.reshape(BATCH).contiguous(),
        starts,
        out_locs,
        selected_src_locs,
        max_occ,
        selected_tile,
    )


@triton.jit
def _row_reduction_partials_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_x_rhs_ptr,
    partial_x_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]

    source_offsets = rows[:, None] * CHANNELS_ + channels[None, :]
    x = tl.load(x_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    weighted = x * weight[None, :]

    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * rhs, 0.0), axis=1)
    tl.store(row_sum_ptr + rows, row_sum, mask=row_mask)
    tl.store(row_dot_ptr + rows, row_dot, mask=row_mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    tl.store(partial_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=channel_mask)
    tl.store(partial_x_ptr + partial_offsets, sum_x, mask=channel_mask)


@triton.jit
def _selected_window_output_kernel(
    x_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    drop_mask_ptr,
    starts_ptr,
    out_locs_ptr,
    selected_src_locs_ptr,
    row_sum_ptr,
    row_dot_ptr,
    partial_dropped_ptr,
    out_transposed_ptr,
    NUM_SELECTED: tl.constexpr,
    SELECTED_TILES_PER_BATCH: tl.constexpr,
    CHANNELS_: tl.constexpr,
    HW_: tl.constexpr,
    KEEP_PROB_: tl.constexpr,
    MAX_OCC: tl.constexpr,
    BLOCK_SELECTED: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    batch = tl.program_id(0)
    tile_selected = tl.program_id(1)
    selected = tile_selected * BLOCK_SELECTED + tl.arange(0, BLOCK_SELECTED)
    channels = tl.arange(0, BLOCK_CHANNELS)
    selected_mask = selected < NUM_SELECTED
    channel_mask = channels < CHANNELS_
    mask = selected_mask[:, None] & channel_mask[None, :]

    src_local = tl.load(
        selected_src_locs_ptr + selected,
        mask=selected_mask,
        other=0,
    )
    rows = batch * HW_ + src_local
    source_offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(x_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(rhs_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + source_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    scale = tl.load(scale_ptr + rows, mask=selected_mask, other=0.0).to(tl.float32)
    row_sum = tl.load(row_sum_ptr + rows, mask=selected_mask, other=0.0).to(tl.float32)
    row_dot = tl.load(row_dot_ptr + rows, mask=selected_mask, other=0.0).to(tl.float32)
    keep = tl.load(drop_mask_ptr + batch).to(tl.float32)
    keep_scale = keep / KEEP_PROB_

    weighted = x * weight[None, :]
    grad = scale[:, None] * (
        weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    dropped = (residual + grad) * keep_scale[:, None]

    start = tl.load(starts_ptr + src_local, mask=selected_mask, other=0)
    end = tl.load(starts_ptr + src_local + 1, mask=selected_mask, other=0)
    count = end - start

    for occ in tl.static_range(0, MAX_OCC):
        occ_mask = selected_mask & (occ < count)
        out_local = tl.load(out_locs_ptr + start + occ, mask=occ_mask, other=0)
        out_rows = batch * HW_ + out_local
        out_offsets = out_rows[:, None] * CHANNELS_ + channels[None, :]
        tl.store(
            out_transposed_ptr + out_offsets,
            dropped,
            mask=occ_mask[:, None] & channel_mask[None, :],
        )

    sum_dropped = tl.sum(
        tl.where(mask, dropped * count[:, None].to(tl.float32), 0.0),
        axis=0,
    )
    partial_offsets = (
        (batch * SELECTED_TILES_PER_BATCH + tile_selected) * CHANNELS_ + channels
    )
    tl.store(partial_dropped_ptr + partial_offsets, sum_dropped, mask=channel_mask)


@triton.jit
def _finalize_column_sums_kernel(
    partial_x_rhs_ptr,
    partial_x_ptr,
    partial_dropped_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_dropped_ptr,
    NUM_ROW_TILES: tl.constexpr,
    NUM_DROPPED_TILES: tl.constexpr,
    CHANNELS_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_rhs = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_dropped = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_rhs += tl.sum(
            tl.load(partial_x_rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    for tile_start in range(0, NUM_DROPPED_TILES, BLOCK_TILES):
        tiles = tile_start + tile_offsets
        mask = (tiles[:, None] < NUM_DROPPED_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_dropped += tl.sum(
            tl.load(partial_dropped_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )

    tl.store(out_x_rhs_ptr + channels, acc_x_rhs, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_dropped_ptr + channels, acc_dropped, mask=channel_mask)


def oracle_triton_prepared(
    x_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    drop_mask_b: torch.Tensor,
    starts: torch.Tensor,
    out_locs: torch.Tensor,
    selected_src_locs: torch.Tensor,
    max_occ: int,
    selected_tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_mc.shape == (ROWS, CHANNELS)
    assert weight_c.shape == (CHANNELS,)
    assert rhs_mc.shape == (ROWS, CHANNELS)
    assert scale_m.shape == (ROWS,)
    assert residual_mc.shape == (ROWS, CHANNELS)
    assert drop_mask_b.shape == (BATCH,)
    assert starts.shape == (HW + 1,)
    assert out_locs.shape == (HW,)
    assert x_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert drop_mask_b.is_contiguous()
    assert starts.is_contiguous()
    assert out_locs.is_contiguous()
    assert selected_src_locs.is_contiguous()

    device = x_mc.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    row_sum = torch.empty((ROWS,), device=device, dtype=torch.float32)
    row_dot = torch.empty((ROWS,), device=device, dtype=torch.float32)
    partial_x_rhs = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_x = torch.empty((num_row_tiles, CHANNELS), device=device, dtype=torch.float32)

    _row_reduction_partials_kernel[(num_row_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        row_sum,
        row_dot,
        partial_x_rhs,
        partial_x,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    num_selected = selected_src_locs.numel()
    selected_tiles_per_batch = triton.cdiv(num_selected, selected_tile)
    num_dropped_tiles = BATCH * selected_tiles_per_batch
    partial_dropped = torch.empty(
        (num_dropped_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out_transposed = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.float32,
    )

    _selected_window_output_kernel[(BATCH, selected_tiles_per_batch)](
        x_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        starts,
        out_locs,
        selected_src_locs,
        row_sum,
        row_dot,
        partial_dropped,
        out_transposed,
        NUM_SELECTED=num_selected,
        SELECTED_TILES_PER_BATCH=selected_tiles_per_batch,
        CHANNELS_=CHANNELS,
        HW_=HW,
        KEEP_PROB_=KEEP_PROB,
        MAX_OCC=max_occ,
        BLOCK_SELECTED=selected_tile,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    out_x_rhs = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_dropped = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_column_sums_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_x_rhs,
        partial_x,
        partial_dropped,
        out_x_rhs,
        out_x,
        out_dropped,
        NUM_ROW_TILES=num_row_tiles,
        NUM_DROPPED_TILES=num_dropped_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_transposed, out_dropped


def oracle_full(
    *inputs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


@oracle_impl(hardware="H100", shapes="(T([25088, 512], f32), T([512], f32), T([128, 196, 512], f32), T([128, 196, 1], f32), T([128, 196, 512], f32), T([128, 1, 1, 1], b8), T([14], i64, gen=Index(14)), S([128, 196, 512]), S([128, 14, 14, 512]), S([128, 2, 7, 2, 7, 512]), S([512, 7, 7, 512]), S([512, 49, 512]), S([25088, 512]), S([512]))")
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
