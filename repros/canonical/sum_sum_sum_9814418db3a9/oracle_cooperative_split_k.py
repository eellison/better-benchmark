"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full Swin layer-norm-backward/drop-path/window-reverse return tuple from Repro.forward by reducing every `[25088, 512]` source row once, cooperatively accumulating the two original `[512]` column reductions, building the tiny inverse `fmod` window map to avoid replaying duplicated advanced-index gathers, writing the returned non-contiguous `[512, 25088]` transposed side output in window-reverse layout, and finalizing the post-window `[512]` column reduction with source-row multiplicities, whereas Inductor currently schedules the row reductions, stochastic-depth scale, two advanced-index gathers, window layout change, transposed side-output store, and sibling column sums as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local layer-norm-backward reductions, arbitrary gather/window layout stores, and multiple compatible column accumulators in one producer/finalizer plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the shifted-window gather/window-reverse transpose store through an inverse-map producer, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_9814418db3a9"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_d4777b22"

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

TILE_ROWS = 16
TILE_SELECTED = 4
TILE_CHANNELS = 512
FINAL_BLOCK_CHANNELS = 2
FINAL_BLOCK_TILES = 2048


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


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    moved: list[object] = []
    for value in module.make_inputs():
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(
    inputs: tuple[object, ...],
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_torch(
    mm_153: torch.Tensor,
    primals_93: torch.Tensor,
    mul_58: torch.Tensor,
    div_109: torch.Tensor,
    view_1219: torch.Tensor,
    lt_8: torch.Tensor,
    fmod_8: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    _shape_param_3,
    _shape_param_4,
    _shape_param_5,
    _shape_param_6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    )

    x = mm_153.reshape(BATCH, HW, CHANNELS)
    rhs = mul_58.reshape(BATCH, HW, CHANNELS)
    weighted = x * primals_93
    row_sum = weighted.sum(dim=2, keepdim=True)
    row_dot = (weighted * rhs).sum(dim=2, keepdim=True)
    grad_delta = div_109 * (
        weighted * CHANNELS - row_sum - rhs * row_dot
    )

    out_x_rhs = (x * rhs).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))

    dropped = (view_1219 + grad_delta).reshape(BATCH, HEIGHT, WIDTH, CHANNELS)
    dropped = dropped * (lt_8.to(torch.float32) / KEEP_PROB)
    indexed = torch.ops.aten.index.Tensor(dropped, [None, None, fmod_8])
    indexed = torch.ops.aten.index.Tensor(indexed, [None, fmod_8])
    windows = indexed.reshape(BATCH, 2, WINDOW, 2, WINDOW, CHANNELS)
    flat = (
        windows.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(BATCH * WINDOWS_PER_IMAGE, WINDOW_AREA, CHANNELS)
        .reshape(ROWS, CHANNELS)
    )
    out_transposed = flat.permute(1, 0)
    out_final_sum = flat.sum(dim=0)
    return out_x_rhs, out_x, out_transposed, out_final_sum


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


def prepare_oracle_inputs(*inputs: object) -> tuple[object, ...]:
    (
        mm_153,
        primals_93,
        mul_58,
        div_109,
        view_1219,
        lt_8,
        fmod_8,
        *_shape_params,
    ) = inputs

    if not isinstance(fmod_8, torch.Tensor):
        raise TypeError("fmod_8 must be a tensor")

    if triton is None or mm_153.device.type != "cuda":
        return inputs

    starts, out_locs, selected_src_locs, max_occ, selected_tile = _build_window_maps(
        fmod_8
    )
    return (
        mm_153.reshape(ROWS, CHANNELS).contiguous(),
        primals_93.contiguous(),
        mul_58.reshape(ROWS, CHANNELS).contiguous(),
        div_109.reshape(ROWS).contiguous(),
        view_1219.reshape(ROWS, CHANNELS).contiguous(),
        lt_8.reshape(BATCH).contiguous(),
        starts,
        out_locs,
        selected_src_locs,
        max_occ,
        selected_tile,
    )


if triton is not None:

    @triton.jit
    def _source_row_store_and_reduce_kernel(
        x_ptr,
        weight_ptr,
        rhs_ptr,
        scale_ptr,
        residual_ptr,
        drop_mask_ptr,
        starts_ptr,
        out_locs_ptr,
        partial_x_rhs_ptr,
        partial_x_ptr,
        partial_dropped_ptr,
        out_transposed_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        HW_: tl.constexpr,
        KEEP_PROB_: tl.constexpr,
        MAX_OCC: tl.constexpr,
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

        batch = rows // HW_
        src_local = rows - batch * HW_
        start = tl.load(starts_ptr + src_local, mask=row_mask, other=0)
        end = tl.load(starts_ptr + src_local + 1, mask=row_mask, other=0)
        count = end - start
        selected_mask = row_mask & (count > 0)
        selected_element_mask = selected_mask[:, None] & channel_mask[None, :]

        residual = tl.load(
            residual_ptr + source_offsets,
            mask=selected_element_mask,
            other=0.0,
        ).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=selected_mask, other=0.0).to(
            tl.float32
        )
        keep = tl.load(drop_mask_ptr + batch, mask=selected_mask, other=0).to(
            tl.float32
        )
        keep_scale = keep / KEEP_PROB_
        grad = scale[:, None] * (
            weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
        )
        dropped = (residual + grad) * keep_scale[:, None]

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

        partial_offsets = tile_row * CHANNELS_ + channels
        sum_x_rhs = tl.sum(tl.where(mask, x * rhs, 0.0), axis=0)
        sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
        sum_dropped = tl.sum(
            tl.where(
                selected_element_mask,
                dropped * count[:, None].to(tl.float32),
                0.0,
            ),
            axis=0,
        )
        tl.store(partial_x_rhs_ptr + partial_offsets, sum_x_rhs, mask=channel_mask)
        tl.store(partial_x_ptr + partial_offsets, sum_x, mask=channel_mask)
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
                tl.load(
                    partial_x_rhs_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32),
                axis=0,
            )
            acc_x += tl.sum(
                tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )

        for tile_start in range(0, NUM_DROPPED_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
            mask = (tiles[:, None] < NUM_DROPPED_TILES) & channel_mask[None, :]
            offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
            acc_dropped += tl.sum(
                tl.load(
                    partial_dropped_ptr + offsets,
                    mask=mask,
                    other=0.0,
                ).to(tl.float32),
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
    if triton is None:
        raise RuntimeError("triton is not available")
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
    partial_x_rhs = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_x = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    partial_dropped = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out_transposed = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.float32,
    )

    _source_row_store_and_reduce_kernel[(num_row_tiles,)](
        x_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        starts,
        out_locs,
        partial_x_rhs,
        partial_x,
        partial_dropped,
        out_transposed,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        HW_=HW,
        KEEP_PROB_=KEEP_PROB,
        MAX_OCC=max_occ,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=4,
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
        NUM_DROPPED_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_rhs, out_x, out_transposed, out_dropped


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "torch":
        return oracle_torch(*inputs)
    if impl == "triton":
        prepared = prepare_oracle_inputs(*inputs)
        if len(prepared) != 11:
            raise RuntimeError("prepared Triton inputs were not created")
        return oracle_triton_prepared(*prepared)
    raise ValueError(f"unknown impl: {impl}")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, impl: str, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = _as_tuple(oracle_full(*inputs, impl=impl))
        synchronize(device)

    ok = len(actual) == len(expected)
    if not ok:
        print(f"output_count: actual={len(actual)} expected={len(expected)} allclose=False")

    for idx, (got, ref) in enumerate(zip(actual, expected)):
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        output_ok = shape_ok and dtype_ok and stride_ok and value_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn, device: torch.device, warmup: int, rep: int) -> float:
    if device.type == "cuda" and triton is not None:
        return triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        ) * 1000.0

    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def run_bench(device: torch.device, impl: str, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    actual_impl = impl
    if actual_impl == "auto":
        actual_impl = "triton" if device.type == "cuda" and triton is not None else "torch"

    with torch.no_grad():
        if actual_impl == "triton":
            prepared = prepare_oracle_inputs(*inputs)
            oracle_triton_prepared(*prepared)
            synchronize(device)
            oracle_us = benchmark(
                lambda: oracle_triton_prepared(*prepared),
                device,
                warmup,
                rep,
            )
            num_selected = prepared[8].numel()
            num_partial_tiles = (ROWS + TILE_ROWS - 1) // TILE_ROWS
        else:
            oracle_full(*inputs, impl=actual_impl)
            synchronize(device)
            oracle_us = benchmark(
                lambda: oracle_full(*inputs, impl=actual_impl),
                device,
                warmup,
                rep,
            )
            num_selected = HW
            num_partial_tiles = (ROWS + TILE_ROWS - 1) // TILE_ROWS

    source_fraction = num_selected / HW
    partial_bytes = num_partial_tiles * CHANNELS * 4 * 3
    logical_read_bytes = (
        (ROWS * CHANNELS * 4) * 2
        + CHANNELS * 4
        + int(ROWS * CHANNELS * 4 * source_fraction)
        + int(ROWS * 4 * source_fraction)
        + BATCH
        + HW * 8 * 2
        + partial_bytes
    )
    logical_write_bytes = (
        partial_bytes
        + ROWS * CHANNELS * 4
        + CHANNELS * 4 * 3
    )

    print(
        f"oracle_full cooperative split-k Swin LN/window tuple: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )
    print(
        f"selected_source_rows={num_selected}/{HW} "
        f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(
        device=device,
        impl=args.impl,
        rtol=args.rtol,
        atol=args.atol,
    ):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, impl=args.impl, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
