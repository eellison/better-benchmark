"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Swin window-reverse layer-norm-backward/drop-path return tuple by row-tiling the `[25088, 512]` producer, reconstructing the fused window layout, writing the returned non-contiguous `[512, 25088]` transposed dropped-gradient side output, and cooperatively accumulating the two pre-drop `[512]` column reductions plus the post-drop `[512]` column reduction from the same tiles, whereas Inductor currently schedules the reshape/view/permute/clone layout change, hidden-dimension row reductions, drop-path epilogue, transposed side-output store, and sibling `sum([0, 1, 2])`/`sum([0])` reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps a layout-changing producer, row-local reduction scalars, a dependent full-tensor side store, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the window-layout remap and drop-path transpose store, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl

import repro as repro_module


SHAPE_LABEL = "timm_swin_base_patch4_window7_224_train_001_34e07cb7"

BATCH = 128
HEIGHT = 14
WIDTH = 14
WINDOW = 7
WINDOW_BLOCKS_W = 2
HW = HEIGHT * WIDTH
ROWS = BATCH * HW
CHANNELS = 512
KEEP_PROB = 0.9782608672976494

TILE_ROWS = 8
TILE_CHANNELS = 512
KERNEL_WARPS = 4
KERNEL_STAGES = 4


@triton.jit
def _row_tile_store_and_atomic_reduce_kernel(
    mm_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    residual_ptr,
    drop_mask_ptr,
    out_x_rhs_ptr,
    out_x_ptr,
    out_dropped_ptr,
    out_transposed_ptr,
    ROWS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    HW_: tl.constexpr,
    WIDTH_: tl.constexpr,
    WINDOW_: tl.constexpr,
    WINDOW_BLOCKS_W_: tl.constexpr,
    KEEP_PROB_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)

    spatial = rows % HW_
    h = spatial // WIDTH_
    w = spatial - h * WIDTH_
    block_h = h // WINDOW_
    inner_h = h - block_h * WINDOW_
    block_w = w // WINDOW_
    inner_w = w - block_w * WINDOW_
    source_rows = (
        (rows // HW_) * HW_
        + block_h * (WINDOW_BLOCKS_W_ * WINDOW_ * WINDOW_)
        + block_w * (WINDOW_ * WINDOW_)
        + inner_h * WINDOW_
        + inner_w
    )

    source_offsets = source_rows[:, None] * CHANNELS_ + channels[None, :]
    image_offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(mm_ptr + source_offsets).to(tl.float32)
    rhs = tl.load(rhs_ptr + image_offsets).to(tl.float32)
    residual = tl.load(residual_ptr + image_offsets).to(tl.float32)
    weight = tl.load(weight_ptr + channels).to(tl.float32)
    scale = tl.load(scale_ptr + rows).to(tl.float32)
    keep_scale = tl.load(drop_mask_ptr + rows // HW_).to(tl.float32) / KEEP_PROB_

    weighted = x * weight[None, :]
    row_sum = tl.sum(weighted, axis=1)
    row_dot = tl.sum(weighted * rhs, axis=1)
    grad = scale[:, None] * (
        weighted * CHANNELS_ - row_sum[:, None] - rhs * row_dot[:, None]
    )
    dropped = (residual + grad) * keep_scale[:, None]

    tl.store(out_transposed_ptr + image_offsets, dropped)

    tl.atomic_add(out_x_rhs_ptr + channels, tl.sum(x * rhs, axis=0), sem="relaxed")
    tl.atomic_add(out_x_ptr + channels, tl.sum(x, axis=0), sem="relaxed")
    tl.atomic_add(out_dropped_ptr + channels, tl.sum(dropped, axis=0), sem="relaxed")


def _as_tuple(out: object) -> tuple[torch.Tensor, ...]:
    if isinstance(out, tuple):
        return out
    return (out,)


def make_inputs() -> tuple[object, ...]:
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in repro_module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    model = repro_module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_146,
        arg50_1,
        arg253_1,
        arg529_1,
        view_575,
        arg252_1,
        *_shape_params,
    ) = inputs

    return (
        mm_146.reshape(ROWS, CHANNELS).contiguous(),
        arg50_1.contiguous(),
        arg253_1.reshape(ROWS, CHANNELS).contiguous(),
        arg529_1.reshape(ROWS).contiguous(),
        view_575.reshape(ROWS, CHANNELS).contiguous(),
        arg252_1.reshape(BATCH).contiguous(),
    )


def oracle_triton_prepared(
    mm_mc: torch.Tensor,
    weight_c: torch.Tensor,
    rhs_mc: torch.Tensor,
    scale_m: torch.Tensor,
    residual_mc: torch.Tensor,
    drop_mask_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if mm_mc.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert ROWS % TILE_ROWS == 0
    assert CHANNELS == TILE_CHANNELS
    assert mm_mc.shape == (ROWS, CHANNELS)
    assert weight_c.shape == (CHANNELS,)
    assert rhs_mc.shape == (ROWS, CHANNELS)
    assert scale_m.shape == (ROWS,)
    assert residual_mc.shape == (ROWS, CHANNELS)
    assert drop_mask_b.shape == (BATCH,)
    assert mm_mc.is_contiguous()
    assert weight_c.is_contiguous()
    assert rhs_mc.is_contiguous()
    assert scale_m.is_contiguous()
    assert residual_mc.is_contiguous()
    assert drop_mask_b.is_contiguous()

    device = mm_mc.device
    out_transposed = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out_sums = torch.zeros((3, CHANNELS), device=device, dtype=torch.float32)
    out_x_rhs = out_sums[0]
    out_x = out_sums[1]
    out_dropped = out_sums[2]

    _row_tile_store_and_atomic_reduce_kernel[(ROWS // TILE_ROWS,)](
        mm_mc,
        weight_c,
        rhs_mc,
        scale_m,
        residual_mc,
        drop_mask_b,
        out_x_rhs,
        out_x,
        out_dropped,
        out_transposed,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        HW_=HW,
        WIDTH_=WIDTH,
        WINDOW_=WINDOW,
        WINDOW_BLOCKS_W_=WINDOW_BLOCKS_W,
        KEEP_PROB_=KEEP_PROB,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=KERNEL_WARPS,
        num_stages=KERNEL_STAGES,
    )

    return out_x_rhs, out_x, out_transposed, out_dropped


def oracle_full(
    *inputs: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1.0e-8)
    return diff.max().item(), rel.max().item()


def run_check(rtol: float, atol: float) -> bool:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle check")

    torch.manual_seed(0)
    inputs = make_inputs()
    with torch.no_grad():
        expected = reference_outputs(inputs)
        actual = _as_tuple(oracle_full(*inputs))
        torch.cuda.synchronize()

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


def run_bench(warmup: int, rep: int) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle benchmark")

    torch.manual_seed(0)
    inputs = make_inputs()
    oracle_inputs = prepare_oracle_inputs(*inputs)
    with torch.no_grad():
        oracle_triton_prepared(*oracle_inputs)
        torch.cuda.synchronize()
        oracle_ms = triton.testing.do_bench(
            lambda: oracle_triton_prepared(*oracle_inputs),
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )

    logical_read_bytes = (
        ROWS * CHANNELS * 4
        + CHANNELS * 4
        + ROWS * CHANNELS * 4
        + ROWS * 4
        + ROWS * CHANNELS * 4
        + BATCH
    )
    logical_write_bytes = ROWS * CHANNELS * 4 + 3 * CHANNELS * 4
    atomic_updates = (ROWS // TILE_ROWS) * CHANNELS * 3
    print(
        f"oracle_full cooperative split-k Swin window LN/drop-path tuple: "
        f"{oracle_ms * 1000.0:.3f} us shape={SHAPE_LABEL}"
    )
    print(
        f"logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB "
        f"atomic_updates: {atomic_updates}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=3.0e-3)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        raise SystemExit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
