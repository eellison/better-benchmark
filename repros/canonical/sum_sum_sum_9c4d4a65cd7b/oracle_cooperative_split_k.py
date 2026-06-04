"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DeiT layer-norm-backward and positional-embedding return tuple by row-tiling the `[128, 198, 768]` producer, computing each row's hidden-dimension reductions once, accumulating the two global channel reductions, the full `[1, 198, 768]` token reduction, the class/dist-token reductions, and the patch-token reduction without materializing the intermediate gradient tensor, whereas Inductor currently schedules the reshape, row reductions, dependent layer-norm-backward pointwise expression, residual add, slice/permute/reshape view path, and sibling reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local scalar reductions with multiple compatible batch/token/channel reductions and view-equivalent epilogues in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: add an Inductor split-row multi-output reduction template for layer-norm-backward tails that shares row scalars, emits partial accumulators for token/global channel reductions, and finalizes the full return tuple without materializing the intermediate gradient tensor."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


REPRO_ID = "sum_sum_sum_9c4d4a65cd7b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_deit_base_distilled_patch16_224_train_5ae49b96"

BATCH = 128
TOKENS = 198
PATCH_START = 2
CHANNELS = 768
ROWS = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS

TILE_ROWS = 16
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 1024
FINAL_BLOCK_TOKENS = 256

sys.path.insert(0, str(REPO_ROOT))


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
        mm_98,
        primals_7,
        cat,
        primals_4,
        getitem_1,
        rsqrt,
        add_135,
        *_shape_params,
    ) = inputs

    return (
        mm_98.reshape(ROWS, CHANNELS).contiguous(),
        primals_7.contiguous(),
        cat.reshape(ROWS, CHANNELS).contiguous(),
        primals_4.reshape(TOKENS, CHANNELS).contiguous(),
        getitem_1.reshape(ROWS).contiguous(),
        rsqrt.reshape(ROWS).contiguous(),
        add_135.reshape(ROWS, CHANNELS).contiguous(),
    )


@triton.jit
def _row_tile_partial_and_token_kernel(
    x_ptr,
    weight_ptr,
    cat_ptr,
    pos_ptr,
    mean_ptr,
    rsqrt_ptr,
    residual_ptr,
    partial_x_norm_ptr,
    partial_x_ptr,
    token_sum_ptr,
    ROWS_: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    INV_CHANNELS_: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tile_row = tl.program_id(0)
    rows = tile_row * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    channels = tl.arange(0, BLOCK_CHANNELS)
    row_mask = rows < ROWS_
    channel_mask = channels < CHANNELS_
    mask = row_mask[:, None] & channel_mask[None, :]
    offsets = rows[:, None] * CHANNELS_ + channels[None, :]

    token = rows % TOKENS_
    pos_offsets = token[:, None] * CHANNELS_ + channels[None, :]

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    cat = tl.load(cat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pos = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

    norm = (cat + pos - mean[:, None]) * rsqrt[:, None]
    weighted = x * weight[None, :]
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
    row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
    ln_grad = rsqrt[:, None] * INV_CHANNELS_ * (
        weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None]
    )
    add_value = residual + ln_grad

    token_offsets = token[:, None] * CHANNELS_ + channels[None, :]
    tl.atomic_add(token_sum_ptr + token_offsets, add_value, sem="relaxed", mask=mask)

    partial_offsets = tile_row * CHANNELS_ + channels
    tl.store(
        partial_x_norm_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * norm, 0.0), axis=0),
        mask=channel_mask,
    )
    tl.store(
        partial_x_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=0),
        mask=channel_mask,
    )


@triton.jit
def _finalize_outputs_kernel(
    partial_x_norm_ptr,
    partial_x_ptr,
    token_sum_ptr,
    out_x_norm_ptr,
    out_x_ptr,
    out_dist_ptr,
    out_cls_ptr,
    out_patch_ptr,
    NUM_ROW_TILES: tl.constexpr,
    TOKENS_: tl.constexpr,
    CHANNELS_: tl.constexpr,
    PATCH_START_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < CHANNELS_
    tile_offsets = tl.arange(0, BLOCK_TILES)

    acc_x_norm = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
    for tile_base in range(0, NUM_ROW_TILES, BLOCK_TILES):
        tiles = tile_base + tile_offsets
        mask = (tiles[:, None] < NUM_ROW_TILES) & channel_mask[None, :]
        offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
        acc_x_norm += tl.sum(
            tl.load(partial_x_norm_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        acc_x += tl.sum(
            tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
            axis=0,
        )

    token_offsets = channels
    cls = tl.load(token_sum_ptr + token_offsets, mask=channel_mask, other=0.0).to(
        tl.float32
    )
    dist = tl.load(
        token_sum_ptr + CHANNELS_ + token_offsets,
        mask=channel_mask,
        other=0.0,
    ).to(tl.float32)

    patch_tokens = PATCH_START_ + tl.arange(0, BLOCK_TOKENS)
    patch_mask = (patch_tokens[:, None] < TOKENS_) & channel_mask[None, :]
    patch_offsets = patch_tokens[:, None] * CHANNELS_ + channels[None, :]
    patch_values = tl.load(token_sum_ptr + patch_offsets, mask=patch_mask, other=0.0).to(
        tl.float32
    )
    patch = tl.sum(patch_values, axis=0)

    tl.store(out_x_norm_ptr + channels, acc_x_norm, mask=channel_mask)
    tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
    tl.store(out_dist_ptr + channels, dist, mask=channel_mask)
    tl.store(out_cls_ptr + channels, cls, mask=channel_mask)
    tl.store(out_patch_ptr + channels, patch, mask=channel_mask)


def oracle_triton_prepared(
    x_md: torch.Tensor,
    weight_d: torch.Tensor,
    cat_md: torch.Tensor,
    pos_td: torch.Tensor,
    mean_m: torch.Tensor,
    rsqrt_m: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (ROWS, CHANNELS)
    assert weight_d.shape == (CHANNELS,)
    assert cat_md.shape == (ROWS, CHANNELS)
    assert pos_td.shape == (TOKENS, CHANNELS)
    assert mean_m.shape == (ROWS,)
    assert rsqrt_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert weight_d.is_contiguous()
    assert cat_md.is_contiguous()
    assert pos_td.is_contiguous()
    assert mean_m.is_contiguous()
    assert rsqrt_m.is_contiguous()
    assert residual_md.is_contiguous()

    device = x_md.device
    num_row_tiles = triton.cdiv(ROWS, TILE_ROWS)
    partials = torch.empty(
        (2, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    token_sum = torch.empty((1, TOKENS, CHANNELS), device=device, dtype=torch.float32)
    token_sum.zero_()

    _row_tile_partial_and_token_kernel[(num_row_tiles,)](
        x_md,
        weight_d,
        cat_md,
        pos_td,
        mean_m,
        rsqrt_m,
        residual_md,
        partials[0],
        partials[1],
        token_sum,
        ROWS_=ROWS,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_CHANNELS=TILE_CHANNELS,
        num_warps=8,
    )

    out_x_norm = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_x = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    out_dist = torch.empty((1, 1, CHANNELS), device=device, dtype=torch.float32)
    out_cls = torch.empty((1, 1, CHANNELS), device=device, dtype=torch.float32)
    out_patch = torch.empty((CHANNELS,), device=device, dtype=torch.float32)

    _finalize_outputs_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partials[0],
        partials[1],
        token_sum,
        out_x_norm,
        out_x,
        out_dist,
        out_cls,
        out_patch,
        NUM_ROW_TILES=num_row_tiles,
        TOKENS_=TOKENS,
        CHANNELS_=CHANNELS,
        PATCH_START_=PATCH_START,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_TOKENS=FINAL_BLOCK_TOKENS,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_norm, out_x, token_sum, out_dist, out_cls, out_patch


def oracle_full(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        offset_ok = got.storage_offset() == ref.storage_offset()
        if shape_ok:
            max_abs, max_rel = _max_diff(got, ref)
            value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        else:
            max_abs, max_rel = math.inf, math.inf
            value_ok = False
        output_ok = shape_ok and value_ok and dtype_ok and stride_ok and offset_ok
        ok = ok and output_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} expected_shape={list(ref.shape)} "
            f"dtype={got.dtype} expected_dtype={ref.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"storage_offset={got.storage_offset()} expected_offset={ref.storage_offset()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} allclose={value_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} offset_match={offset_ok}"
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

    logical_read_bytes = (ROWS * CHANNELS * 4) * 5 + ROWS * 4 * 2 + TOKENS * CHANNELS * 4
    logical_write_bytes = (
        TOKENS * CHANNELS * 4
        + (2 * triton.cdiv(ROWS, TILE_ROWS) * CHANNELS * 4)
        + (CHANNELS * 4 * 5)
    )
    print(
        f"oracle_full cooperative split-k DeiT LN/token reductions: "
        f"{oracle_ms * 1000.0:.3f} us shape={SHAPE_LABEL}"
    )
    print(f"logical traffic including partials: {(logical_read_bytes + logical_write_bytes) / 1.0e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the full Repro.forward tuple")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    if args.check and not run_check(rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    with torch.no_grad():
        main()
