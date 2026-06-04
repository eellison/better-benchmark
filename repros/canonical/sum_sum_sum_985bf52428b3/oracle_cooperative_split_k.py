"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ViT patch-embed/position-add/layer-norm-backward return tuple by row-tiling the `[128, 256, 768]` producer, sharing each row's hidden-dimension reductions while accumulating the two upstream channel sums, the `[1, 256, 768]` token side sum, and the final patch-layout channel sum, whereas Inductor currently schedules the convolution reshape/permute, row reductions, dependent layer-norm-backward pointwise expression, residual add, token reduction, and sibling channel reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local scalar reductions with multiple compatible batch/token/channel reductions and view-equivalent patch epilogues in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward tails across row tiles, share row scalars, emit partial accumulators for all channel reductions, and finalize the full tuple without materializing the intermediate gradient tensor."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps CPU-only syntax checks usable.
    triton = None
    tl = None


REPRO_ID = "sum_sum_sum_985bf52428b3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_vit_base_patch16_siglip_256_train_0a8651c2"

BATCH = 128
TOKENS = 256
CHANNELS = 768
PATCH_H = 16
PATCH_W = 16
ROWS = BATCH * TOKENS
INV_CHANNELS = 1.0 / CHANNELS

TILE_ROWS = 8
TILE_CHANNELS = 1024
FINAL_BLOCK_CHANNELS = 16
FINAL_BLOCK_TILES = 256

CONV_STRIDE = (CHANNELS * TOKENS, 1, PATCH_W * CHANNELS, CHANNELS)

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
    mm_105: torch.Tensor,
    primals_5: torch.Tensor,
    convolution: torch.Tensor,
    primals_4: torch.Tensor,
    getitem_1: torch.Tensor,
    rsqrt: torch.Tensor,
    add_141: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = mm_105.reshape(_shape_param_0)
    weighted = x * primals_5
    row_sum = weighted.sum(dim=2, keepdim=True)

    patch_tokens = convolution.reshape(_shape_param_1).permute(0, 2, 1)
    norm = (patch_tokens + primals_4 - getitem_1) * rsqrt
    row_dot = (weighted * norm).sum(dim=2, keepdim=True)
    grad = (rsqrt * INV_CHANNELS) * (
        weighted * CHANNELS - row_sum - norm * row_dot
    )
    add_value = add_141 + grad

    out_x_norm = (x * norm).sum(dim=(0, 1))
    out_x = x.sum(dim=(0, 1))
    out_token = add_value.sum(dim=0, keepdim=True)
    out_patch = (
        add_value.permute(0, 2, 1)
        .reshape(_shape_param_2)
        .sum(dim=(0, 2, 3))
    )
    return out_x_norm, out_x, out_token, out_patch


if triton is not None:

    @triton.jit
    def _row_tile_partial_and_token_kernel(
        x_ptr,
        weight_ptr,
        conv_ptr,
        pos_ptr,
        mean_ptr,
        rsqrt_ptr,
        residual_ptr,
        partial_x_norm_ptr,
        partial_x_ptr,
        partial_add_ptr,
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
        tokens = rows % TOKENS_
        token_offsets = tokens[:, None] * CHANNELS_ + channels[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        patch = tl.load(conv_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        pos = tl.load(pos_ptr + token_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )
        mean = tl.load(mean_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        inv = tl.load(rsqrt_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)

        norm = (patch + pos - mean[:, None]) * inv[:, None]
        weighted = x * weight[None, :]
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, weighted * norm, 0.0), axis=1)
        grad = inv[:, None] * INV_CHANNELS_ * (
            weighted * CHANNELS_ - row_sum[:, None] - norm * row_dot[:, None]
        )
        add_value = residual + grad

        tl.atomic_add(
            token_sum_ptr + token_offsets,
            add_value,
            sem="relaxed",
            mask=mask,
        )

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
        tl.store(
            partial_add_ptr + partial_offsets,
            tl.sum(tl.where(mask, add_value, 0.0), axis=0),
            mask=channel_mask,
        )


    @triton.jit
    def _finalize_channel_sums_kernel(
        partial_x_norm_ptr,
        partial_x_ptr,
        partial_add_ptr,
        out_x_norm_ptr,
        out_x_ptr,
        out_patch_ptr,
        NUM_ROW_TILES: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_CHANNELS: tl.constexpr,
    ):
        channels = tl.program_id(0) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
        channel_mask = channels < CHANNELS_
        tile_offsets = tl.arange(0, BLOCK_TILES)

        acc_x_norm = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_x = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)
        acc_add = tl.zeros((BLOCK_CHANNELS,), dtype=tl.float32)

        for tile_start in range(0, NUM_ROW_TILES, BLOCK_TILES):
            tiles = tile_start + tile_offsets
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
            acc_add += tl.sum(
                tl.load(partial_add_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                ),
                axis=0,
            )

        tl.store(out_x_norm_ptr + channels, acc_x_norm, mask=channel_mask)
        tl.store(out_x_ptr + channels, acc_x, mask=channel_mask)
        tl.store(out_patch_ptr + channels, acc_add, mask=channel_mask)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def prepare_oracle_inputs(*inputs: object) -> tuple[torch.Tensor, ...]:
    (
        mm_105,
        primals_5,
        convolution,
        primals_4,
        getitem_1,
        rsqrt,
        add_141,
        *_shape_params,
    ) = inputs

    return (
        mm_105.reshape(ROWS, CHANNELS).contiguous(),
        primals_5.contiguous(),
        convolution,
        primals_4.reshape(TOKENS, CHANNELS).contiguous(),
        getitem_1.reshape(ROWS).contiguous(),
        rsqrt.reshape(ROWS).contiguous(),
        add_141.reshape(ROWS, CHANNELS).contiguous(),
    )


def oracle_triton_prepared(
    x_md: torch.Tensor,
    weight_d: torch.Tensor,
    conv_nchw: torch.Tensor,
    pos_td: torch.Tensor,
    mean_m: torch.Tensor,
    rsqrt_m: torch.Tensor,
    residual_md: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not available")
    if x_md.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    assert x_md.shape == (ROWS, CHANNELS)
    assert weight_d.shape == (CHANNELS,)
    assert conv_nchw.shape == (BATCH, CHANNELS, PATCH_H, PATCH_W)
    assert conv_nchw.stride() == CONV_STRIDE
    assert pos_td.shape == (TOKENS, CHANNELS)
    assert mean_m.shape == (ROWS,)
    assert rsqrt_m.shape == (ROWS,)
    assert residual_md.shape == (ROWS, CHANNELS)
    assert x_md.is_contiguous()
    assert weight_d.is_contiguous()
    assert pos_td.is_contiguous()
    assert mean_m.is_contiguous()
    assert rsqrt_m.is_contiguous()
    assert residual_md.is_contiguous()

    device = x_md.device
    num_row_tiles = _cdiv(ROWS, TILE_ROWS)
    partials = torch.empty(
        (3, num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    token_sum = torch.empty(
        (1, TOKENS, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    token_sum.zero_()

    _row_tile_partial_and_token_kernel[(num_row_tiles,)](
        x_md,
        weight_d,
        conv_nchw,
        pos_td,
        mean_m,
        rsqrt_m,
        residual_md,
        partials[0],
        partials[1],
        partials[2],
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
    out_patch = torch.empty((CHANNELS,), device=device, dtype=torch.float32)
    _finalize_channel_sums_kernel[(_cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partials[0],
        partials[1],
        partials[2],
        out_x_norm,
        out_x,
        out_patch,
        NUM_ROW_TILES=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_x_norm, out_x, token_sum, out_patch


def oracle_triton(*inputs: object) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return oracle_triton_prepared(*prepare_oracle_inputs(*inputs))


def oracle_full(
    *inputs: object,
    impl: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first_tensor = next(value for value in inputs if isinstance(value, torch.Tensor))
    if impl == "auto":
        impl = "triton" if first_tensor.device.type == "cuda" and triton is not None else "torch"
    if impl == "triton":
        return oracle_triton(*inputs)
    if impl == "torch":
        return oracle_torch(*inputs)
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


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
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
        oracle_full(*inputs, impl=actual_impl)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_full(*inputs, impl=actual_impl),
            device,
            warmup,
            rep,
        )

    num_row_tiles = _cdiv(ROWS, TILE_ROWS)
    logical_read_bytes = (
        ROWS * CHANNELS * 4
        + CHANNELS * 4
        + ROWS * CHANNELS * 4
        + TOKENS * CHANNELS * 4
        + ROWS * 4
        + ROWS * 4
        + ROWS * CHANNELS * 4
    )
    logical_write_bytes = (
        TOKENS * CHANNELS * 4
        + 3 * num_row_tiles * CHANNELS * 4
        + 3 * CHANNELS * 4
    )
    print(
        f"oracle_full cooperative split-k ViT patch/LN tuple: {oracle_us:.3f} us "
        f"impl={actual_impl} shape={SHAPE_LABEL} device={device}"
    )
    print(f"logical traffic including partials: {(logical_read_bytes + logical_write_bytes) / 1.0e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare against the full Repro.forward tuple")
    parser.add_argument("--bench", action="store_true", help="run timing benchmark")
    parser.add_argument("--impl", choices=("auto", "triton", "torch"), default="auto")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
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
