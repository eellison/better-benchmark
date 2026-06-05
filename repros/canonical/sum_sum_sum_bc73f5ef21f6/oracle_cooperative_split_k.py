"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DeiT layer-norm-backward and positional-embedding return tuple by row-tiling the `[128, 198, 768]` producer, computing each row's hidden-dimension reductions once, accumulating the two global channel reductions, the full `[1, 198, 768]` token reduction, the class/dist-token reductions, and the patch-token reduction without materializing the intermediate gradient tensor, whereas Inductor currently schedules the view, row reductions, dependent layer-norm-backward pointwise expression, residual add, slice/permute/view path, and sibling reductions as separate generic pointwise/reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that combines row-local scalar reductions with multiple compatible batch/token/channel reductions and view-equivalent epilogues in one coordinated producer/finalizer; the fix is COOPERATIVE_SPLIT_K: add an Inductor split-row multi-output reduction template for layer-norm-backward tails that shares row scalars, emits partial accumulators for token/global channel reductions, and finalizes the full return tuple without materializing the intermediate gradient tensor."""
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


REPRO_ID = "sum_sum_sum_bc73f5ef21f6"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_deit_base_distilled_patch16_224_train_001_5ae49b96"

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
        arg3_1,
        arg78_1,
        arg2_1,
        arg79_1,
        arg80_1,
        add_47,
        *_shape_params,
    ) = inputs

    return (
        mm_98.reshape(ROWS, CHANNELS).contiguous(),
        arg3_1.contiguous(),
        arg78_1.reshape(ROWS, CHANNELS).contiguous(),
        arg2_1.reshape(TOKENS, CHANNELS).contiguous(),
        arg79_1.reshape(ROWS).contiguous(),
        arg80_1.reshape(ROWS).contiguous(),
        add_47.reshape(ROWS, CHANNELS).contiguous(),
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
