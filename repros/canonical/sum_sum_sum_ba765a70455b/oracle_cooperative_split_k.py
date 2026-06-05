"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full `Repro.forward` Swin window-reverse layer-norm-backward/drop-path return tuple by row-tiling the `[25088, 512]` producer, reconstructing the fused window layout, writing the returned non-contiguous `[512, 25088]` transposed dropped-gradient side output, and cooperatively accumulating the two pre-drop `[512]` column reductions plus the post-drop `[512]` column reduction from the same tiles, whereas Inductor currently schedules the reshape/view/permute/clone layout change, hidden-dimension row reductions, drop-path epilogue, transposed side-output store, and sibling `sum([0, 1, 2])`/`sum([0])` reductions as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps a layout-changing producer, row-local reduction scalars, a dependent full-tensor side store, and multiple compatible column accumulators in one coordinated producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible layer-norm-backward column reductions across row tiles, fuse the window-layout remap and drop-path transpose store, and finalize all sibling column partials together."""
from __future__ import annotations

import argparse

import torch
from pathlib import Path
import triton
import triton.language as tl

import repro as repro_module

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_sum_ba765a70455b"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"


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
