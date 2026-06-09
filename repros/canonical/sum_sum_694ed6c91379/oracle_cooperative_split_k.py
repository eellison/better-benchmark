"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full RMSNorm-backward tail by summing the three BF16 matmul-gradient inputs once, keeping each row's hidden-dimension dot product in registers, writing the returned transposed BF16 input-gradient side output directly, and cooperatively accumulating the BF16 weight-gradient column reduction from the same row tiles, whereas Inductor currently schedules the BF16 add/multiply producer, row reduction, transposed side-output store, and sibling `[1024]` sum as separate generic pointwise and reduction regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates row-local reductions, a layout-changing side output, BF16 intermediate rounding, and a cross-row column reduction in one planned kernel pair; the fix is COOPERATIVE_SPLIT_K: add an RMSNorm-backward row-tiled lowering that preserves the graph's BF16 casts, stores transposed outputs in target layout, and finalizes compatible column partials across row tiles."""
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
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_ID = "sum_sum_694ed6c91379"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 4
SEQ = 512
ROWS = BATCH * SEQ
CHANNELS = 1024
INV_CHANNELS = 1.0 / CHANNELS

ROW_GROUP = 8
BLOCK_CHANNELS = 1024
FINAL_BLOCK_TILES = 256
FINAL_BLOCK_CHANNELS = 16


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _rmsnorm_row_tile_kernel(
        mm0_ptr,
        mm1_ptr,
        mm2_ptr,
        weight_ptr,
        x_ptr,
        rstd_ptr,
        residual_ptr,
        out_transposed_ptr,
        partial_weight_grad_ptr,
        ROWS_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        INV_CHANNELS_: tl.constexpr,
        ROW_GROUP_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        tile = tl.program_id(0)
        rows = tile * ROW_GROUP_ + tl.arange(0, ROW_GROUP_)
        channels = tl.arange(0, BLOCK_CHANNELS_)
        row_mask = rows < ROWS_
        channel_mask = channels < CHANNELS_
        mask = row_mask[:, None] & channel_mask[None, :]
        offsets = rows[:, None] * CHANNELS_ + channels[None, :]

        mm0 = tl.load(mm0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm1 = tl.load(mm1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mm2 = tl.load(mm2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        add01 = (mm0 + mm1).to(tl.bfloat16)
        grad = (add01.to(tl.float32) + mm2).to(tl.bfloat16)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        rstd = tl.load(rstd_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(
            tl.float32
        )

        xhat = (x * rstd[:, None]).to(tl.bfloat16)
        weight_grad_term = (grad.to(tl.float32) * xhat.to(tl.float32)).to(
            tl.bfloat16
        )
        partial_weight_grad = tl.sum(
            tl.where(mask, weight_grad_term.to(tl.float32), 0.0),
            axis=0,
        )
        partial_offsets = tile * CHANNELS_ + channels
        tl.store(
            partial_weight_grad_ptr + partial_offsets,
            partial_weight_grad,
            mask=channel_mask,
        )

        weighted = (grad.to(tl.float32) * weight[None, :]).to(tl.bfloat16)
        weighted_f32 = weighted.to(tl.float32)
        row_dot = tl.sum(tl.where(mask, weighted_f32 * x, 0.0), axis=1)
        rstd3 = rstd * rstd * rstd
        correction = ((row_dot * -0.5) * rstd3 * INV_CHANNELS_)[:, None]
        dx = weighted_f32 * rstd[:, None] + correction * (x * 2.0)
        dx_bf16 = dx.to(tl.bfloat16)

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        out = (residual + dx_bf16.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_transposed_ptr + offsets, out, mask=mask)

    @triton.jit
    def _finalize_weight_grad_kernel(
        partial_weight_grad_ptr,
        out_weight_grad_ptr,
        NUM_ROW_TILES_: tl.constexpr,
        CHANNELS_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
    ):
        channels = tl.program_id(0) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        channel_mask = channels < CHANNELS_
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        acc = tl.zeros((BLOCK_CHANNELS_,), dtype=tl.float32)

        for tile_base in range(0, NUM_ROW_TILES_, BLOCK_TILES_):
            tiles = tile_base + tile_offsets
            mask = (tiles[:, None] < NUM_ROW_TILES_) & channel_mask[None, :]
            offsets = tiles[:, None] * CHANNELS_ + channels[None, :]
            values = tl.load(partial_weight_grad_ptr + offsets, mask=mask, other=0.0)
            acc += tl.sum(values.to(tl.float32), axis=0)

        tl.store(out_weight_grad_ptr + channels, acc, mask=channel_mask)


@oracle_impl(hardware="H100", shapes="(T([2048, 1024], bf16), T([2048, 1024], bf16), T([2048, 1024], bf16), T([1024], bf16), T([4, 512, 1024], bf16), T([4, 512, 1], f32), T([4, 512, 1024], bf16), S([4, 512, 1024]), S([4, 512, 1024]), S([4, 512, 1024]), S([1024]), S([4, 512, 1024]), S([2048, 1024]))")
def oracle_forward(inputs):
    """Run the full-scope Triton oracle."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    (
        mm_375,
        mm_377,
        mm_379,
        arg14_1,
        arg333_1,
        arg334_1,
        add_397,
        *_shape_params,
    ) = inputs

    if mm_375.device.type != "cuda":
        raise RuntimeError("CUDA inputs are required for this Triton oracle")

    assert mm_375.shape == (ROWS, CHANNELS)
    assert mm_377.shape == (ROWS, CHANNELS)
    assert mm_379.shape == (ROWS, CHANNELS)
    assert arg14_1.shape == (CHANNELS,)
    assert arg333_1.shape == (BATCH, SEQ, CHANNELS)
    assert arg334_1.shape == (BATCH, SEQ, 1)
    assert add_397.shape == (BATCH, SEQ, CHANNELS)
    assert mm_375.is_contiguous()
    assert mm_377.is_contiguous()
    assert mm_379.is_contiguous()
    assert arg14_1.is_contiguous()
    assert arg333_1.is_contiguous()
    assert arg334_1.is_contiguous()
    assert add_397.is_contiguous()

    device = mm_375.device
    num_row_tiles = triton.cdiv(ROWS, ROW_GROUP)
    partial_weight_grad = torch.empty(
        (num_row_tiles, CHANNELS),
        device=device,
        dtype=torch.float32,
    )
    out_weight_grad = torch.empty((CHANNELS,), device=device, dtype=torch.bfloat16)
    out_transposed = torch.empty_strided(
        (CHANNELS, ROWS),
        (1, CHANNELS),
        device=device,
        dtype=torch.bfloat16,
    )

    _rmsnorm_row_tile_kernel[(num_row_tiles,)](
        mm_375,
        mm_377,
        mm_379,
        arg14_1,
        arg333_1,
        arg334_1,
        add_397,
        out_transposed,
        partial_weight_grad,
        ROWS_=ROWS,
        CHANNELS_=CHANNELS,
        INV_CHANNELS_=INV_CHANNELS,
        ROW_GROUP_=ROW_GROUP,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        num_warps=8,
    )

    _finalize_weight_grad_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_CHANNELS),)](
        partial_weight_grad,
        out_weight_grad,
        NUM_ROW_TILES_=num_row_tiles,
        CHANNELS_=CHANNELS,
        BLOCK_TILES_=FINAL_BLOCK_TILES,
        BLOCK_CHANNELS_=FINAL_BLOCK_CHANNELS,
        num_warps=8,
    )

    return out_weight_grad, out_transposed


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
