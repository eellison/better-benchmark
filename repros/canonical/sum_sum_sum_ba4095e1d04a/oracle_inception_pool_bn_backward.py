"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Inception avg-pool-backward plus dual BatchNorm-backward tail returned by Repro.forward, including the three residual adds, the 384/96 channel slices, both affine-ReLU masks, four sibling channel reductions, vector epilogues, and the two full tensor BN-backward epilogues, whereas Inductor currently schedules the avg-pool backward, slice/pointwise mask producers, duplicate reductions, and tensor epilogues as separate generic kernels over materialized intermediates; Inductor cannot do this today because its scheduler does not build one dependent multi-output reduction plan that shares a nontrivial pooled producer across sibling channel reductions and then sinks the finalized summaries into both tensor outputs; the fix is SCHEDULER_FUSION: add a full-scope reduction schedule for pooled producers feeding multiple per-channel BN-backward reductions and their fused epilogues."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 128
FULL_CHANNELS = 768
CHANNELS_384 = 384
CHANNELS_96 = 96
RETURN_CHANNELS = CHANNELS_384 + CHANNELS_96
HEIGHT = 17
WIDTH = 17
HW = HEIGHT * WIDTH
NHW = BATCH * HW
INV_NHW = 2.703287197231834e-05

SHAPE_384 = (BATCH, CHANNELS_384, HEIGHT, WIDTH)
STRIDE_384 = (CHANNELS_384 * HW, HW, WIDTH, 1)
SHAPE_96 = (BATCH, CHANNELS_96, HEIGHT, WIDTH)
STRIDE_96 = (CHANNELS_96 * HW, HW, WIDTH, 1)
SHAPE_FULL = (BATCH, FULL_CHANNELS, HEIGHT, WIDTH)
STRIDE_FULL = (FULL_CHANNELS * HW, HW, WIDTH, 1)
SHAPE_VEC_384 = (CHANNELS_384,)
SHAPE_VEC_96 = (CHANNELS_96,)
STRIDE_VEC = (1,)

REDUCE_BLOCK = 1024
EPILOGUE_BLOCK = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_where_reduce_kernel(
        grad_pool_ptr,
        add0_ptr,
        add1_ptr,
        add2_ptr,
        act96_ptr,
        mean96_ptr,
        invstd96_ptr,
        weight96_ptr,
        bias96_ptr,
        act384_ptr,
        mean384_ptr,
        invstd384_ptr,
        weight384_ptr,
        bias384_ptr,
        full_ptr,
        out96_ptr,
        out384_ptr,
        partial_sum_ptr,
        partial_dot_ptr,
        nhw: tl.constexpr,
        hw_size: tl.constexpr,
        width: tl.constexpr,
        height: tl.constexpr,
        full_channels: tl.constexpr,
        channels_384: tl.constexpr,
        channels_96: tl.constexpr,
        num_tiles: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        total_c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < nhw

        n = k // hw_size
        hw = k - n * hw_size
        h = hw // width
        w = hw - h * width

        full_offset = n * full_channels * hw_size + total_c * hw_size + hw
        up = h > 0
        down = h < (height - 1)
        left = w > 0
        right = w < (width - 1)

        pool_sum = tl.load(grad_pool_ptr + full_offset, mask=active, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset - width, mask=active & up, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset + width, mask=active & down, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset - 1, mask=active & left, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset + 1, mask=active & right, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset - width - 1, mask=active & up & left, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset - width + 1, mask=active & up & right, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset + width - 1, mask=active & down & left, other=0.0).to(tl.float32)
        pool_sum += tl.load(grad_pool_ptr + full_offset + width + 1, mask=active & down & right, other=0.0).to(tl.float32)

        pooled = pool_sum * 0.1111111111111111
        slice_value = (
            pooled
            + tl.load(add0_ptr + full_offset, mask=active, other=0.0).to(tl.float32)
            + tl.load(add1_ptr + full_offset, mask=active, other=0.0).to(tl.float32)
            + tl.load(add2_ptr + full_offset, mask=active, other=0.0).to(tl.float32)
        )

        is_384 = total_c < channels_384
        c384 = tl.minimum(total_c, channels_384 - 1)
        c96 = total_c - channels_384
        c96_safe = tl.maximum(c96, 0)

        offset384 = n * channels_384 * hw_size + c384 * hw_size + hw
        offset96 = n * channels_96 * hw_size + c96_safe * hw_size + hw

        act384 = tl.load(act384_ptr + offset384, mask=active & is_384, other=0.0).to(tl.float32)
        act96 = tl.load(act96_ptr + offset96, mask=active & ~is_384, other=0.0).to(tl.float32)
        act = tl.where(is_384, act384, act96)

        mean384 = tl.load(mean384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        mean96 = tl.load(mean96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)
        invstd384 = tl.load(invstd384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        invstd96 = tl.load(invstd96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)
        weight384 = tl.load(weight384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        weight96 = tl.load(weight96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)
        bias384 = tl.load(bias384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        bias96 = tl.load(bias96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)

        mean = tl.where(is_384, mean384, mean96)
        invstd = tl.where(is_384, invstd384, invstd96)
        weight = tl.where(is_384, weight384, weight96)
        bias = tl.where(is_384, bias384, bias96)
        centered = act - mean
        pre_relu = centered * invstd * weight + bias
        full_value = tl.load(full_ptr).to(tl.float32)
        where_value = tl.where(pre_relu <= 0.0, full_value, slice_value)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)

        tl.store(out384_ptr + offset384, where_value, mask=active & is_384)
        tl.store(out96_ptr + offset96, where_value, mask=active & ~is_384)

        partial_offset = total_c * num_tiles + tile
        tl.store(partial_sum_ptr + partial_offset, tl.sum(where_value, axis=0))
        tl.store(partial_dot_ptr + partial_offset, tl.sum(where_value * centered, axis=0))

    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_dot_ptr,
        invstd96_ptr,
        weight96_ptr,
        invstd384_ptr,
        weight384_ptr,
        stats_ptr,
        out_vec96_ptr,
        out_vec384_ptr,
        channels_384: tl.constexpr,
        return_channels: tl.constexpr,
        inv_nhw: tl.constexpr,
        num_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        total_c = tl.program_id(0)
        tiles = tl.arange(0, BLOCK_TILES)
        tile_mask = tiles < num_tiles
        partial_offsets = total_c * num_tiles + tiles

        sum_where = tl.sum(
            tl.load(partial_sum_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32),
            axis=0,
        )
        sum_dot = tl.sum(
            tl.load(partial_dot_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32),
            axis=0,
        )

        is_384 = total_c < channels_384
        c384 = tl.minimum(total_c, channels_384 - 1)
        c96 = total_c - channels_384
        c96_safe = tl.maximum(c96, 0)

        invstd384 = tl.load(invstd384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        invstd96 = tl.load(invstd96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)
        weight384 = tl.load(weight384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        weight96 = tl.load(weight96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)
        invstd = tl.where(is_384, invstd384, invstd96)
        weight = tl.where(is_384, weight384, weight96)

        mean_term = sum_where * inv_nhw
        dot_coeff = sum_dot * inv_nhw * invstd * invstd
        scale = invstd * weight

        tl.store(stats_ptr + total_c, mean_term)
        tl.store(stats_ptr + return_channels + total_c, dot_coeff)
        tl.store(stats_ptr + 2 * return_channels + total_c, scale)
        tl.store(out_vec384_ptr + c384, sum_dot * invstd, mask=is_384)
        tl.store(out_vec96_ptr + c96_safe, sum_dot * invstd, mask=~is_384)

    @triton.jit
    def _bn_epilogue_kernel(
        act96_ptr,
        mean96_ptr,
        act384_ptr,
        mean384_ptr,
        stats_ptr,
        out96_ptr,
        out384_ptr,
        nhw: tl.constexpr,
        hw_size: tl.constexpr,
        channels_384: tl.constexpr,
        channels_96: tl.constexpr,
        return_channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        total_c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK + tl.arange(0, BLOCK)
        active = k < nhw

        n = k // hw_size
        hw = k - n * hw_size

        is_384 = total_c < channels_384
        c384 = tl.minimum(total_c, channels_384 - 1)
        c96 = total_c - channels_384
        c96_safe = tl.maximum(c96, 0)

        offset384 = n * channels_384 * hw_size + c384 * hw_size + hw
        offset96 = n * channels_96 * hw_size + c96_safe * hw_size + hw

        where384 = tl.load(out384_ptr + offset384, mask=active & is_384, other=0.0).to(tl.float32)
        where96 = tl.load(out96_ptr + offset96, mask=active & ~is_384, other=0.0).to(tl.float32)
        act384 = tl.load(act384_ptr + offset384, mask=active & is_384, other=0.0).to(tl.float32)
        act96 = tl.load(act96_ptr + offset96, mask=active & ~is_384, other=0.0).to(tl.float32)
        mean384 = tl.load(mean384_ptr + c384, mask=is_384, other=0.0).to(tl.float32)
        mean96 = tl.load(mean96_ptr + c96_safe, mask=~is_384, other=0.0).to(tl.float32)

        where_value = tl.where(is_384, where384, where96)
        centered = tl.where(is_384, act384 - mean384, act96 - mean96)
        mean_term = tl.load(stats_ptr + total_c).to(tl.float32)
        dot_coeff = tl.load(stats_ptr + return_channels + total_c).to(tl.float32)
        scale = tl.load(stats_ptr + 2 * return_channels + total_c).to(tl.float32)
        out_value = (where_value - centered * dot_coeff - mean_term) * scale

        tl.store(out384_ptr + offset384, out_value, mask=active & is_384)
        tl.store(out96_ptr + offset96, out_value, mask=active & ~is_384)


def _require_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 16:
        raise ValueError(f"{REPRO_ID} expects 16 inputs, got {len(inputs)}")

    (
        getitem_162,
        arg337_1,
        getitem_177,
        getitem_186,
        getitem_189,
        arg333_1,
        arg334_1,
        arg335_1,
        arg75_1,
        arg76_1,
        full_1,
        arg324_1,
        arg325_1,
        arg326_1,
        arg68_1,
        arg69_1,
    ) = inputs

    for name, tensor in (
        ("getitem_162", getitem_162),
        ("arg337_1", arg337_1),
        ("getitem_177", getitem_177),
        ("getitem_186", getitem_186),
        ("getitem_189", getitem_189),
    ):
        _require_f32_tensor(name, tensor, SHAPE_FULL, STRIDE_FULL)

    for name, tensor in (("arg333_1", arg333_1),):
        _require_f32_tensor(name, tensor, SHAPE_96, STRIDE_96)
    for name, tensor in (("arg324_1", arg324_1),):
        _require_f32_tensor(name, tensor, SHAPE_384, STRIDE_384)

    for name, tensor in (("arg334_1", arg334_1), ("arg335_1", arg335_1)):
        _require_f32_tensor(name, tensor, (1, CHANNELS_96, 1, 1), (CHANNELS_96, 1, 1, 1))
    for name, tensor in (("arg325_1", arg325_1), ("arg326_1", arg326_1)):
        _require_f32_tensor(name, tensor, (1, CHANNELS_384, 1, 1), (CHANNELS_384, 1, 1, 1))
    for name, tensor in (("arg75_1", arg75_1), ("arg76_1", arg76_1)):
        _require_f32_tensor(name, tensor, SHAPE_VEC_96, STRIDE_VEC)
    for name, tensor in (("arg68_1", arg68_1), ("arg69_1", arg69_1)):
        _require_f32_tensor(name, tensor, SHAPE_VEC_384, STRIDE_VEC)
    _require_f32_tensor("full_1", full_1, (), ())

    device = getitem_162.device
    for value in inputs:
        if isinstance(value, torch.Tensor) and value.device != device:
            raise ValueError("all tensor inputs must be on the same CUDA device")

    return (
        getitem_162,
        arg337_1,
        getitem_177,
        getitem_186,
        getitem_189,
        arg333_1,
        arg334_1,
        arg335_1,
        arg75_1,
        arg76_1,
        full_1,
        arg324_1,
        arg325_1,
        arg326_1,
        arg68_1,
        arg69_1,
    )


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    (
        getitem_162,
        _arg337_1,
        getitem_177,
        getitem_186,
        getitem_189,
        arg333_1,
        arg334_1,
        arg335_1,
        arg75_1,
        arg76_1,
        full_1,
        arg324_1,
        arg325_1,
        arg326_1,
        arg68_1,
        arg69_1,
    ) = _validate_inputs(tuple(inputs))
    del _arg337_1

    device = getitem_162.device
    num_tiles = triton.cdiv(NHW, REDUCE_BLOCK)
    block_tiles = triton.next_power_of_2(num_tiles)

    out96 = torch.empty_strided(SHAPE_96, STRIDE_96, device=device, dtype=torch.float32)
    out384 = torch.empty_strided(SHAPE_384, STRIDE_384, device=device, dtype=torch.float32)
    out_vec96 = torch.empty_strided(SHAPE_VEC_96, STRIDE_VEC, device=device, dtype=torch.float32)
    out_vec384 = torch.empty_strided(SHAPE_VEC_384, STRIDE_VEC, device=device, dtype=torch.float32)
    partial_sum = torch.empty_strided(
        (RETURN_CHANNELS, num_tiles),
        (num_tiles, 1),
        device=device,
        dtype=torch.float32,
    )
    partial_dot = torch.empty_strided(
        (RETURN_CHANNELS, num_tiles),
        (num_tiles, 1),
        device=device,
        dtype=torch.float32,
    )

    _partial_where_reduce_kernel[(RETURN_CHANNELS, num_tiles)](
        getitem_162,
        getitem_177,
        getitem_186,
        getitem_189,
        arg333_1,
        arg334_1,
        arg335_1,
        arg75_1,
        arg76_1,
        arg324_1,
        arg325_1,
        arg326_1,
        arg68_1,
        arg69_1,
        full_1,
        out96,
        out384,
        partial_sum,
        partial_dot,
        NHW,
        HW,
        WIDTH,
        HEIGHT,
        FULL_CHANNELS,
        CHANNELS_384,
        CHANNELS_96,
        num_tiles,
        BLOCK=REDUCE_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    stats = torch.empty_strided(
        (3, RETURN_CHANNELS),
        (RETURN_CHANNELS, 1),
        device=device,
        dtype=torch.float32,
    )
    _finalize_stats_kernel[(RETURN_CHANNELS,)](
        partial_sum,
        partial_dot,
        arg335_1,
        arg75_1,
        arg326_1,
        arg68_1,
        stats,
        out_vec96,
        out_vec384,
        CHANNELS_384,
        RETURN_CHANNELS,
        INV_NHW,
        num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=8,
        num_stages=4,
    )

    _bn_epilogue_kernel[(RETURN_CHANNELS, triton.cdiv(NHW, EPILOGUE_BLOCK))](
        arg333_1,
        arg334_1,
        arg324_1,
        arg325_1,
        stats,
        out96,
        out384,
        NHW,
        HW,
        CHANNELS_384,
        CHANNELS_96,
        RETURN_CHANNELS,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    return out96, out_vec96, out384, out_vec384


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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
