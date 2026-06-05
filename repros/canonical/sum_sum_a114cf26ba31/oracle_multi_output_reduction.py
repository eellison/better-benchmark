"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet BN-backward tail from `Repro.forward` by sharing the ReLU-mask `where`, channel centering, and two sibling per-channel reductions across all C=544 channels, then using the finalized reductions in the dependent epilogue for live channels 512:544 while adding all 15 residual slices into the returned side output; whereas Inductor schedules the residual slice-add chain, sibling reductions, reduction-dependent BN epilogue, and final slice/add as ordinary separate regions. Inductor cannot do this today because the scheduler does not form one full-scope multi-output reduction plan that shares the masked/centered producer and sinks the side-output epilogue to the returned channel slice. The fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse compatible sibling reductions with a dependent slice-limited epilogue and keep the residual slice chain inside that fused plan."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 544
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 512
SLICE_C = C - SLICE_START
SCALE = 7.971938775510203e-05


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_reduce_partial_kernel(
        mask_input_ptr,
        full_ptr,
        source_ptr,
        centered_source_ptr,
        mean_ptr,
        partial_sum_where_ptr,
        partial_sum_centered_ptr,
        num_tiles: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_SPATIAL_: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        active = k < TOTAL_SPATIAL_

        n = k // HW_
        hw = k - n * HW_
        offsets = n * (C_ * HW_) + c * HW_ + hw

        mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c).to(tl.float32)

        where_self = tl.where(mask_input <= 0.0, full_value, source)
        where_self = tl.where(active, where_self, 0.0)
        centered = centered_source - mean

        partial_offset = c * num_tiles + tile
        tl.store(partial_sum_where_ptr + partial_offset, tl.sum(where_self, axis=0))
        tl.store(partial_sum_centered_ptr + partial_offset, tl.sum(where_self * centered, axis=0))


    @triton.jit
    def _finalize_and_slice_epilogue_kernel(
        residual0_ptr,
        residual1_ptr,
        residual2_ptr,
        residual3_ptr,
        residual4_ptr,
        residual5_ptr,
        residual6_ptr,
        residual7_ptr,
        residual8_ptr,
        residual9_ptr,
        residual10_ptr,
        residual11_ptr,
        residual12_ptr,
        residual13_ptr,
        residual14_ptr,
        mask_input_ptr,
        full_ptr,
        source_ptr,
        centered_source_ptr,
        mean_ptr,
        invstd_ptr,
        affine_weight_ptr,
        partial_sum_where_ptr,
        partial_sum_centered_ptr,
        vector_out_ptr,
        add_out_ptr,
        num_tiles: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_SPATIAL_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        SLICE_C_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
        BLOCKS_PER_CHANNEL: tl.constexpr,
    ):
        pid = tl.program_id(0)
        is_vector_program = pid < C_
        extra_pid = pid - C_

        extra_slice_c = extra_pid // (BLOCKS_PER_CHANNEL - 1)
        extra_block = extra_pid - extra_slice_c * (BLOCKS_PER_CHANNEL - 1) + 1
        c = tl.where(is_vector_program, pid, SLICE_START_ + extra_slice_c)
        block = tl.where(is_vector_program, 0, extra_block)

        tiles = tl.arange(0, BLOCK_TILES)
        tile_mask = tiles < num_tiles
        partial_offsets = c * num_tiles + tiles
        sum_where_values = tl.load(
            partial_sum_where_ptr + partial_offsets,
            mask=tile_mask,
            other=0.0,
        ).to(tl.float32)
        sum_centered_values = tl.load(
            partial_sum_centered_ptr + partial_offsets,
            mask=tile_mask,
            other=0.0,
        ).to(tl.float32)
        sum_where = tl.sum(sum_where_values, axis=0)
        sum_centered = tl.sum(sum_centered_values, axis=0)
        invstd = tl.load(invstd_ptr + c).to(tl.float32)

        tl.store(vector_out_ptr + c, sum_centered * invstd, mask=is_vector_program)

        k = block * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        active = k < TOTAL_SPATIAL_
        slice_c = c - SLICE_START_
        in_slice = c >= SLICE_START_
        epilogue_mask = active & in_slice

        n = k // HW_
        hw = k - n * HW_
        input_offsets = n * (C_ * HW_) + c * HW_ + hw
        output_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + hw

        residual_offsets0 = n * (1024 * HW_) + c * HW_ + hw
        residual_offsets1 = n * (992 * HW_) + c * HW_ + hw
        residual_offsets2 = n * (960 * HW_) + c * HW_ + hw
        residual_offsets3 = n * (928 * HW_) + c * HW_ + hw
        residual_offsets4 = n * (896 * HW_) + c * HW_ + hw
        residual_offsets5 = n * (864 * HW_) + c * HW_ + hw
        residual_offsets6 = n * (832 * HW_) + c * HW_ + hw
        residual_offsets7 = n * (800 * HW_) + c * HW_ + hw
        residual_offsets8 = n * (768 * HW_) + c * HW_ + hw
        residual_offsets9 = n * (736 * HW_) + c * HW_ + hw
        residual_offsets10 = n * (704 * HW_) + c * HW_ + hw
        residual_offsets11 = n * (672 * HW_) + c * HW_ + hw
        residual_offsets12 = n * (640 * HW_) + c * HW_ + hw
        residual_offsets13 = n * (608 * HW_) + c * HW_ + hw
        residual_offsets14 = n * (576 * HW_) + c * HW_ + hw

        residual = tl.load(residual0_ptr + residual_offsets0, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual1_ptr + residual_offsets1, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual2_ptr + residual_offsets2, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual3_ptr + residual_offsets3, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual4_ptr + residual_offsets4, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual5_ptr + residual_offsets5, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual6_ptr + residual_offsets6, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual7_ptr + residual_offsets7, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual8_ptr + residual_offsets8, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual9_ptr + residual_offsets9, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual10_ptr + residual_offsets10, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual11_ptr + residual_offsets11, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual12_ptr + residual_offsets12, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual13_ptr + residual_offsets13, mask=epilogue_mask, other=0.0).to(tl.float32)
        residual += tl.load(residual14_ptr + residual_offsets14, mask=epilogue_mask, other=0.0).to(tl.float32)

        mask_input = tl.load(mask_input_ptr + input_offsets, mask=epilogue_mask, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        source = tl.load(source_ptr + input_offsets, mask=epilogue_mask, other=0.0).to(tl.float32)
        centered_source = tl.load(centered_source_ptr + input_offsets, mask=epilogue_mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=in_slice, other=0.0).to(tl.float32)
        affine_weight = tl.load(affine_weight_ptr + c, mask=in_slice, other=0.0).to(tl.float32)

        where_self = tl.where(mask_input <= 0.0, full_value, source)
        centered = centered_source - mean
        mean_term = sum_where * SCALE_
        variance_term = sum_centered * SCALE_ * invstd * invstd
        affine_term = invstd * affine_weight
        epilogue = (where_self - centered * variance_term - mean_term) * affine_term
        tl.store(add_out_ptr + output_offsets, residual + epilogue, mask=epilogue_mask)


def _oracle_torch(inputs):
    residual = inputs[0][:, SLICE_START:C, :, :]
    for residual_input in inputs[1:15]:
        residual = residual + residual_input[:, SLICE_START:C, :, :]

    mask_input = inputs[15]
    full = inputs[16]
    source = inputs[17]
    centered_source = inputs[18]
    mean = inputs[19]
    invstd = inputs[20]
    affine_weight = inputs[21]

    where_self = torch.where(mask_input <= 0, full, source)
    centered = centered_source - mean
    sum_where = torch.sum(where_self, dim=(0, 2, 3))
    sum_centered = torch.sum(where_self * centered, dim=(0, 2, 3))
    vector_out = sum_centered * invstd

    mean_term = sum_where * SCALE
    variance_term = sum_centered * SCALE * invstd * invstd
    affine_term = invstd * affine_weight
    epilogue = (
        where_self
        - centered * variance_term[None, :, None, None]
        - mean_term[None, :, None, None]
    ) * affine_term[None, :, None, None]
    return vector_out, residual + epilogue[:, SLICE_START:C, :, :]


def _oracle_fused(
    mul_306: torch.Tensor,
    mul_324: torch.Tensor,
    mul_342: torch.Tensor,
    mul_360: torch.Tensor,
    mul_378: torch.Tensor,
    mul_396: torch.Tensor,
    mul_414: torch.Tensor,
    mul_432: torch.Tensor,
    mul_450: torch.Tensor,
    mul_468: torch.Tensor,
    mul_486: torch.Tensor,
    mul_504: torch.Tensor,
    mul_522: torch.Tensor,
    mul_540: torch.Tensor,
    mul_558: torch.Tensor,
    arg420_1: torch.Tensor,
    full: torch.Tensor,
    getitem_186: torch.Tensor,
    arg418_1: torch.Tensor,
    arg674_1: torch.Tensor,
    arg419_1: torch.Tensor,
    arg116_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_multi_output_reduction.py")

    residuals = (
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
        mul_540,
        mul_558,
    )
    expected_channels = (
        1024,
        992,
        960,
        928,
        896,
        864,
        832,
        800,
        768,
        736,
        704,
        672,
        640,
        608,
        576,
    )
    for residual, channels in zip(residuals, expected_channels):
        assert residual.shape == (N, channels, H, W)
        assert residual.is_contiguous()
    assert arg420_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_186.shape == (N, C, H, W)
    assert arg418_1.shape == (N, C, H, W)
    assert arg674_1.shape == (1, C, 1, 1)
    assert arg419_1.shape == (C,)
    assert arg116_1.shape == (C,)
    assert arg420_1.is_contiguous()
    assert getitem_186.is_contiguous()
    assert arg418_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum_where = torch.empty((C, num_tiles), device=arg420_1.device, dtype=torch.float32)
    partial_sum_centered = torch.empty((C, num_tiles), device=arg420_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg420_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg420_1,
        full,
        getitem_186,
        arg418_1,
        arg674_1,
        partial_sum_where,
        partial_sum_centered,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg420_1.device, dtype=torch.float32)
    block_elems = 512
    blocks_per_channel = triton.cdiv(TOTAL_SPATIAL, block_elems)
    block_tiles = triton.next_power_of_2(num_tiles)
    finalize_programs = C + SLICE_C * (blocks_per_channel - 1)
    _finalize_and_slice_epilogue_kernel[(finalize_programs,)](
        mul_306,
        mul_324,
        mul_342,
        mul_360,
        mul_378,
        mul_396,
        mul_414,
        mul_432,
        mul_450,
        mul_468,
        mul_486,
        mul_504,
        mul_522,
        mul_540,
        mul_558,
        arg420_1,
        full,
        getitem_186,
        arg418_1,
        arg674_1,
        arg419_1,
        arg116_1,
        partial_sum_where,
        partial_sum_centered,
        vector_out,
        add_out,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        BLOCK_TILES=block_tiles,
        BLOCK_ELEMS=block_elems,
        BLOCKS_PER_CHANNEL=blocks_per_channel,
        num_warps=4,
    )

    return vector_out, add_out


def oracle_forward(inputs):
    """Run the full-scope DenseNet BN-backward tail oracle."""
    if triton is None or not inputs[0].is_cuda:
        return _oracle_torch(inputs)
    return _oracle_fused(*inputs)


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
