"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DenseNet BN-backward tail from `Repro.forward` by split-K reducing the shared masked producer into two per-channel accumulators, finalizing the `[160]` vector output, and using the finalized summaries in a slice-limited epilogue that adds all eleven live residual slices for channels 128:160; whereas Inductor currently schedules the sibling reductions, broadcast BN-backward epilogue, and residual slice-add chain as generic regions around materialized intermediates. Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that keeps compatible channel reductions, dependent vector math, and a sliced full-tensor side output in one coordinated plan. The fix is COOPERATIVE_SPLIT_K: add a split-K multi-accumulator reduction lowering that finalizes sibling channel summaries once and sinks the dependent live-channel epilogue and residual slice reads into the same full-scope plan."""
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
C = 160
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 128
SLICE_C = C - SLICE_START
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 1.992984693877551e-05


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
    def _finalize_reduce_kernel(
        partial_sum_where_ptr,
        partial_sum_centered_ptr,
        invstd_ptr,
        sum_where_ptr,
        sum_centered_ptr,
        vector_out_ptr,
        num_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0)
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

        tl.store(sum_where_ptr + c, sum_where)
        tl.store(sum_centered_ptr + c, sum_centered)
        tl.store(vector_out_ptr + c, sum_centered * invstd)


    @triton.jit
    def _slice_add_epilogue_kernel(
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
        mask_input_ptr,
        full_ptr,
        source_ptr,
        centered_source_ptr,
        mean_ptr,
        invstd_ptr,
        affine_weight_ptr,
        sum_where_ptr,
        sum_centered_ptr,
        add_out_ptr,
        NUMEL_SLICE_: tl.constexpr,
        C_: tl.constexpr,
        HW_: tl.constexpr,
        SLICE_START_: tl.constexpr,
        SLICE_C_: tl.constexpr,
        SCALE_: tl.constexpr,
        BLOCK_ELEMS: tl.constexpr,
    ):
        out_offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
        active = out_offsets < NUMEL_SLICE_

        hw = out_offsets % HW_
        slice_c = (out_offsets // HW_) % SLICE_C_
        n = out_offsets // (SLICE_C_ * HW_)
        c = SLICE_START_ + slice_c
        input_offsets = n * (C_ * HW_) + c * HW_ + hw

        residual_offsets0 = n * (512 * HW_) + c * HW_ + hw
        residual_offsets1 = n * (480 * HW_) + c * HW_ + hw
        residual_offsets2 = n * (448 * HW_) + c * HW_ + hw
        residual_offsets3 = n * (416 * HW_) + c * HW_ + hw
        residual_offsets4 = n * (384 * HW_) + c * HW_ + hw
        residual_offsets5 = n * (352 * HW_) + c * HW_ + hw
        residual_offsets6 = n * (320 * HW_) + c * HW_ + hw
        residual_offsets7 = n * (288 * HW_) + c * HW_ + hw
        residual_offsets8 = n * (256 * HW_) + c * HW_ + hw
        residual_offsets9 = n * (224 * HW_) + c * HW_ + hw
        residual_offsets10 = n * (192 * HW_) + c * HW_ + hw

        residual = tl.load(residual0_ptr + residual_offsets0, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual1_ptr + residual_offsets1, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual2_ptr + residual_offsets2, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual3_ptr + residual_offsets3, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual4_ptr + residual_offsets4, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual5_ptr + residual_offsets5, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual6_ptr + residual_offsets6, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual7_ptr + residual_offsets7, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual8_ptr + residual_offsets8, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual9_ptr + residual_offsets9, mask=active, other=0.0).to(tl.float32)
        residual += tl.load(residual10_ptr + residual_offsets10, mask=active, other=0.0).to(tl.float32)

        mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
        full_value = tl.load(full_ptr).to(tl.float32)
        source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
        centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
        affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
        sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
        sum_centered = tl.load(sum_centered_ptr + c, mask=active, other=0.0).to(tl.float32)

        where_self = tl.where(mask_input <= 0.0, full_value, source)
        centered = centered_source - mean
        mean_term = sum_where * SCALE_
        variance_term = sum_centered * SCALE_ * invstd * invstd
        grad = (where_self - centered * variance_term - mean_term) * (invstd * affine_weight)

        tl.store(add_out_ptr + out_offsets, residual + grad, mask=active)


def _oracle_fused(
    mul_747: torch.Tensor,
    mul_765: torch.Tensor,
    mul_783: torch.Tensor,
    mul_801: torch.Tensor,
    mul_819: torch.Tensor,
    mul_837: torch.Tensor,
    mul_855: torch.Tensor,
    mul_873: torch.Tensor,
    mul_891: torch.Tensor,
    mul_909: torch.Tensor,
    mul_927: torch.Tensor,
    arg296_1: torch.Tensor,
    full: torch.Tensor,
    getitem_309: torch.Tensor,
    arg294_1: torch.Tensor,
    arg715_1: torch.Tensor,
    arg295_1: torch.Tensor,
    arg34_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_densenet_splitk.py")

    residuals = (
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        mul_891,
        mul_909,
        mul_927,
    )
    expected_residual_channels = (512, 480, 448, 416, 384, 352, 320, 288, 256, 224, 192)
    for residual, residual_c in zip(residuals, expected_residual_channels):
        assert residual.shape == (N, residual_c, H, W)
        assert residual.is_contiguous()

    assert arg296_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_309.shape == (N, C, H, W)
    assert arg294_1.shape == (N, C, H, W)
    assert arg715_1.shape == (1, C, 1, 1)
    assert arg295_1.shape == (C,)
    assert arg34_1.shape == (C,)
    assert arg296_1.is_contiguous()
    assert getitem_309.is_contiguous()
    assert arg294_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum_where = torch.empty((C, num_tiles), device=arg296_1.device, dtype=torch.float32)
    partial_sum_centered = torch.empty((C, num_tiles), device=arg296_1.device, dtype=torch.float32)
    sum_where = torch.empty((C,), device=arg296_1.device, dtype=torch.float32)
    sum_centered = torch.empty((C,), device=arg296_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg296_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg296_1,
        full,
        getitem_309,
        arg294_1,
        arg715_1,
        partial_sum_where,
        partial_sum_centered,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_reduce_kernel[(C,)](
        partial_sum_where,
        partial_sum_centered,
        arg295_1,
        sum_where,
        sum_centered,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg296_1.device, dtype=torch.float32)
    block_elems = 1024
    _slice_add_epilogue_kernel[(triton.cdiv(SLICE_NUMEL, block_elems),)](
        mul_747,
        mul_765,
        mul_783,
        mul_801,
        mul_819,
        mul_837,
        mul_855,
        mul_873,
        mul_891,
        mul_909,
        mul_927,
        arg296_1,
        full,
        getitem_309,
        arg294_1,
        arg715_1,
        arg295_1,
        arg34_1,
        sum_where,
        sum_centered,
        add_out,
        NUMEL_SLICE_=SLICE_NUMEL,
        C_=C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return vector_out, add_out


def oracle_forward(inputs):
    """Run the full-scope oracle for `Repro()(*make_inputs())`."""
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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
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
