"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ShuffleNet BN-backward tail by reading the channel-shuffle slice directly from odd channels of the original `[512,464,7,7]` tensor, sharing the ReLU-mask/where producer across two sibling channel reductions, and feeding both sums into the returned full `[512,232,7,7]` epilogue plus `[232]` vector output, whereas Inductor currently lowers the view/permute/clone/slice producer, masked pointwise work, reductions, and dependent broadcast epilogue as generic schedules with avoidable materialization and duplicated producer work; Inductor cannot do this today because the scheduler does not represent the channel shuffle as a virtual layout feeding a multi-output reduction with a dependent full-tensor epilogue; the fix is SCHEDULER_FUSION: teach reduction scheduling to co-schedule compatible sibling reductions while sinking static layout producers and their dependent epilogues into the generated plan."""
from __future__ import annotations

import argparse
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


REPRO_ID = "sum_sum_a793eef69186"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C = 232
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SHUFFLED_C = C * 2
BN_SCALE = 1.0 / TOTAL_SPATIAL


def get_inputs():
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _partial_dual_reduce_kernel(
    shuffled_source_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    full_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    NUM_TILES: tl.constexpr,
    C_: tl.constexpr,
    SHUFFLED_C_: tl.constexpr,
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
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw
    shuffled_offsets = n * (SHUFFLED_C_ * HW_) + (2 * c + 1) * HW_ + hw

    bn_input = tl.load(bn_input_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    shuffled = tl.load(shuffled_source_ptr + shuffled_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = bn_input - mean
    affine = centered * invstd * affine_weight + affine_bias
    where_value = tl.where(affine <= 0.0, full_value, shuffled)
    where_value = tl.where(active, where_value, 0.0)

    partial_offset = c * NUM_TILES + tile
    tl.store(partial_sum1_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(partial_sum2_ptr + partial_offset, tl.sum(where_value * centered, axis=0))


@triton.jit
def _finalize_dual_reduce_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    partial_offsets = c * NUM_TILES + tiles

    sum1_values = tl.load(partial_sum1_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
    sum2_values = tl.load(partial_sum2_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum1_values, axis=0)
    sum2 = tl.sum(sum2_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vector_out_ptr + c, sum2 * invstd)


@triton.jit
def _bn_backward_epilogue_kernel(
    shuffled_source_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    full_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    SHUFFLED_C_: tl.constexpr,
    HW_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    bn_offsets = offsets
    shuffled_offsets = n * (SHUFFLED_C_ * HW_) + (2 * c + 1) * HW_ + hw

    bn_input = tl.load(bn_input_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    shuffled = tl.load(shuffled_source_ptr + shuffled_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = bn_input - mean
    affine = centered * invstd * affine_weight + affine_bias
    where_value = tl.where(affine <= 0.0, full_value, shuffled)

    variance_term = sum2 * BN_SCALE_ * invstd * invstd
    mean_term = sum1 * BN_SCALE_
    result = (where_value - centered * variance_term - mean_term) * (invstd * affine_weight)
    tl.store(out_ptr + offsets, result, mask=active)


def oracle_fused(
    shuffled_source: torch.Tensor,
    bn_input: torch.Tensor,
    mean: torch.Tensor,
    invstd: torch.Tensor,
    affine_weight: torch.Tensor,
    affine_bias: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert shuffled_source.shape == (N, SHUFFLED_C, H, W)
    assert bn_input.shape == (N, C, H, W)
    assert mean.shape == (1, C, 1, 1)
    assert invstd.shape == (1, C, 1, 1)
    assert affine_weight.shape == (C,)
    assert affine_bias.shape == (C,)
    assert full.shape == ()
    assert shuffled_source.stride() == (SHUFFLED_C * HW, HW, W, 1)
    assert bn_input.stride() == (C * HW, HW, W, 1)
    assert mean.stride() == (C, 1, 1, 1)
    assert invstd.stride() == (C, 1, 1, 1)
    assert affine_weight.stride() == (1,)
    assert affine_bias.stride() == (1,)
    assert _shape_param_0 == [N, C, 2, H, W]
    assert _shape_param_1 == [N, SHUFFLED_C, H, W]

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=bn_input.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=bn_input.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=bn_input.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=bn_input.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=bn_input.device, dtype=torch.float32)

    _partial_dual_reduce_kernel[(C, num_tiles)](
        shuffled_source,
        bn_input,
        mean,
        invstd,
        affine_weight,
        affine_bias,
        full,
        partial_sum1,
        partial_sum2,
        NUM_TILES=num_tiles,
        C_=C,
        SHUFFLED_C_=SHUFFLED_C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_dual_reduce_kernel[(C,)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        vector_out,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    out = torch.empty((N, C, H, W), device=bn_input.device, dtype=torch.float32)
    block_elems = 256
    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        shuffled_source,
        bn_input,
        mean,
        invstd,
        affine_weight,
        affine_bias,
        full,
        sum1,
        sum2,
        out,
        NUMEL_=NUMEL,
        C_=C,
        SHUFFLED_C_=SHUFFLED_C,
        HW_=HW,
        BN_SCALE_=BN_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, vector_out


def oracle_forward(inputs):
    """Run the full-scope oracle for Repro.forward()."""
    return oracle_fused(*inputs)


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
                    print(f"WARNING: oracle is slower than compile "
                          f"(ratio={result['ratio']:.3f}x)")
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
