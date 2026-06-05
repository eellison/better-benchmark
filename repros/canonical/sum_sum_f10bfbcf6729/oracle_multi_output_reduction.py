"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full LearningToPaint BN-backward tail by replacing avg_pool2d_backward with `where(arg102_1 <= 0, 0, mm / 16)`, sharing that producer across the two channel reductions over `[0, 2, 3]`, and feeding both sibling sums into the required `[1024, 512, 4, 4]` gradient epilogue plus `[512]` vector output, whereas Inductor currently lowers the decomposed avg-pool-backward/where/sub/mul/sum/sum/unsqueeze/broadcast epilogue graph as generic producer, reduction, and consumer schedules with duplicated reduction input work; Inductor cannot do this today because its scheduler does not form one multi-output channel-reduction schedule whose accumulators are reused by both the vector side output and the full broadcast epilogue; the fix is SCHEDULER_FUSION: teach the reduction scheduler to co-schedule sibling reductions that share a producer and sink their scalar/vector accumulators into dependent broadcast epilogues."""
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



REPRO_ID = "sum_sum_f10bfbcf6729"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 1024
C = 512
H = 4
W = 4
HW = H * W
TOTAL_REDUCE = N * HW
NUMEL = N * C * HW
AVGPOOL_SCALE = 0.0625
BN_SCALE = 6.103515625e-05



@triton.jit
def _dual_reduce_channel_kernel(
    mm_ptr,
    mask_input_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCE_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    acc1 = tl.full((), 0.0, tl.float32)
    acc2 = tl.full((), 0.0, tl.float32)

    for tile in range(0, NUM_TILES):
        k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
        active = k < TOTAL_REDUCE_
        n = k // HW_
        hw = k - n * HW_
        offsets = n * (C_ * HW_) + c * HW_ + hw
        mm_offsets = n * C_ + c

        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
        mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        where_self = tl.where(mask_input <= 0.0, 0.0, pooled_grad)
        where_self = tl.where(active, where_self, 0.0)

        source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered = source - mean
        acc1 += tl.sum(where_self, axis=0)
        acc2 += tl.sum(where_self * centered, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum1_ptr + c, acc1)
    tl.store(sum2_ptr + c, acc2)
    tl.store(vector_out_ptr + c, acc2 * invstd)


@triton.jit
def _full_epilogue_kernel(
    mm_ptr,
    mask_input_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    AVGPOOL_SCALE_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    input_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    pooled_grad = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32) * AVGPOOL_SCALE_
    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    where_self = tl.where(mask_input <= 0.0, 0.0, pooled_grad)

    source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = source - mean
    mean_term = sum1 * BN_SCALE_
    variance_term = sum2 * BN_SCALE_ * invstd * invstd
    affine_term = invstd * affine_weight
    result = (where_self - centered * variance_term - mean_term) * affine_term
    tl.store(out_ptr + input_offsets, result, mask=active)


def oracle_fused(
    mm: torch.Tensor,
    arg102_1: torch.Tensor,
    arg100_1: torch.Tensor,
    arg105_1: torch.Tensor,
    arg101_1: torch.Tensor,
    arg42_1: torch.Tensor,
    _shape_param_0: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mm.shape == (N, C)
    assert arg102_1.shape == (N, C, H, W)
    assert arg100_1.shape == (N, C, H, W)
    assert arg105_1.shape == (1, C, 1, 1)
    assert arg101_1.shape == (C,)
    assert arg42_1.shape == (C,)
    assert mm.dtype == torch.float32
    assert arg102_1.dtype == torch.float32
    assert arg100_1.dtype == torch.float32
    assert arg105_1.dtype == torch.float32
    assert arg101_1.dtype == torch.float32
    assert arg42_1.dtype == torch.float32
    assert mm.stride() == (C, 1)
    assert arg102_1.stride() == (C * HW, HW, W, 1)
    assert arg100_1.stride() == (C * HW, HW, W, 1)
    assert arg105_1.stride() == (C, 1, 1, 1)
    assert arg101_1.stride() == (1,)
    assert arg42_1.stride() == (1,)

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_REDUCE, block_k)
    sum1 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=mm.device, dtype=torch.float32)

    _dual_reduce_channel_kernel[(C,)](
        mm,
        arg102_1,
        arg100_1,
        arg105_1,
        arg101_1,
        sum1,
        sum2,
        vector_out,
        C_=C,
        HW_=HW,
        TOTAL_REDUCE_=TOTAL_REDUCE,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BLOCK_K=block_k,
        NUM_TILES=num_tiles,
        num_warps=4,
    )

    out = torch.empty((N, C, H, W), device=mm.device, dtype=torch.float32)
    block_elems = 512
    _full_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg102_1,
        arg100_1,
        arg105_1,
        arg101_1,
        arg42_1,
        sum1,
        sum2,
        out,
        NUMEL_=NUMEL,
        C_=C,
        HW_=HW,
        AVGPOOL_SCALE_=AVGPOOL_SCALE,
        BN_SCALE_=BN_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, vector_out


def oracle_forward(inputs):
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
