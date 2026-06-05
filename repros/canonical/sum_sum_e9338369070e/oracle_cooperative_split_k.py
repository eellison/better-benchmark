"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet BN-backward-style return from Repro.forward by reading the sliced add producer directly from the two original `[512, 80, 14, 14]` inputs, cooperatively split-K reducing the two sibling per-channel sums, and fusing the dependent affine epilogue into the final channels-last `[512, 40, 14, 14]` output plus `[40]` side vector, whereas Inductor currently lowers the add/copy/clone/slice, compatible `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as generic scheduled reduction and pointwise work with avoidable intermediate materialization and insufficient reduction-axis parallelism; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template that can preserve a sliced channels-last producer and feed a dependent full-tensor epilogue with the required output layout; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible per-channel reductions across the reduced N/H/W dimension, combine the partials, and fuse the sliced-add producer plus BN-backward epilogue and side vector output into the same reduction plan."""
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


REPRO_ID = "sum_sum_e9338369070e"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
IN_C = 80
C = 40
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
REDUCE_SCALE = 9.964923469387754e-06


@triton.jit
def _partial_dual_sum_kernel(
    clone_ptr,
    getitem_ptr,
    centered_ptr,
    mean_ptr,
    partial_x_ptr,
    partial_x_centered_ptr,
    num_tiles: tl.constexpr,
    IN_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
    centered_offsets = n * (C_ * HW_) + c * HW_ + hw

    x0 = tl.load(clone_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    x = x0 + x1
    centered = tl.load(centered_ptr + centered_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = centered - mean

    partial_offset = c * num_tiles + tile
    tl.store(partial_x_ptr + partial_offset, tl.sum(x, axis=0))
    tl.store(partial_x_centered_ptr + partial_offset, tl.sum(x * centered, axis=0))


@triton.jit
def _finalize_dual_sum_kernel(
    partial_x_ptr,
    partial_x_centered_ptr,
    invstd_ptr,
    sum_x_ptr,
    sum_x_centered_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < num_tiles
    offsets = c * num_tiles + tiles

    x_values = tl.load(partial_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    xc_values = tl.load(partial_x_centered_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.sum(x_values, axis=0)
    sum_x_centered = tl.sum(xc_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_x_ptr + c, sum_x)
    tl.store(sum_x_centered_ptr + c, sum_x_centered)
    tl.store(vector_out_ptr + c, sum_x_centered * invstd)


@triton.jit
def _bn_epilogue_kernel(
    clone_ptr,
    getitem_ptr,
    centered_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_x_ptr,
    sum_x_centered_ptr,
    out_ptr,
    IN_C_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    W_: tl.constexpr,
    NUMEL_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = linear < NUMEL_

    hw = linear % HW_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)
    h = hw // W_
    w = hw - h * W_

    source_offsets = n * (IN_C_ * HW_) + (c + C_) * HW_ + hw
    centered_offsets = n * (C_ * HW_) + c * HW_ + hw
    out_offsets = n * (C_ * HW_) + h * (W_ * C_) + w * C_ + c

    x0 = tl.load(clone_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(getitem_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
    x = x0 + x1
    centered = tl.load(centered_ptr + centered_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_x_centered = tl.load(sum_x_centered_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = centered - mean
    mean_term = sum_x * REDUCE_SCALE_
    variance_term = sum_x_centered * REDUCE_SCALE_ * invstd * invstd
    out = (x - centered * variance_term - mean_term) * (invstd * affine_weight)
    tl.store(out_ptr + out_offsets, out, mask=mask)


def oracle_full(
    clone_8: torch.Tensor,
    getitem_156: torch.Tensor,
    arg319_1: torch.Tensor,
    arg510_1: torch.Tensor,
    arg320_1: torch.Tensor,
    arg88_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert clone_8.shape == (N, IN_C, H, W)
    assert getitem_156.shape == (N, IN_C, H, W)
    assert arg319_1.shape == (N, C, H, W)
    assert arg510_1.shape == (1, C, 1, 1)
    assert arg320_1.shape == (C,)
    assert arg88_1.shape == (C,)
    assert clone_8.is_contiguous()
    assert getitem_156.is_contiguous()
    assert arg319_1.is_contiguous()
    assert arg510_1.is_contiguous()
    assert arg320_1.is_contiguous()
    assert arg88_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_x = torch.empty((C, num_tiles), device=clone_8.device, dtype=torch.float32)
    partial_x_centered = torch.empty((C, num_tiles), device=clone_8.device, dtype=torch.float32)
    sum_x = torch.empty((C,), device=clone_8.device, dtype=torch.float32)
    sum_x_centered = torch.empty((C,), device=clone_8.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=clone_8.device, dtype=torch.float32)

    _partial_dual_sum_kernel[(C, num_tiles)](
        clone_8,
        getitem_156,
        arg319_1,
        arg510_1,
        partial_x,
        partial_x_centered,
        num_tiles=num_tiles,
        IN_C_=IN_C,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_dual_sum_kernel[(C,)](
        partial_x,
        partial_x_centered,
        arg320_1,
        sum_x,
        sum_x_centered,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=clone_8.device,
        dtype=torch.float32,
    )
    block_elems = 256
    _bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        clone_8,
        getitem_156,
        arg319_1,
        arg510_1,
        arg320_1,
        arg88_1,
        sum_x,
        sum_x_centered,
        out,
        IN_C_=IN_C,
        C_=C,
        HW_=HW,
        W_=W,
        NUMEL_=NUMEL,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, vector_out


def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
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
