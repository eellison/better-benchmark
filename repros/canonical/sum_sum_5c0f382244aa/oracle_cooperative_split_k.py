"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle split-K-reduces the DenseNet batch-norm-backward channel summaries over the shared `(N, H, W)` reduction domain, then uses the finalized summaries in one slice epilogue that writes the full `Repro.forward` return tuple by producing `sum(where_self * centered) * rsqrt` and adding the dependent last-32-channel BN input-gradient slice to the eighteen upstream gradient slices, whereas Inductor currently schedules the sibling channel reductions, the masked BN-backward producer, and the sliced add epilogue as generic reduction and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible BN channel summaries with a dependent sliced side-output epilogue and many same-slice producer inputs; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced N/H/W domain, combine the partial summaries, and fuse the downstream returned-slice epilogue with the live upstream slice additions."""
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



REPRO_ID = "sum_sum_5c0f382244aa"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 448
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 416
SLICE_C = 32
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 7.971938775510203e-05



@triton.jit
def _masked_bn_dual_reduce_split_k_kernel(
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c_mask = c < C_
    k_mask = k < TOTAL_SPATIAL_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]

    mask_input = tl.load(mask_input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask, tl.where(mask_input <= 0.0, full_value, source), 0.0)
    centered = centered_source - mean[:, None]

    tl.atomic_add(
        sum_where_ptr + c,
        tl.sum(where_self, axis=1),
        sem="relaxed",
        mask=c_mask,
    )
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(where_self * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _final_sum_and_slice_epilogue_kernel(
    mul_306_ptr,
    mul_324_ptr,
    mul_342_ptr,
    mul_360_ptr,
    mul_378_ptr,
    mul_396_ptr,
    mul_414_ptr,
    mul_432_ptr,
    mul_450_ptr,
    mul_468_ptr,
    mul_486_ptr,
    mul_504_ptr,
    mul_522_ptr,
    mul_540_ptr,
    mul_558_ptr,
    mul_576_ptr,
    mul_594_ptr,
    mul_612_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_sum_ptr,
    out_slice_ptr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    SLICE_NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    SUM_BLOCK_C: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)

    out_c = pid * SUM_BLOCK_C + tl.arange(0, SUM_BLOCK_C)
    out_c_mask = out_c < 448
    out_sum_centered = tl.load(sum_where_centered_ptr + out_c, mask=out_c_mask, other=0.0)
    out_rsqrt = tl.load(rsqrt_ptr + out_c, mask=out_c_mask, other=0.0)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_rsqrt, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < SLICE_NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % SLICE_C_
    n = linear // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c

    slice_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + hw
    offsets_448 = n * (448 * HW_) + c * HW_ + hw

    acc = tl.load(mul_306_ptr + n * (1024 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_324_ptr + n * (992 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_342_ptr + n * (960 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_360_ptr + n * (928 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_378_ptr + n * (896 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_396_ptr + n * (864 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_414_ptr + n * (832 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_432_ptr + n * (800 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_450_ptr + n * (768 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_468_ptr + n * (736 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_486_ptr + n * (704 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_504_ptr + n * (672 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_522_ptr + n * (640 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_540_ptr + n * (608 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_558_ptr + n * (576 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_576_ptr + n * (544 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_594_ptr + n * (512 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)
    acc += tl.load(mul_612_ptr + n * (480 * HW_) + c * HW_ + hw, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + offsets_448, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + offsets_448, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + offsets_448, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * rsqrt * rsqrt
    grad = (where_self - centered * variance_term - mean_term) * (rsqrt * affine_weight)

    tl.store(out_slice_ptr + slice_offsets, acc + grad, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
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
    mul_576: torch.Tensor,
    mul_594: torch.Tensor,
    mul_612: torch.Tensor,
    arg402_1: torch.Tensor,
    full: torch.Tensor,
    getitem_204: torch.Tensor,
    arg400_1: torch.Tensor,
    arg680_1: torch.Tensor,
    arg401_1: torch.Tensor,
    arg104_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert mul_306.shape == (N, 1024, H, W)
    assert mul_324.shape == (N, 992, H, W)
    assert mul_342.shape == (N, 960, H, W)
    assert mul_360.shape == (N, 928, H, W)
    assert mul_378.shape == (N, 896, H, W)
    assert mul_396.shape == (N, 864, H, W)
    assert mul_414.shape == (N, 832, H, W)
    assert mul_432.shape == (N, 800, H, W)
    assert mul_450.shape == (N, 768, H, W)
    assert mul_468.shape == (N, 736, H, W)
    assert mul_486.shape == (N, 704, H, W)
    assert mul_504.shape == (N, 672, H, W)
    assert mul_522.shape == (N, 640, H, W)
    assert mul_540.shape == (N, 608, H, W)
    assert mul_558.shape == (N, 576, H, W)
    assert mul_576.shape == (N, 544, H, W)
    assert mul_594.shape == (N, 512, H, W)
    assert mul_612.shape == (N, 480, H, W)
    assert arg402_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_204.shape == (N, C, H, W)
    assert arg400_1.shape == (N, C, H, W)
    assert arg680_1.shape == (1, C, 1, 1)
    assert arg401_1.shape == (C,)
    assert arg104_1.shape == (C,)

    device = arg402_1.device
    mean = arg680_1.reshape(C)
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 512
    _masked_bn_dual_reduce_split_k_kernel[
        (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    ](
        arg402_1,
        full,
        getitem_204,
        arg400_1,
        mean,
        sum_where,
        sum_where_centered,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )

    out_sum = torch.empty((C,), device=device, dtype=torch.float32)
    out_slice = torch.empty((N, SLICE_C, H, W), device=device, dtype=torch.float32)

    sum_block_c = 4
    block_elems = 256
    epilogue_grid = (
        max(triton.cdiv(C, sum_block_c), triton.cdiv(SLICE_NUMEL, block_elems)),
    )
    _final_sum_and_slice_epilogue_kernel[epilogue_grid](
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
        mul_576,
        mul_594,
        mul_612,
        arg402_1,
        full,
        getitem_204,
        arg400_1,
        mean,
        arg401_1,
        arg104_1,
        sum_where,
        sum_where_centered,
        out_sum,
        out_slice,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SLICE_NUMEL_=SLICE_NUMEL,
        SCALE_=SCALE,
        SUM_BLOCK_C=sum_block_c,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out_sum, out_slice


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


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
