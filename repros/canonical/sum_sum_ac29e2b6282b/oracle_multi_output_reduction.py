"""
Full-scope oracle for sum_sum_ac29e2b6282b (DenseNet BN-backward tail).

Gap diagnosis (classification: SCHEDULER_FUSION): The timed oracle consumes the
same 26 original inputs as repro.py and returns the same `[416]` vector plus
contiguous `[64, 32, 14, 14]` residual-add slice. It differs from Inductor by
sharing the ReLU-mask `where`, channel centering, and two sibling channel
reductions across all C=416 BN channels, then using the finalized reductions in
the dependent BN-backward epilogue for only the live channels 384:416 while
adding all nineteen upstream residual slices directly into the returned output.
Inductor cannot do this today because the scheduler keeps the residual slice
chain, sibling reductions, dependent epilogue, and final slice/add as ordinary
scheduled nodes instead of forming one full-scope multi-output reduction
template that shares producers and sinks the side-output epilogue to the live
slice. The fix is SCHEDULER_FUSION: teach scheduler/codegen to fuse compatible
sibling reductions with a dependent, slice-limited epilogue and keep the
residual slice chain inside that fused epilogue.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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


REPRO_ID = "sum_sum_ac29e2b6282b"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 416
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 384
SLICE_C = C - SLICE_START
SCALE = 7.971938775510203e-05
HISTORICAL_BEST_US = 31.136000528931618

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning=True", {"coordinate_descent_tuning": True}),
    (
        (
            "combo_kernels=True,combo_kernel_per_subkernel_blocks=True,"
            "coordinate_descent_tuning=True,benchmark_combo_kernel=True,triton.multi_kernel=3"
        ),
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def _dual_reduce_partial_kernel(
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
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
    tl.store(partial_sum1_ptr + partial_offset, tl.sum(where_self, axis=0))
    tl.store(partial_sum2_ptr + partial_offset, tl.sum(where_self * centered, axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    scale_ptr,
    sum1_ptr,
    sum2_ptr,
    vector_out_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    tile_mask = tiles < num_tiles
    partial_offsets = c * num_tiles + tiles

    sum1_values = tl.load(partial_sum1_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum2_values = tl.load(partial_sum2_ptr + partial_offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum1 = tl.sum(sum1_values, axis=0)
    sum2 = tl.sum(sum2_values, axis=0)

    scale_value = tl.load(scale_ptr + c).to(tl.float32)
    tl.store(sum1_ptr + c, sum1)
    tl.store(sum2_ptr + c, sum2)
    tl.store(vector_out_ptr + c, sum2 * scale_value)


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
    residual11_ptr,
    residual12_ptr,
    residual13_ptr,
    residual14_ptr,
    residual15_ptr,
    residual16_ptr,
    residual17_ptr,
    residual18_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    scale_ptr,
    affine_weight_ptr,
    sum1_ptr,
    sum2_ptr,
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
    residual_offsets15 = n * (544 * HW_) + c * HW_ + hw
    residual_offsets16 = n * (512 * HW_) + c * HW_ + hw
    residual_offsets17 = n * (480 * HW_) + c * HW_ + hw
    residual_offsets18 = n * (448 * HW_) + c * HW_ + hw

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
    residual += tl.load(residual11_ptr + residual_offsets11, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual12_ptr + residual_offsets12, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual13_ptr + residual_offsets13, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual14_ptr + residual_offsets14, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual15_ptr + residual_offsets15, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual16_ptr + residual_offsets16, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual17_ptr + residual_offsets17, mask=active, other=0.0).to(tl.float32)
    residual += tl.load(residual18_ptr + residual_offsets18, mask=active, other=0.0).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    source = tl.load(source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    scale_value = tl.load(scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=active, other=0.0).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * scale_value * scale_value
    affine_term = scale_value * affine_weight
    epilogue = (where_self - centered * variance_term - mean_term) * affine_term
    tl.store(add_out_ptr + out_offsets, residual + epilogue, mask=active)


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    mul_630: torch.Tensor,
    arg396_1: torch.Tensor,
    full: torch.Tensor,
    getitem_210: torch.Tensor,
    arg394_1: torch.Tensor,
    arg682_1: torch.Tensor,
    arg395_1: torch.Tensor,
    arg100_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
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
        mul_576,
        mul_594,
        mul_612,
        mul_630,
    )
    expected_residual_channels = (
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
        544,
        512,
        480,
        448,
    )
    for residual, residual_c in zip(residuals, expected_residual_channels):
        assert residual.shape == (N, residual_c, H, W)
        assert residual.is_contiguous()
    assert arg396_1.shape == (N, C, H, W)
    assert getitem_210.shape == (N, C, H, W)
    assert arg394_1.shape == (N, C, H, W)
    assert arg682_1.shape == (1, C, 1, 1)
    assert arg395_1.shape == (C,)
    assert arg100_1.shape == (C,)
    assert arg396_1.is_contiguous()
    assert getitem_210.is_contiguous()
    assert arg394_1.is_contiguous()

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum1 = torch.empty((C, num_tiles), device=arg396_1.device, dtype=torch.float32)
    partial_sum2 = torch.empty((C, num_tiles), device=arg396_1.device, dtype=torch.float32)
    sum1 = torch.empty((C,), device=arg396_1.device, dtype=torch.float32)
    sum2 = torch.empty((C,), device=arg396_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg396_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg396_1,
        full,
        getitem_210,
        arg394_1,
        arg682_1,
        partial_sum1,
        partial_sum2,
        num_tiles=num_tiles,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=block_k,
        num_warps=4,
    )

    block_tiles = triton.next_power_of_2(num_tiles)
    _finalize_reduce_kernel[(C,)](
        partial_sum1,
        partial_sum2,
        arg395_1,
        sum1,
        sum2,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg396_1.device, dtype=torch.float32)
    block_elems = 512
    numel_slice = N * SLICE_C * HW
    _slice_add_epilogue_kernel[(triton.cdiv(numel_slice, block_elems),)](
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
        mul_630,
        arg396_1,
        full,
        getitem_210,
        arg394_1,
        arg682_1,
        arg395_1,
        arg100_1,
        sum1,
        sum2,
        add_out,
        NUMEL_SLICE_=numel_slice,
        C_=C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=8,
    )

    return vector_out, add_out


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([64, 480, 14, 14], f32), T([64, 448, 14, 14], f32), T([64, 416, 14, 14], f32), T([], f32), T([64, 416, 14, 14], f32), T([64, 416, 14, 14], f32), T([1, 416, 1, 1], f32), T([416], f32), T([416], f32))")
def oracle_forward(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    return oracle_fused(*inputs)


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return model(*inputs)


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
        status = "PASS" if ok else "FAIL"
        print(f"Correctness: {status}")
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
