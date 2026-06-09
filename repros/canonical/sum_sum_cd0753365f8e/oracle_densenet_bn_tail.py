"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet batch-norm-backward tail from `Repro.forward` by sharing the masked `where` producer across two per-channel reductions, finalizing both channel summaries once, and using them in the returned channels 480:512 epilogue while adding all 16 live residual slices; whereas Inductor currently schedules the residual slice-add chain, sibling reductions, dependent BN-backward epilogue, and final slice/add as ordinary regions around materialized intermediates. Inductor cannot do this today because its scheduler/codegen does not form one full-scope multi-output reduction template that shares the masked/centered producer and sinks the dependent side output to only the live channel slice. The fix is SCHEDULER_FUSION: teach the scheduler to fuse compatible sibling reductions with a dependent, slice-limited epilogue and keep the residual slice-add chain inside that fused plan."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 512
H = 14
W = 14
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 480
SLICE_C = C - SLICE_START
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 7.971938775510203e-05
RESIDUAL_CHANNELS = (
    1024, 992, 960, 928, 896, 864, 832, 800,
    768, 736, 704, 672, 640, 608, 576, 544,
)


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
        residual11_ptr,
        residual12_ptr,
        residual13_ptr,
        residual14_ptr,
        residual15_ptr,
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


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    if len(inputs) != 23:
        raise ValueError(f"expected 23 inputs, got {len(inputs)}")

    residuals = inputs[:16]
    for idx, (residual, residual_c) in enumerate(zip(residuals, RESIDUAL_CHANNELS)):
        if not isinstance(residual, torch.Tensor):
            raise TypeError(f"residual input {idx} must be a tensor")
        if tuple(residual.shape) != (N, residual_c, H, W):
            raise ValueError(
                f"residual input {idx} expected shape={(N, residual_c, H, W)}, "
                f"got {tuple(residual.shape)}"
            )
        if residual.dtype != torch.float32 or residual.device.type != "cuda":
            raise ValueError(f"residual input {idx} must be CUDA float32")
        if not residual.is_contiguous():
            raise ValueError(f"residual input {idx} must be contiguous")

    arg414_1, full, getitem_192, arg412_1, arg676_1, arg413_1, arg112_1 = inputs[16:]
    expected = {
        "arg414_1": (arg414_1, (N, C, H, W)),
        "full": (full, ()),
        "getitem_192": (getitem_192, (N, C, H, W)),
        "arg412_1": (arg412_1, (N, C, H, W)),
        "arg676_1": (arg676_1, (1, C, 1, 1)),
        "arg413_1": (arg413_1, (C,)),
        "arg112_1": (arg112_1, (C,)),
    }
    for name, (tensor, shape) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} expected shape={shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != torch.float32 or tensor.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA float32")

    for name, tensor in (
        ("arg414_1", arg414_1),
        ("getitem_192", getitem_192),
        ("arg412_1", arg412_1),
        ("arg676_1", arg676_1),
        ("arg413_1", arg413_1),
        ("arg112_1", arg112_1),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


@oracle_impl(hardware="H100", shapes="(T([64, 1024, 14, 14], f32), T([64, 992, 14, 14], f32), T([64, 960, 14, 14], f32), T([64, 928, 14, 14], f32), T([64, 896, 14, 14], f32), T([64, 864, 14, 14], f32), T([64, 832, 14, 14], f32), T([64, 800, 14, 14], f32), T([64, 768, 14, 14], f32), T([64, 736, 14, 14], f32), T([64, 704, 14, 14], f32), T([64, 672, 14, 14], f32), T([64, 640, 14, 14], f32), T([64, 608, 14, 14], f32), T([64, 576, 14, 14], f32), T([64, 544, 14, 14], f32), T([64, 512, 14, 14], f32), T([], f32), T([64, 512, 14, 14], f32), T([64, 512, 14, 14], f32), T([1, 512, 1, 1], f32), T([512], f32), T([512], f32))")
def oracle_forward(inputs):
    """Run the full-scope oracle for `Repro()(*make_inputs())`."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_densenet_bn_tail.py")

    inputs = tuple(inputs)
    _validate_inputs(inputs)

    (
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
        arg414_1,
        full,
        getitem_192,
        arg412_1,
        arg676_1,
        arg413_1,
        arg112_1,
    ) = inputs

    block_k = 1024
    num_tiles = triton.cdiv(TOTAL_SPATIAL, block_k)
    partial_sum_where = torch.empty((C, num_tiles), device=arg414_1.device, dtype=torch.float32)
    partial_sum_centered = torch.empty((C, num_tiles), device=arg414_1.device, dtype=torch.float32)
    sum_where = torch.empty((C,), device=arg414_1.device, dtype=torch.float32)
    sum_centered = torch.empty((C,), device=arg414_1.device, dtype=torch.float32)
    vector_out = torch.empty((C,), device=arg414_1.device, dtype=torch.float32)

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg414_1,
        full,
        getitem_192,
        arg412_1,
        arg676_1,
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
        arg413_1,
        sum_where,
        sum_centered,
        vector_out,
        num_tiles=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
    )

    add_out = torch.empty((N, SLICE_C, H, W), device=arg414_1.device, dtype=torch.float32)
    block_elems = 1024
    _slice_add_epilogue_kernel[(triton.cdiv(SLICE_NUMEL, block_elems),)](
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
        arg414_1,
        full,
        getitem_192,
        arg412_1,
        arg676_1,
        arg413_1,
        arg112_1,
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
