"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete DenseNet batch-norm-backward tail returned by `Repro.forward`, including the `[160]` scale-gradient vector, the three surrounding residual slice adds, and the dependent `[64, 32, 56, 56]` sliced input-gradient side output, whereas Inductor currently schedules the sibling channel reductions, masked BN-backward producer, vector epilogue, and residual-slice side-output epilogue as separate generic reduction and pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates finalized per-channel summaries with a dependent sliced side-output store and live residual slice producers; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced N/H/W domain, combine partial summaries, and fuse the downstream vector and residual-slice epilogues."""
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



REPRO_ID = "sum_sum_62f7c19427cb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 160
H = 56
W = 56
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 128
SLICE_C = 32
SLICE_NUMEL = N * SLICE_C * HW
SCALE = 4.982461734693877e-06



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
    mul_972_ptr,
    mul_990_ptr,
    mul_1008_ptr,
    mask_input_ptr,
    full_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    affine_weight_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_sum_ptr,
    out_slice_ptr,
    C_: tl.constexpr,
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
    out_c_mask = out_c < C_
    out_sum_centered = tl.load(
        sum_where_centered_ptr + out_c,
        mask=out_c_mask,
        other=0.0,
    ).to(tl.float32)
    out_invstd = tl.load(invstd_ptr + out_c, mask=out_c_mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_invstd, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < SLICE_NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % SLICE_C_
    n = linear // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c

    slice_offsets = n * (SLICE_C_ * HW_) + slice_c * HW_ + hw
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw

    acc = tl.load(
        mul_972_ptr + n * (256 * HW_) + c * HW_ + hw,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    acc += tl.load(
        mul_990_ptr + n * (224 * HW_) + c * HW_ + hw,
        mask=active,
        other=0.0,
    ).to(tl.float32)
    acc += tl.load(
        mul_1008_ptr + n * (192 * HW_) + c * HW_ + hw,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    mask_input = tl.load(mask_input_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    source = tl.load(source_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    centered_source = tl.load(centered_source_ptr + bn_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=active, other=0.0).to(
        tl.float32
    )
    full_value = tl.load(full_ptr).to(tl.float32)

    where_self = tl.where(mask_input <= 0.0, full_value, source)
    centered = centered_source - mean
    mean_term = sum_where * SCALE_
    variance_term = sum_where_centered * SCALE_ * invstd * invstd
    grad = (where_self - centered * variance_term - mean_term) * (invstd * affine_weight)

    tl.store(out_slice_ptr + slice_offsets, acc + grad, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mul_972: torch.Tensor,
    mul_990: torch.Tensor,
    mul_1008: torch.Tensor,
    arg268_1: torch.Tensor,
    full: torch.Tensor,
    getitem_336: torch.Tensor,
    arg266_1: torch.Tensor,
    arg724_1: torch.Tensor,
    arg267_1: torch.Tensor,
    arg16_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    assert mul_972.shape == (N, 256, H, W)
    assert mul_990.shape == (N, 224, H, W)
    assert mul_1008.shape == (N, 192, H, W)
    assert arg268_1.shape == (N, C, H, W)
    assert full.shape == ()
    assert getitem_336.shape == (N, C, H, W)
    assert arg266_1.shape == (N, C, H, W)
    assert arg724_1.shape == (1, C, 1, 1)
    assert arg267_1.shape == (C,)
    assert arg16_1.shape == (C,)
    assert mul_972.is_contiguous()
    assert mul_990.is_contiguous()
    assert mul_1008.is_contiguous()
    assert arg268_1.is_contiguous()
    assert getitem_336.is_contiguous()
    assert arg266_1.is_contiguous()
    assert arg724_1.is_contiguous()
    assert arg267_1.is_contiguous()
    assert arg16_1.is_contiguous()

    device = arg268_1.device
    mean = arg724_1.reshape(C)
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 1024
    _masked_bn_dual_reduce_split_k_kernel[
        (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    ](
        arg268_1,
        full,
        getitem_336,
        arg266_1,
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
        mul_972,
        mul_990,
        mul_1008,
        arg268_1,
        full,
        getitem_336,
        arg266_1,
        mean,
        arg267_1,
        arg16_1,
        sum_where,
        sum_where_centered,
        out_sum,
        out_slice,
        C_=C,
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


@oracle_impl(hardware="H100", shapes="(T([64, 256, 56, 56], f32), T([64, 224, 56, 56], f32), T([64, 192, 56, 56], f32), T([64, 160, 56, 56], f32), T([], f32), T([64, 160, 56, 56], f32), T([64, 160, 56, 56], f32), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32))")
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
