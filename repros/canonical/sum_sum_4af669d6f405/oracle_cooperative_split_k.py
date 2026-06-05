"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete ShuffleNet channel-shuffle plus batch-norm-backward return tuple from Repro.forward by reconstructing the shuffled high-half slice from the original cat/view/permute/clone producer, cooperatively split-K reducing both per-channel summaries over the shared N/H/W domain, and using the finalized summaries in one epilogue that writes the full input-gradient tensor plus scale-gradient side output, whereas Inductor currently schedules the channel shuffle, ReLU-gated producer, sibling sums, and dependent BN-backward epilogues as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent full-tensor side-output epilogue fed by a layout-changing producer; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN channel reductions across the reduced N/H/W dimension, combine partial summaries once, and fuse the channel-shuffle producer plus downstream tensor/vector epilogues."""
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



REPRO_ID = "sum_sum_4af669d6f405"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 512
C_IN = 116
C = 58
H = 28
W = 28
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
HIGH_HALF = 58
SCALE = 2.4912308673469386e-06
REDUCE_BLOCK_K = 2048
EPILOGUE_BLOCK_ELEMS = 1024



@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _add_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _load_shuffled_high_half(
    view_28_ptr,
    getitem_138_ptr,
    n,
    c,
    hw,
    active,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
):
    # After cat([view_28[:, :58], getitem_138], 1).view(N, 58, 2, H, W)
    # and permute(0, 2, 1, 3, 4), output channel 58 + c reads cat channel
    # 2*c + 1. Channels below 58 come from view_28; the rest from getitem_138.
    cat_c = c * 2 + 1
    from_view = cat_c < C_
    view_offset = n * (C_IN_ * HW_) + cat_c * HW_ + hw
    get_c = tl.where(from_view, 0, cat_c - C_)
    getitem_offset = n * (C_ * HW_) + get_c * HW_ + hw

    view_value = tl.load(
        view_28_ptr + view_offset,
        mask=active & from_view,
        other=0.0,
    ).to(tl.float32)
    getitem_value = tl.load(
        getitem_138_ptr + getitem_offset,
        mask=active & ~from_view,
        other=0.0,
    ).to(tl.float32)
    return tl.where(from_view, view_value, getitem_value)


@triton.jit
def _shuffled_bn_dual_reduce_kernel(
    view_28_ptr,
    getitem_138_ptr,
    arg162_ptr,
    arg163_ptr,
    arg164_ptr,
    arg21_ptr,
    arg22_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    bn_offsets = n * (C_ * HW_) + c * HW_ + hw

    source = _load_shuffled_high_half(
        view_28_ptr,
        getitem_138_ptr,
        n,
        c,
        hw,
        active,
        C_IN_,
        C_,
        HW_,
    )
    bn_input = tl.load(arg162_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg163_ptr + c).to(tl.float32)
    invstd = tl.load(arg164_ptr + c).to(tl.float32)
    affine_weight = tl.load(arg21_ptr + c).to(tl.float32)
    affine_bias = tl.load(arg22_ptr + c).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)

    centered = bn_input - mean
    normalized = _mul_rn_f32(centered, invstd)
    relu_input = _add_rn_f32(_mul_rn_f32(normalized, affine_weight), affine_bias)
    where_self = tl.where(active, tl.where(relu_input <= 0.0, full_value, source), 0.0)

    tl.atomic_add(sum_where_ptr + c, tl.sum(where_self, axis=0), sem="relaxed")
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(_mul_rn_f32(where_self, centered), axis=0),
        sem="relaxed",
    )


@triton.jit
def _bn_backward_epilogue_kernel(
    view_28_ptr,
    getitem_138_ptr,
    arg162_ptr,
    arg163_ptr,
    arg164_ptr,
    arg21_ptr,
    arg22_ptr,
    full_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_grad_ptr,
    out_scale_grad_ptr,
    C_IN_: tl.constexpr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_

    hw = linear % HW_
    c = (linear // HW_) % C_
    n = linear // (C_ * HW_)
    bn_offsets = linear

    source = _load_shuffled_high_half(
        view_28_ptr,
        getitem_138_ptr,
        n,
        c,
        hw,
        active,
        C_IN_,
        C_,
        HW_,
    )
    bn_input = tl.load(arg162_ptr + bn_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg163_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(arg164_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_weight = tl.load(arg21_ptr + c, mask=active, other=0.0).to(tl.float32)
    affine_bias = tl.load(arg22_ptr + c, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(full_ptr).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(
        sum_where_centered_ptr + c,
        mask=active,
        other=0.0,
    ).to(tl.float32)

    centered = bn_input - mean
    normalized = _mul_rn_f32(centered, invstd)
    relu_input = _add_rn_f32(_mul_rn_f32(normalized, affine_weight), affine_bias)
    where_self = tl.where(relu_input <= 0.0, full_value, source)

    mean_term = _mul_rn_f32(sum_where, SCALE_)
    scaled_centered_sum = _mul_rn_f32(sum_where_centered, SCALE_)
    invstd_sq = _mul_rn_f32(invstd, invstd)
    variance_term = _mul_rn_f32(scaled_centered_sum, invstd_sq)
    affine_term = _mul_rn_f32(invstd, affine_weight)

    centered_variance = _mul_rn_f32(centered, variance_term)
    sub_mean = _add_rn_f32(_add_rn_f32(where_self, -centered_variance), -mean_term)
    out_grad = _mul_rn_f32(sub_mean, affine_term)
    tl.store(out_grad_ptr + linear, out_grad, mask=active)

    tl.store(
        out_scale_grad_ptr + c,
        _mul_rn_f32(sum_where_centered, invstd),
        mask=active & (n == 0) & (hw == 0),
    )


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_full(
    view_28: torch.Tensor,
    getitem_138: torch.Tensor,
    arg162_1: torch.Tensor,
    arg163_1: torch.Tensor,
    arg164_1: torch.Tensor,
    arg21_1: torch.Tensor,
    arg22_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert _shape_param_0 == [N, C, 2, H, W]
    assert _shape_param_1 == [N, C_IN, H, W]
    assert view_28.shape == (N, C_IN, H, W)
    assert getitem_138.shape == (N, C, H, W)
    assert arg162_1.shape == (N, C, H, W)
    assert arg163_1.shape == (1, C, 1, 1)
    assert arg164_1.shape == (1, C, 1, 1)
    assert arg21_1.shape == (C,)
    assert arg22_1.shape == (C,)
    assert full.shape == ()
    assert view_28.is_contiguous()
    assert getitem_138.is_contiguous()
    assert arg162_1.is_contiguous()
    assert arg163_1.is_contiguous()
    assert arg164_1.is_contiguous()
    assert arg21_1.is_contiguous()
    assert arg22_1.is_contiguous()
    assert full.is_contiguous()

    device = arg162_1.device
    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    _shuffled_bn_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, REDUCE_BLOCK_K))](
        view_28,
        getitem_138,
        arg162_1,
        arg163_1,
        arg164_1,
        arg21_1,
        arg22_1,
        full,
        sum_where,
        sum_where_centered,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=8,
    )

    out_grad = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    out_scale_grad = torch.empty((C,), device=device, dtype=torch.float32)

    _bn_backward_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK_ELEMS),)](
        view_28,
        getitem_138,
        arg162_1,
        arg163_1,
        arg164_1,
        arg21_1,
        arg22_1,
        full,
        sum_where,
        sum_where_centered,
        out_grad,
        out_scale_grad,
        C_IN_=C_IN,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        SCALE_=SCALE,
        BLOCK_ELEMS=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    return out_grad, out_scale_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def _bench_us(fn, warmup: int, rep: int) -> float:
    return (
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="min",
        )
        * 1000.0
    )


def oracle_forward(inputs):
    return oracle_full(*inputs)


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
