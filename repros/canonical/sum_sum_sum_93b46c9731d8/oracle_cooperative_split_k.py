"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full RegNet/ResNeSt dual-branch ReLU plus batch-norm-backward return tuple by cooperatively split-K reducing the shared masked pool-gradient producer into one per-channel mask sum and two centered channel sums, then using those finalized summaries in a fused epilogue that writes both returned `[32, 2240, 7, 7]` gradients and `[2240]` scale-gradient vectors, whereas Inductor currently schedules the decomposed pointwise mask, sibling `sum([0, 2, 3])` reductions, and two BN-backward epilogues as ordinary reductions/pointwise kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible sibling channel reductions and dependent full-tensor epilogues across two BN branches; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the `(N, H, W)` domain, combine the partials, and fuse the downstream branch epilogues plus side-vector outputs."""
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



REPRO_ID = "sum_sum_sum_93b46c9731d8"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 2240
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL



def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


@triton.jit
def _dual_bn_reduce_split_k_kernel(
    mm_ptr,
    x1_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    mean2_ptr,
    rsqrt2_ptr,
    weight2_ptr,
    bias2_ptr,
    sum_mask_ptr,
    sum_centered1_ptr,
    sum_centered2_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
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
    x_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mm_offsets = n[None, :] * C_ + c[:, None]

    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

    mean1 = tl.load(mean1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt1 = tl.load(rsqrt1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    mean2 = tl.load(mean2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt2 = tl.load(rsqrt2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered1 = x1 - mean1[:, None]
    centered2 = x2 - mean2[:, None]
    affine1 = centered1 * rsqrt1[:, None] * weight1[:, None] + bias1[:, None]
    affine2 = centered2 * rsqrt2[:, None] * weight2[:, None] + bias2[:, None]
    masked_grad = tl.where((affine1 + affine2) <= 0.0, 0.0, mm * INV_HW_)

    tl.store(sum_mask_ptr + c, tl.sum(masked_grad, axis=1), mask=c_mask)
    tl.store(sum_centered1_ptr + c, tl.sum(masked_grad * centered1, axis=1), mask=c_mask)
    tl.store(sum_centered2_ptr + c, tl.sum(masked_grad * centered2, axis=1), mask=c_mask)


@triton.jit
def _dual_bn_epilogue_kernel(
    mm_ptr,
    x1_ptr,
    mean1_ptr,
    rsqrt1_ptr,
    weight1_ptr,
    bias1_ptr,
    x2_ptr,
    mean2_ptr,
    rsqrt2_ptr,
    weight2_ptr,
    bias2_ptr,
    sum_mask_ptr,
    sum_centered1_ptr,
    sum_centered2_ptr,
    grad2_ptr,
    weight_grad2_ptr,
    grad1_ptr,
    weight_grad1_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = offsets < NUMEL_

    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)
    mm_offsets = n * C_ + c

    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    mean1 = tl.load(mean1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt1 = tl.load(rsqrt1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight1 = tl.load(weight1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias1 = tl.load(bias1_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean2 = tl.load(mean2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt2 = tl.load(rsqrt2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight2 = tl.load(weight2_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias2 = tl.load(bias2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    sum_mask = tl.load(sum_mask_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_centered1 = tl.load(sum_centered1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_centered2 = tl.load(sum_centered2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered1 = x1 - mean1
    centered2 = x2 - mean2
    affine1 = centered1 * rsqrt1 * weight1 + bias1
    affine2 = centered2 * rsqrt2 * weight2 + bias2
    masked_grad = tl.where((affine1 + affine2) <= 0.0, 0.0, mm * INV_HW_)

    mean_term = sum_mask * SCALE_
    variance_term1 = sum_centered1 * SCALE_ * rsqrt1 * rsqrt1
    variance_term2 = sum_centered2 * SCALE_ * rsqrt2 * rsqrt2

    grad1 = (masked_grad - centered1 * variance_term1 - mean_term) * (rsqrt1 * weight1)
    grad2 = (masked_grad - centered2 * variance_term2 - mean_term) * (rsqrt2 * weight2)

    tl.store(grad2_ptr + offsets, grad2, mask=mask)
    tl.store(grad1_ptr + offsets, grad1, mask=mask)

    first_element_for_channel = mask & (n == 0) & ((offsets % HW_) == 0)
    tl.store(weight_grad2_ptr + c, sum_centered2 * rsqrt2, mask=first_element_for_channel)
    tl.store(weight_grad1_ptr + c, sum_centered1 * rsqrt1, mask=first_element_for_channel)


def oracle_full(
    mm: torch.Tensor,
    arg438_1: torch.Tensor,
    arg439_1: torch.Tensor,
    arg440_1: torch.Tensor,
    arg179_1: torch.Tensor,
    arg180_1: torch.Tensor,
    arg441_1: torch.Tensor,
    arg442_1: torch.Tensor,
    arg443_1: torch.Tensor,
    arg182_1: torch.Tensor,
    arg183_1: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    assert mm.shape == (N, C)
    assert arg438_1.shape == (N, C, H, W)
    assert arg439_1.shape == (1, C, 1, 1)
    assert arg440_1.shape == (1, C, 1, 1)
    assert arg179_1.shape == (C,)
    assert arg180_1.shape == (C,)
    assert arg441_1.shape == (N, C, H, W)
    assert arg442_1.shape == (1, C, 1, 1)
    assert arg443_1.shape == (1, C, 1, 1)
    assert arg182_1.shape == (C,)
    assert arg183_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg438_1.is_contiguous()
    assert arg439_1.is_contiguous()
    assert arg440_1.is_contiguous()
    assert arg179_1.is_contiguous()
    assert arg180_1.is_contiguous()
    assert arg441_1.is_contiguous()
    assert arg442_1.is_contiguous()
    assert arg443_1.is_contiguous()
    assert arg182_1.is_contiguous()
    assert arg183_1.is_contiguous()

    device = arg438_1.device
    sum_mask = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered2 = torch.empty((C,), device=device, dtype=torch.float32)

    block_c = 2
    block_k = 2048
    reduce_grid = (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    _dual_bn_reduce_split_k_kernel[reduce_grid](
        mm,
        arg438_1,
        arg439_1,
        arg440_1,
        arg179_1,
        arg180_1,
        arg441_1,
        arg442_1,
        arg443_1,
        arg182_1,
        arg183_1,
        sum_mask,
        sum_centered1,
        sum_centered2,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )

    grad2 = torch.empty_like(arg441_1)
    weight_grad2 = torch.empty((C,), device=device, dtype=torch.float32)
    grad1 = torch.empty_like(arg438_1)
    weight_grad1 = torch.empty((C,), device=device, dtype=torch.float32)

    block_elems = 512
    _dual_bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg438_1,
        arg439_1,
        arg440_1,
        arg179_1,
        arg180_1,
        arg441_1,
        arg442_1,
        arg443_1,
        arg182_1,
        arg183_1,
        sum_mask,
        sum_centered1,
        sum_centered2,
        grad2,
        weight_grad2,
        grad1,
        weight_grad1,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return grad2, weight_grad2, grad1, weight_grad1


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@oracle_impl(hardware="H100", shapes="(T([32, 2240], f32), T([32, 2240, 7, 7], f32), T([1, 2240, 1, 1], f32), T([1, 2240, 1, 1], f32), T([2240], f32), T([2240], f32), T([32, 2240, 7, 7], f32), T([1, 2240, 1, 1], f32), T([1, 2240, 1, 1], f32), T([2240], f32), T([2240], f32), S([32, 2240, 1, 1]), S([32, 2240, 7, 7]))")
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
