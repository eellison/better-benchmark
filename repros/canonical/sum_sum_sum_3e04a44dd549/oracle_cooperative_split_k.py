"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full GhostNet two-branch batch-norm-backward return tuple by reducing the shared 160-channel add/copy/clone producer and its high-channel 80-channel slice in one split-K Triton pass, then using the finalized channel summaries to write the contiguous full-gradient tensor, the channels-last slice-gradient tensor, and both scale-gradient vectors, whereas Inductor currently schedules the memory-format copy/clone/slice producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogues as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that shares a memory-format-aware producer across multiple channel reductions and dependent full-tensor side-output stores; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile the `(N, H, W)` reduction domain for compatible channel reductions, accumulate the full and sliced branch summaries together, and fuse the downstream tensor/vector epilogues with layout-aware stores."""
from __future__ import annotations

import argparse
import importlib.util
import math
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



REPRO_ID = "sum_sum_sum_3e04a44dd549"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_ghostnet_100_train_001_4577889e"

N = 512
C = 160
C_SLICE = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL_FULL = N * C * HW
NUMEL_SLICE = N * C_SLICE * HW
SCALE = 3.985969387755102e-05

FULL_OUT_STRIDE = (C * HW, HW, W, 1)
SLICE_OUT_STRIDE = (C_SLICE * HW, 1, W * C_SLICE, C_SLICE)

REDUCE_BLOCK_C = 2
REDUCE_BLOCK_K = 1024
EPILOGUE_BLOCK_ELEMS = 1024



def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(
        value.cuda() if isinstance(value, torch.Tensor) else value
        for value in module.make_inputs()
    )


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@triton.jit
def _two_branch_reduce_split_k_kernel(
    clone_ptr,
    getitem_ptr,
    rhs_full_ptr,
    mean_full_ptr,
    rhs_slice_ptr,
    mean_slice_ptr,
    sum_full_ptr,
    sum_full_rhs_ptr,
    sum_slice_ptr,
    sum_slice_rhs_ptr,
    C_: tl.constexpr,
    C_SLICE_: tl.constexpr,
    SLICE_START_: tl.constexpr,
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
    full_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]

    x = (
        tl.load(clone_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
    )
    rhs_full = (
        tl.load(rhs_full_ptr + full_offsets, mask=mask, other=0.0).to(tl.float32)
        - tl.load(mean_full_ptr + c, mask=c_mask, other=0.0).to(tl.float32)[:, None]
    )

    tl.atomic_add(
        sum_full_ptr + c,
        tl.sum(tl.where(mask, x, 0.0), axis=1),
        sem="relaxed",
        mask=c_mask,
    )
    tl.atomic_add(
        sum_full_rhs_ptr + c,
        tl.sum(tl.where(mask, x * rhs_full, 0.0), axis=1),
        sem="relaxed",
        mask=c_mask,
    )

    slice_active = c >= SLICE_START_
    slice_c = tl.where(slice_active, c - SLICE_START_, 0)
    slice_mask = mask & slice_active[:, None] & (slice_c[:, None] < C_SLICE_)
    slice_offsets = (
        n[None, :] * (C_SLICE_ * HW_) + slice_c[:, None] * HW_ + hw[None, :]
    )
    rhs_slice = (
        tl.load(rhs_slice_ptr + slice_offsets, mask=slice_mask, other=0.0).to(tl.float32)
        - tl.load(mean_slice_ptr + slice_c, mask=c_mask & slice_active, other=0.0).to(
            tl.float32
        )[:, None]
    )

    tl.atomic_add(
        sum_slice_ptr + slice_c,
        tl.sum(tl.where(slice_mask, x, 0.0), axis=1),
        sem="relaxed",
        mask=c_mask & slice_active,
    )
    tl.atomic_add(
        sum_slice_rhs_ptr + slice_c,
        tl.sum(tl.where(slice_mask, x * rhs_slice, 0.0), axis=1),
        sem="relaxed",
        mask=c_mask & slice_active,
    )


@triton.jit
def _full_bn_epilogue_kernel(
    clone_ptr,
    getitem_ptr,
    rhs_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    out_ptr,
    out_scale_grad_ptr,
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

    x = (
        tl.load(clone_ptr + linear, mask=active, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + linear, mask=active, other=0.0).to(tl.float32)
    )
    rhs = (
        tl.load(rhs_ptr + linear, mask=active, other=0.0).to(tl.float32)
        - tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    )
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs_ptr + c, mask=active, other=0.0).to(tl.float32)

    mean_term = sum_x * SCALE_
    variance_term = sum_x_rhs * SCALE_ * rsqrt * rsqrt
    out = (x - rhs * variance_term - mean_term) * (rsqrt * weight)

    tl.store(out_ptr + linear, out, mask=active)
    tl.store(
        out_scale_grad_ptr + c,
        sum_x_rhs * rsqrt,
        mask=active & (n == 0) & (hw == 0),
    )


@triton.jit
def _slice_bn_epilogue_kernel(
    clone_ptr,
    getitem_ptr,
    rhs_ptr,
    mean_ptr,
    rsqrt_ptr,
    weight_ptr,
    sum_x_ptr,
    sum_x_rhs_ptr,
    out_ptr,
    out_scale_grad_ptr,
    C_: tl.constexpr,
    C_SLICE_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < NUMEL_

    c = linear % C_SLICE_
    hw = (linear // C_SLICE_) % HW_
    n = linear // (C_SLICE_ * HW_)
    x_offsets = n * (C_ * HW_) + (SLICE_START_ + c) * HW_ + hw
    rhs_offsets = n * (C_SLICE_ * HW_) + c * HW_ + hw

    x = (
        tl.load(clone_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        + tl.load(getitem_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
    )
    rhs = (
        tl.load(rhs_ptr + rhs_offsets, mask=active, other=0.0).to(tl.float32)
        - tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    )
    rsqrt = tl.load(rsqrt_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x = tl.load(sum_x_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_x_rhs = tl.load(sum_x_rhs_ptr + c, mask=active, other=0.0).to(tl.float32)

    mean_term = sum_x * SCALE_
    variance_term = sum_x_rhs * SCALE_ * rsqrt * rsqrt
    out = (x - rhs * variance_term - mean_term) * (rsqrt * weight)

    tl.store(out_ptr + linear, out, mask=active)
    tl.store(
        out_scale_grad_ptr + c,
        sum_x_rhs * rsqrt,
        mask=active & (n == 0) & (hw == 0),
    )


def oracle_full(
    clone_3: torch.Tensor,
    getitem_63: torch.Tensor,
    arg409_1: torch.Tensor,
    arg491_1: torch.Tensor,
    arg410_1: torch.Tensor,
    arg150_1: torch.Tensor,
    arg404_1: torch.Tensor,
    arg493_1: torch.Tensor,
    arg405_1: torch.Tensor,
    arg146_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clone_3.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")

    assert clone_3.shape == (N, C, H, W)
    assert getitem_63.shape == (N, C, H, W)
    assert arg409_1.shape == (N, C, H, W)
    assert arg491_1.shape == (1, C, 1, 1)
    assert arg410_1.shape == (C,)
    assert arg150_1.shape == (C,)
    assert arg404_1.shape == (N, C_SLICE, H, W)
    assert arg493_1.shape == (1, C_SLICE, 1, 1)
    assert arg405_1.shape == (C_SLICE,)
    assert arg146_1.shape == (C_SLICE,)
    assert clone_3.is_contiguous()
    assert getitem_63.is_contiguous()
    assert arg409_1.is_contiguous()
    assert arg491_1.is_contiguous()
    assert arg410_1.is_contiguous()
    assert arg150_1.is_contiguous()
    assert arg404_1.is_contiguous()
    assert arg493_1.is_contiguous()
    assert arg405_1.is_contiguous()
    assert arg146_1.is_contiguous()

    device = clone_3.device
    sum_full = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_full_rhs = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_slice = torch.zeros((C_SLICE,), device=device, dtype=torch.float32)
    sum_slice_rhs = torch.zeros((C_SLICE,), device=device, dtype=torch.float32)

    _two_branch_reduce_split_k_kernel[
        (triton.cdiv(C, REDUCE_BLOCK_C), triton.cdiv(TOTAL_SPATIAL, REDUCE_BLOCK_K))
    ](
        clone_3,
        getitem_63,
        arg409_1,
        arg491_1,
        arg404_1,
        arg493_1,
        sum_full,
        sum_full_rhs,
        sum_slice,
        sum_slice_rhs,
        C_=C,
        C_SLICE_=C_SLICE,
        SLICE_START_=SLICE_START,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        BLOCK_C=REDUCE_BLOCK_C,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=4,
    )

    out_full = torch.empty_strided(
        (N, C, H, W),
        FULL_OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    out_full_scale_grad = torch.empty((C,), device=device, dtype=torch.float32)
    _full_bn_epilogue_kernel[(triton.cdiv(NUMEL_FULL, EPILOGUE_BLOCK_ELEMS),)](
        clone_3,
        getitem_63,
        arg409_1,
        arg491_1,
        arg410_1,
        arg150_1,
        sum_full,
        sum_full_rhs,
        out_full,
        out_full_scale_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL_FULL,
        SCALE_=SCALE,
        BLOCK_ELEMS=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    out_slice = torch.empty_strided(
        (N, C_SLICE, H, W),
        SLICE_OUT_STRIDE,
        device=device,
        dtype=torch.float32,
    )
    out_slice_scale_grad = torch.empty((C_SLICE,), device=device, dtype=torch.float32)
    _slice_bn_epilogue_kernel[(triton.cdiv(NUMEL_SLICE, EPILOGUE_BLOCK_ELEMS),)](
        clone_3,
        getitem_63,
        arg404_1,
        arg493_1,
        arg405_1,
        arg146_1,
        sum_slice,
        sum_slice_rhs,
        out_slice,
        out_slice_scale_grad,
        C_=C,
        C_SLICE_=C_SLICE,
        SLICE_START_=SLICE_START,
        HW_=HW,
        NUMEL_=NUMEL_SLICE,
        SCALE_=SCALE,
        BLOCK_ELEMS=EPILOGUE_BLOCK_ELEMS,
        num_warps=4,
    )

    return out_full, out_full_scale_grad, out_slice, out_slice_scale_grad


@oracle_impl(hardware="H100", shapes="(T([512, 160, 7, 7], f32), T([512, 160, 7, 7], f32), T([512, 160, 7, 7], f32), T([1, 160, 1, 1], f32), T([160], f32), T([160], f32), T([512, 80, 7, 7], f32), T([1, 80, 1, 1], f32), T([80], f32), T([80], f32))")
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
