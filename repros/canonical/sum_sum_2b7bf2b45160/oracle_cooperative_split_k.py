"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete MobileNetV3 hard-swish plus batch-norm-backward return tuple from Repro.forward by split-K reducing the two per-channel summaries over the shared [256, 7, 7] domain, returning sum(where * centered) * invstd, and using the finalized summaries to write the full [256, 960, 7, 7] input-gradient tensor, whereas Inductor currently schedules the hard-swish derivative producer, sibling channel reductions, and dependent BN-backward epilogue as separate generic pointwise and reduction kernels over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates compatible channel reductions with a dependent full-tensor side-output epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split the N/H/W reduction domain into partial channel summaries, finalize them once, and fuse the downstream BN-backward output store plus vector epilogue with those summaries."""
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



REPRO_ID = "sum_sum_2b7bf2b45160"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 256
C = 960
H = 7
W = 7
HW = H * W
TOTAL_REDUCTION = N * HW
NUMEL = N * C * HW
POOL_SCALE = 1.0 / HW
BN_SCALE = 7.971938775510203e-05

SUMMARY_BLOCK_C = 4
SUMMARY_BLOCK_K = 512
FINAL_BLOCK_C = 16
EPILOGUE_BLOCK = 256



@triton.jit
def _summary_partials_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    partial_where_ptr,
    partial_centered_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_REDUCTION_: tl.constexpr,
    POOL_SCALE_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
    BLOCK_K_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    k = tl.program_id(1) * BLOCK_K_ + tl.arange(0, BLOCK_K_)
    c_mask = c < C_
    k_mask = k < TOTAL_REDUCTION_
    mask = c_mask[:, None] & k_mask[None, :]

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mm_offsets = n[None, :] * C_ + c[:, None]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)

    centered = x - mean[:, None]
    affine = centered * invstd[:, None] * weight[:, None] + bias[:, None]
    pooled_grad = mm * POOL_SCALE_
    middle = pooled_grad * (affine * 0.3333333333333333 + 0.5)
    where_value = tl.where(affine < 3.0, middle, pooled_grad)
    where_value = tl.where(affine <= -3.0, full, where_value)
    where_value = tl.where(mask, where_value, 0.0)
    centered = tl.where(mask, centered, 0.0)

    partial_offset = tl.program_id(1) * C_ + c
    tl.store(
        partial_where_ptr + partial_offset,
        tl.sum(where_value, axis=1),
        mask=c_mask,
    )
    tl.store(
        partial_centered_ptr + partial_offset,
        tl.sum(where_value * centered, axis=1),
        mask=c_mask,
    )


@triton.jit
def _finalize_summaries_kernel(
    partial_where_ptr,
    partial_centered_ptr,
    invstd_ptr,
    sum_where_ptr,
    sum_centered_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    NUM_TILES_: tl.constexpr,
    BLOCK_TILES_: tl.constexpr,
    BLOCK_C_: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)
    tiles = tl.arange(0, BLOCK_TILES_)
    mask = (tiles[:, None] < NUM_TILES_) & (c[None, :] < C_)
    offsets = tiles[:, None] * C_ + c[None, :]

    sum_where = tl.sum(tl.load(partial_where_ptr + offsets, mask=mask, other=0.0), axis=0)
    sum_centered = tl.sum(
        tl.load(partial_centered_ptr + offsets, mask=mask, other=0.0),
        axis=0,
    )
    c_mask = c < C_
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    tl.store(sum_where_ptr + c, sum_where, mask=c_mask)
    tl.store(sum_centered_ptr + c, sum_centered, mask=c_mask)
    tl.store(out_scale_ptr + c, sum_centered * invstd, mask=c_mask)


@triton.jit
def _bn_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    full_ptr,
    sum_where_ptr,
    sum_centered_ptr,
    out_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL_: tl.constexpr,
    POOL_SCALE_: tl.constexpr,
    BN_SCALE_: tl.constexpr,
    BLOCK_ELEMS_: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_ELEMS_ + tl.arange(0, BLOCK_ELEMS_)
    active = offsets < NUMEL_

    hw = offsets % HW_
    c = (offsets // HW_) % C_
    n = offsets // (C_ * HW_)

    x = tl.load(x_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + n * C_ + c, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_centered = tl.load(sum_centered_ptr + c, mask=active, other=0.0).to(tl.float32)
    full = tl.load(full_ptr).to(tl.float32)

    centered = x - mean
    affine = centered * invstd * weight + bias
    pooled_grad = mm * POOL_SCALE_
    middle = pooled_grad * (affine * 0.3333333333333333 + 0.5)
    where_value = tl.where(affine < 3.0, middle, pooled_grad)
    where_value = tl.where(affine <= -3.0, full, where_value)

    mean_term = sum_where * BN_SCALE_
    variance_term = sum_centered * BN_SCALE_ * invstd * invstd
    out = (where_value - centered * variance_term - mean_term) * (invstd * weight)
    tl.store(out_ptr + offsets, out, mask=active)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(value.cuda() if isinstance(value, torch.Tensor) else value for value in module.make_inputs())


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


def oracle_full(
    mm_2: torch.Tensor,
    arg315_1: torch.Tensor,
    arg316_1: torch.Tensor,
    arg317_1: torch.Tensor,
    arg130_1: torch.Tensor,
    arg131_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Triton oracle")
    if mm_2.device.type != "cuda":
        raise RuntimeError("oracle_full expects CUDA inputs")

    assert mm_2.shape == (N, C)
    assert arg315_1.shape == (N, C, H, W)
    assert arg316_1.shape == (1, C, 1, 1)
    assert arg317_1.shape == (1, C, 1, 1)
    assert arg130_1.shape == (C,)
    assert arg131_1.shape == (C,)
    assert full.shape == ()
    assert mm_2.is_contiguous()
    assert arg315_1.is_contiguous()
    assert arg316_1.is_contiguous()
    assert arg317_1.is_contiguous()
    assert arg130_1.is_contiguous()
    assert arg131_1.is_contiguous()

    num_tiles = triton.cdiv(TOTAL_REDUCTION, SUMMARY_BLOCK_K)
    device = mm_2.device
    partial_where = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_centered = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    sum_centered = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)

    _summary_partials_kernel[(triton.cdiv(C, SUMMARY_BLOCK_C), num_tiles)](
        mm_2,
        arg315_1,
        arg316_1,
        arg317_1,
        arg130_1,
        arg131_1,
        full,
        partial_where,
        partial_centered,
        C_=C,
        HW_=HW,
        TOTAL_REDUCTION_=TOTAL_REDUCTION,
        POOL_SCALE_=POOL_SCALE,
        BLOCK_C_=SUMMARY_BLOCK_C,
        BLOCK_K_=SUMMARY_BLOCK_K,
        num_warps=4,
    )

    _finalize_summaries_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_where,
        partial_centered,
        arg317_1,
        sum_where,
        sum_centered,
        out_scale,
        C_=C,
        NUM_TILES_=num_tiles,
        BLOCK_TILES_=triton.next_power_of_2(num_tiles),
        BLOCK_C_=FINAL_BLOCK_C,
        num_warps=4,
    )

    out_grad = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    _bn_epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        mm_2,
        arg315_1,
        arg316_1,
        arg317_1,
        arg130_1,
        arg131_1,
        full,
        sum_where,
        sum_centered,
        out_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        POOL_SCALE_=POOL_SCALE,
        BN_SCALE_=BN_SCALE,
        BLOCK_ELEMS_=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return out_grad, out_scale


@oracle_impl(hardware="H100", shapes="(T([256, 960], f32), T([256, 960, 7, 7], f32), T([1, 960, 1, 1], f32), T([1, 960, 1, 1], f32), T([960], f32), T([960], f32), T([], f32), S([256, 960, 1, 1]), S([256, 960, 7, 7]))")
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
