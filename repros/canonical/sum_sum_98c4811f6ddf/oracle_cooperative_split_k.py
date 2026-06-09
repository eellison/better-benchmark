"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle cooperatively split-K-reduces the DenseNet batch-norm-backward channel summaries over the shared `(N, H, W)` reduction domain, then uses the finalized summaries to write the full Repro return tuple consisting of `sum2 * rsqrt` and the strided last-32-channel input-gradient slice, whereas Inductor currently schedules generic channel reductions and a separate sliced epilogue around the same masked BN-backward producer; Inductor cannot do this today because the scheduler has no cooperative split-K multi-output reduction template that keeps the ReLU-mask producer, sibling channel accumulators, and required strided side-output epilogue in one coordinated plan; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible BN-backward reduction dimensions across spatial tiles, atomically or partially accumulate multiple channel summaries, and fuse the dependent returned-slice epilogue without rereading or materializing unnecessary channels."""
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



REPRO_ID = "sum_sum_98c4811f6ddf"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 64
C = 1024
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
SLICE_START = 992
SLICE_C = 32
SCALE = 0.00031887755102040814



@triton.jit
def _masked_bn_dual_reduce_split_k_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
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
    x_offsets = n[None, :] * (C_ * HW_) + c[:, None] * HW_ + hw[None, :]
    mm_offsets = n[None, :] * C_ + c[:, None]

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    centered = x - mean[:, None]
    bn_affine = centered * rsqrt[:, None] * affine_weight[:, None] + affine_bias[:, None]
    masked_grad = tl.where(bn_affine > 0.0, mm / 49.0, 0.0)

    tl.atomic_add(
        sum_where_ptr + c,
        tl.sum(masked_grad, axis=1),
        sem="relaxed",
        mask=c_mask,
    )
    tl.atomic_add(
        sum_where_centered_ptr + c,
        tl.sum(masked_grad * centered, axis=1),
        sem="relaxed",
        mask=c_mask,
    )


@triton.jit
def _final_sum_and_slice_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum_where_ptr,
    sum_where_centered_ptr,
    out_sum_ptr,
    full_grad_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    SLICE_C_: tl.constexpr,
    NUMEL_: tl.constexpr,
    SCALE_: tl.constexpr,
    SUM_BLOCK_C: tl.constexpr,
    BLOCK_ELEMS: tl.constexpr,
):
    pid = tl.program_id(0)

    out_c = pid * SUM_BLOCK_C + tl.arange(0, SUM_BLOCK_C)
    out_c_mask = out_c < C_
    out_sum_centered = tl.load(sum_where_centered_ptr + out_c, mask=out_c_mask, other=0.0)
    out_rsqrt = tl.load(rsqrt_ptr + out_c, mask=out_c_mask, other=0.0)
    tl.store(out_sum_ptr + out_c, out_sum_centered * out_rsqrt, mask=out_c_mask)

    linear = pid * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    mask = linear < NUMEL_

    hw = linear % HW_
    slice_c = (linear // HW_) % SLICE_C_
    n = linear // (SLICE_C_ * HW_)
    c = SLICE_START_ + slice_c

    x_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_where_centered = tl.load(sum_where_centered_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = x - mean
    bn_affine = centered * rsqrt * affine_weight + affine_bias
    masked_grad = tl.where(bn_affine > 0.0, mm / 49.0, 0.0)
    variance_term = sum_where_centered * SCALE_ * rsqrt * rsqrt
    mean_term = sum_where * SCALE_
    grad = (masked_grad - centered * variance_term - mean_term) * (rsqrt * affine_weight)

    tl.store(full_grad_ptr + x_offsets, grad, mask=mask)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_fused(
    mm: torch.Tensor,
    x: torch.Tensor,
    mean_1c11: torch.Tensor,
    rsqrt_1c11: torch.Tensor,
    affine_weight: torch.Tensor,
    affine_bias: torch.Tensor,
    _shape_param_0: object,
    _shape_param_1: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1
    assert mm.shape == (N, C)
    assert x.shape == (N, C, H, W)
    assert mean_1c11.shape == (1, C, 1, 1)
    assert rsqrt_1c11.shape == (1, C, 1, 1)
    assert affine_weight.shape == (C,)
    assert affine_bias.shape == (C,)

    device = x.device
    mean = mean_1c11.reshape(C)
    rsqrt = rsqrt_1c11.reshape(C)

    sum_where = torch.zeros((C,), device=device, dtype=torch.float32)
    sum_where_centered = torch.zeros((C,), device=device, dtype=torch.float32)

    block_c = 8
    block_k = 512
    reduce_grid = (triton.cdiv(C, block_c), triton.cdiv(TOTAL_SPATIAL, block_k))
    _masked_bn_dual_reduce_split_k_kernel[reduce_grid](
        mm,
        x,
        mean,
        rsqrt,
        affine_weight,
        affine_bias,
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
    full_grad = torch.empty((N, C, H, W), device=device, dtype=torch.float32)
    slice_numel = N * SLICE_C * HW
    block_elems = 256
    _final_sum_and_slice_epilogue_kernel[(triton.cdiv(slice_numel, block_elems),)](
        mm,
        x,
        mean,
        rsqrt,
        affine_weight,
        affine_bias,
        sum_where,
        sum_where_centered,
        out_sum,
        full_grad,
        C_=C,
        HW_=HW,
        SLICE_START_=SLICE_START,
        SLICE_C_=SLICE_C,
        NUMEL_=slice_numel,
        SCALE_=SCALE,
        SUM_BLOCK_C=4,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out_sum, full_grad[:, SLICE_START : SLICE_START + SLICE_C]


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        return _as_tuple(model(*inputs))


@oracle_impl(hardware="H100", shapes="(T([64, 1024], f32), T([64, 1024, 7, 7], f32), T([1, 1024, 1, 1], f32), T([1, 1024, 1, 1], f32), T([1024], f32), T([1024], f32), S([64, 1024, 1, 1]), S([64, 1024, 7, 7]))")
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
