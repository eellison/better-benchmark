"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the full EfficientNet SiLU-gradient plus batch-norm-backward return tuple by split-K reducing the two per-channel BN summaries from the original repro inputs and then recomputing the producer in a fused epilogue that writes both returned outputs, whereas Inductor currently lowers the decomposed pointwise producer, sibling `sum([0, 2, 3])` reductions, and dependent BN-backward epilogue as ordinary reduction/pointwise work with too little reduction-axis parallelism; Inductor cannot do this today because its scheduler/codegen does not have a cooperative split-K multi-output reduction template that can coordinate compatible sibling channel reductions and preserve the dependent materialized tensor epilogue; the fix is COOPERATIVE_SPLIT_K: teach Inductor to split compatible small-output channel reductions across the reduced N/H/W dimension, combine the partials, and fuse the downstream BN-backward epilogue and side vector output."""
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



REPRO_ID = "sum_sum_04bdad42e9ef"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 1280
H = 7
W = 7
HW = H * W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
INV_HW = 1.0 / HW
SCALE = 1.0 / TOTAL_SPATIAL



@triton.jit
def _silu_bn_dual_reduce_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum1_ptr,
    sum2_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    TOTAL_SPATIAL_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < TOTAL_SPATIAL_

    n = k // HW_
    hw = k - n * HW_
    x_offsets = n * (C_ * HW_) + c * HW_ + hw
    mm_offsets = n * C_ + c

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c).to(tl.float32)

    centered = x - mean
    z = centered * rsqrt * affine_weight + affine_bias
    sig = 1.0 / (tl.exp(-z) + 1.0)
    producer = (mm * INV_HW_) * sig * (z * (1.0 - sig) + 1.0)

    tl.atomic_add(sum1_ptr + c, tl.sum(producer, axis=0), sem="relaxed")
    tl.atomic_add(sum2_ptr + c, tl.sum(producer * centered, axis=0), sem="relaxed")


@triton.jit
def _silu_bn_epilogue_kernel(
    mm_ptr,
    x_ptr,
    mean_ptr,
    rsqrt_ptr,
    affine_weight_ptr,
    affine_bias_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    out_weight_grad_ptr,
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mm = tl.load(mm_ptr + n * C_ + c, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    rsqrt = tl.load(rsqrt_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_weight = tl.load(affine_weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    affine_bias = tl.load(affine_bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + c, mask=mask, other=0.0).to(tl.float32)

    centered = x - mean
    z = centered * rsqrt * affine_weight + affine_bias
    sig = 1.0 / (tl.exp(-z) + 1.0)
    producer = (mm * INV_HW_) * sig * (z * (1.0 - sig) + 1.0)

    mean_term = sum1 * SCALE_
    variance_term = sum2 * SCALE_ * rsqrt * rsqrt
    out = (producer - centered * variance_term - mean_term) * (rsqrt * affine_weight)
    tl.store(out_ptr + offsets, out, mask=mask)

    is_first_hw_for_channel = mask & (n == 0) & ((offsets % HW_) == 0)
    tl.store(out_weight_grad_ptr + c, sum2 * rsqrt, mask=is_first_hw_for_channel)


def make_inputs() -> tuple[object, ...]:
    module = _load_repro_module()
    return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in module.make_inputs())


def oracle_full(
    mm: torch.Tensor,
    arg405_1: torch.Tensor,
    arg406_1: torch.Tensor,
    arg407_1: torch.Tensor,
    arg162_1: torch.Tensor,
    arg163_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1

    assert mm.shape == (N, C)
    assert arg405_1.shape == (N, C, H, W)
    assert arg406_1.shape == (1, C, 1, 1)
    assert arg407_1.shape == (1, C, 1, 1)
    assert arg162_1.shape == (C,)
    assert arg163_1.shape == (C,)
    assert mm.is_contiguous()
    assert arg405_1.is_contiguous()
    assert arg406_1.is_contiguous()
    assert arg407_1.is_contiguous()
    assert arg162_1.is_contiguous()
    assert arg163_1.is_contiguous()

    sum1 = torch.zeros((C,), device=arg405_1.device, dtype=torch.float32)
    sum2 = torch.zeros((C,), device=arg405_1.device, dtype=torch.float32)

    block_k = 2048
    _silu_bn_dual_reduce_kernel[(C, triton.cdiv(TOTAL_SPATIAL, block_k))](
        mm,
        arg405_1,
        arg406_1,
        arg407_1,
        arg162_1,
        arg163_1,
        sum1,
        sum2,
        C_=C,
        HW_=HW,
        TOTAL_SPATIAL_=TOTAL_SPATIAL,
        INV_HW_=INV_HW,
        BLOCK_K=block_k,
        num_warps=8,
    )

    out = torch.empty_like(arg405_1)
    out_weight_grad = torch.empty((C,), device=arg405_1.device, dtype=torch.float32)

    block_elems = 256
    _silu_bn_epilogue_kernel[(triton.cdiv(NUMEL, block_elems),)](
        mm,
        arg405_1,
        arg406_1,
        arg407_1,
        arg162_1,
        arg163_1,
        sum1,
        sum2,
        out,
        out_weight_grad,
        C_=C,
        HW_=HW,
        NUMEL_=NUMEL,
        INV_HW_=INV_HW,
        SCALE_=SCALE,
        BLOCK_ELEMS=block_elems,
        num_warps=4,
    )

    return out, out_weight_grad


def reference_outputs(inputs: tuple[object, ...]) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().cuda()
    with torch.no_grad():
        out = model(*inputs)
    if isinstance(out, tuple):
        return out
    return (out,)


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
