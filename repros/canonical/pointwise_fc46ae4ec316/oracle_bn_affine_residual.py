"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full captured inference-BatchNorm affine plus residual-add graph in one hand-specialized contiguous Triton streaming kernel, preserving the per-channel fp32 mean/variance conversion, broadcasted scale/bias, fp16 cast before the residual add, and final fp16 `[N,C,H,W]` output with equal-NaN semantics, whereas Inductor already fuses the pointwise graph but uses a slower generic broadcast/indexing schedule for this rank-4 layout; Inductor cannot do this today because its pointwise codegen does not recognize this per-channel BatchNorm-affine-residual broadcast as a layout-specialized streaming template with channel parameters reused across spatial elements; the fix is NEW_PATTERN: add a guarded contiguous NCHW per-channel affine-residual pointwise template or schedule choice that avoids generic broadcast index overhead."""
from __future__ import annotations

import argparse
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    return _harness_get_repro_instance(REPRO_DIR)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
    ],
    key=["TOTAL", "C", "HW"],
)
@triton.jit
def _bn_affine_residual_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL
    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    affine = (x - mean) * tl.rsqrt(var + 1.0e-5) * weight + bias
    rounded_affine = affine.to(tl.float16).to(tl.float32)
    result = rounded_affine.to(tl.float32) + residual.to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


def oracle_forward(inputs):
    mean, x, var, weight, bias, residual = inputs
    n, c, h, w = x.shape
    total = x.numel()
    hw = h * w

    assert mean.shape == (c,)
    assert var.shape == (c,)
    assert weight.shape == (c,)
    assert bias.shape == (c,)
    assert residual.shape == x.shape
    assert x.is_cuda
    assert x.dtype == torch.float16
    assert residual.dtype == torch.float16
    assert mean.dtype == torch.float16
    assert var.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert bias.dtype == torch.float16
    assert x.is_contiguous()
    assert residual.is_contiguous()

    out = torch.empty_strided(x.shape, x.stride(), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(total, 1024),)
    _bn_affine_residual_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        residual,
        out,
        TOTAL=total,
        C=c,
        HW=hw,
    )
    return out


def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
