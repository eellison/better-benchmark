"""
Oracle for var_mean_9f9f7f8de566

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured exact-erf GELU plus fp32 hidden-size-768 LayerNorm in one shape-specialized Triton row kernel, including the `[512, 768] -> [1, 512, 768]` view semantics, `0.5*x*(1+erf(x*0.7071067811865476))`, `var_mean(correction=0, keepdim=True)` with `eps=1e-12`, affine scale/bias, fp16 cast, and final contiguous `[512, 768]` view, whereas tuned Inductor already targets the same full scope with a fused normalization kernel and no avoidable intermediate materialization; Inductor cannot materially improve this today because the remaining cost is dominated by the required activation/affine memory traffic, exact erf evaluation, row reduction, rsqrt, and output store rather than a missed scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute opportunity; the fix is BANDWIDTH_BOUND: record this as a full-scope floor and only reopen for broad norm-template/math-codegen or launch-overhead improvements.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


ROWS = 512
HIDDEN = 768
BLOCK_N = 1024
EPS = 1.0e-12
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _gelu_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        eps: tl.constexpr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_n)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        erf_term = tl.math.erf(x * 0.7071067811865476) + 1.0
        gelu = (x * 0.5) * erf_term

        gelu_for_reduce = tl.where(mask, gelu, 0.0)
        mean = tl.sum(gelu_for_reduce, axis=0) / hidden
        centered = gelu - mean
        var = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / hidden
        invstd = tl.rsqrt(tl.maximum(var, 0.0) + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"expected 5 inputs, got {len(inputs)}")

    x, weight, bias, shape0, shape1 = inputs
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("first three repro inputs must be tensors")
    if tuple(x.shape) != (ROWS, HIDDEN):
        raise ValueError(f"input 0 shape {tuple(x.shape)} != {(ROWS, HIDDEN)}")
    if tuple(weight.shape) != (HIDDEN,) or tuple(bias.shape) != (HIDDEN,):
        raise ValueError(f"affine inputs must both have shape ({HIDDEN},)")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 or bias.dtype != torch.float16:
        raise TypeError("all tensor inputs must be torch.float16")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("all tensor inputs must be contiguous")
    if list(shape0) != [1, ROWS, HIDDEN] or list(shape1) != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return x, weight, bias


def oracle_forward(inputs):
    """Run the full Repro.forward computation with a shape-specialized Triton kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous [512, 768] float16 output tensor.
    """
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, weight, bias = _validate_inputs(inputs)
    out = torch.empty_like(x)
    _gelu_layernorm_kernel[(ROWS,)](
        x,
        weight,
        bias,
        out,
        eps=EPS,
        hidden=HIDDEN,
        block_n=BLOCK_N,
        num_warps=8,
    )
    return out


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
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
