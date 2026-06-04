"""
Oracle for var_mean_1b88cfe94b3c

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete fixed-shape group-normalization region, affine scale/bias, residual add, and ReLU in one specialized Triton row kernel that autotunes how many 32-element groups each program handles and writes the final contiguous `[64, 256, 2, 2]` output directly, whereas Inductor lowers the same fused region through its generic norm/reduction template machinery; Inductor cannot materially improve this repro today because the captured K=32 reduction is already a single small fused normalization pass dominated by mandatory input/residual/affine memory traffic, output stores, rsqrt latency, and launch overhead rather than by a missing cross-op fusion; the fix is BANDWIDTH_BOUND: record this case as at the full-scope floor unless a broader launch/template-overhead reduction applies across many norm-template repros.
"""
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

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 4}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
        ],
        key=["total_groups"],
    )
    @triton.jit
    def _groupnorm_residual_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        out_ptr,
        total_groups: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        elems = tl.arange(0, BLOCK_K)
        mask = (rows[:, None] < total_groups) & (elems[None, :] < 32)
        flat_offset = rows[:, None] * 32 + elems[None, :]

        x = tl.load(x_ptr + flat_offset, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=1) * 0.03125
        sq_mean = tl.sum(x * x, axis=1) * 0.03125
        var = tl.maximum(sq_mean - mean * mean, 0.0)
        centered = x - mean[:, None]

        channel = ((rows[:, None] % 32) * 32 + elems[None, :]) // 4
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + flat_offset, mask=mask, other=0.0).to(tl.float32)
        normalized = centered * tl.rsqrt(var[:, None] + 1.0e-5)
        out = normalized * weight + bias + residual
        out = tl.maximum(out, 0.0)
        tl.store(out_ptr + flat_offset, out, mask=mask)


def _check_exact_default_scope(inputs):
    x, weight, bias, residual, shape0, shape1 = inputs
    expected_shape = (64, 256, 2, 2)
    if (
        tuple(x.shape) != expected_shape
        or tuple(weight.shape) != (256,)
        or tuple(bias.shape) != (256,)
        or tuple(residual.shape) != expected_shape
        or tuple(shape0) != (64, 32, 8, 4)
        or tuple(shape1) != expected_shape
    ):
        raise RuntimeError(
            "oracle_groupnorm_residual.py is specialized to the default "
            "var_mean_1b88cfe94b3c scope: f32[64,256,2,2] with view "
            "[64,32,8,4]"
        )
    if (
        x.dtype is not torch.float32
        or weight.dtype is not torch.float32
        or bias.dtype is not torch.float32
        or residual.dtype is not torch.float32
    ):
        raise RuntimeError("oracle_groupnorm_residual.py expects float32 tensor inputs")
    if not (x.is_cuda and weight.is_cuda and bias.is_cuda and residual.is_cuda):
        raise RuntimeError("oracle_groupnorm_residual.py requires CUDA tensors")
    if not (x.is_contiguous() and weight.is_contiguous() and bias.is_contiguous() and residual.is_contiguous()):
        raise RuntimeError("oracle_groupnorm_residual.py expects contiguous default repro inputs")


def oracle_forward(inputs):
    """Run the full captured Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_groupnorm_residual.py")

    _check_exact_default_scope(inputs)
    x, weight, bias, residual, _shape0, _shape1 = inputs
    out = torch.empty_strided(
        (64, 256, 2, 2),
        (1024, 4, 2, 1),
        device=x.device,
        dtype=torch.float32,
    )
    total_groups = 64 * 32
    grid = lambda meta: (triton.cdiv(total_groups, meta["BLOCK_M"]),)
    _groupnorm_residual_relu_kernel[grid](
        x,
        weight,
        bias,
        residual,
        out,
        total_groups=total_groups,
        BLOCK_K=32,
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
