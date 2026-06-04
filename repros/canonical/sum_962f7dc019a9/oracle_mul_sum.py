"""
Oracle for sum_962f7dc019a9

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete captured `mm * arg307_1` producer and dim-0 sum as one shape-specialized Triton column-reduction kernel that reads both contiguous `f32[128, 768]` inputs once, multiplies in the timed scope, and writes the final contiguous `f32[768]` output directly, whereas Inductor already lowers the same producer-reduction graph to a fused generic reduction kernel; Inductor cannot materially do less work today through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, recompute fusion, or a new pattern because the required two input reads, f32 multiply-accumulate, output store, and one GPU launch dominate this sub-megabyte workload; the fix is BANDWIDTH_BOUND: record this as an at-floor diagnosis unless this full-scope Triton kernel beats both required tuned Inductor configs on the same shape.
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
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "BANDWIDTH_BOUND"

M = 128
N = 768
OUTPUT_SHAPE = (N,)
OUTPUT_STRIDE = (1,)


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _mul_sum_128x768_dim0_kernel(
        lhs_ptr,
        rhs_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, 128)
        offsets = rows[:, None] * 768 + cols[None, :]
        mask = cols < 768
        lhs = tl.load(lhs_ptr + offsets, mask=mask[None, :], other=0.0)
        rhs = tl.load(rhs_ptr + offsets, mask=mask[None, :], other=0.0)
        totals = tl.sum(lhs.to(tl.float32) * rhs.to(tl.float32), axis=0)
        tl.store(out_ptr + cols, totals, mask=mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    lhs, rhs = inputs
    for idx, tensor in enumerate((lhs, rhs)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"expected tensor input {idx}, got {type(tensor)!r}")
        if tensor.dtype is not torch.float32:
            raise TypeError(f"expected float32 input {idx}, got {tensor.dtype}")
        if tuple(tensor.shape) != (M, N):
            raise ValueError(f"expected input {idx} shape {(M, N)}, got {tuple(tensor.shape)}")
        if not tensor.is_cuda:
            raise ValueError("oracle_mul_sum.py expects CUDA inputs")
        if not tensor.is_contiguous():
            raise ValueError(f"expected contiguous input {idx}, got stride={tensor.stride()}")

    return lhs, rhs


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: accepts the same two f32[128, 768] inputs as Repro.forward()
    and returns the same single contiguous f32[768] output. The elementwise
    multiply is inside the timed Triton reduction kernel.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mul_sum.py")

    lhs, rhs = _validate_inputs(inputs)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=lhs.device, dtype=lhs.dtype)
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _mul_sum_128x768_dim0_kernel[grid](lhs, rhs, out)
    return out


def main() -> None:
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
                    print(f"diagnosis_only: required comparison shows not_true_floor "
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
