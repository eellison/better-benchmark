"""
Oracle for var_mean_d61c9475e968

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete metadata reshape, fp32 hidden-size-768 layernorm, affine epilogue, and final contiguous `[128, 768]` reshape in one direct Triton row kernel, whereas Inductor already lowers this small fixed-K layernorm through its generic norm-template machinery with the same essential row-local reduction and store; Inductor cannot materially beat this today because the remaining work is the required read of input/weight/bias, two row reductions, rsqrt, and output write rather than a missed fusion across real producer or consumer kernels; the fix is BANDWIDTH_BOUND: classify this repro as at-floor unless broader norm-template launch overhead or vectorization improvements apply across many such small layernorms.
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

ROWS = 128
HIDDEN = 768
EPS = 1.0e-6
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = []
    stride = 1
    for size in reversed(shape):
        strides.append(stride)
        stride *= max(int(size), 1)
    return tuple(reversed(strides))


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
        ],
        key=["hidden"],
    )
    @triton.jit
    def _layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=0) / hidden
        mean_sq = tl.sum(x_for_reduce * x_for_reduce, axis=0) / hidden

        centered = x - mean
        var = mean_sq - mean * mean
        invstd = tl.rsqrt(var + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    addmm_49, weight, bias, shape0, shape1 = inputs
    tensor_inputs = (addmm_49, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("first three repro inputs must be tensors")

    expected_shapes = ((ROWS, HIDDEN), (HIDDEN,), (HIDDEN,))
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous")

    shape0_tuple = tuple(int(dim) for dim in shape0)
    shape1_tuple = tuple(int(dim) for dim in shape1)
    if shape0_tuple != (ROWS, 1, HIDDEN) or shape1_tuple != (ROWS, HIDDEN):
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return addmm_49, weight, bias, shape0_tuple, shape1_tuple


@oracle_impl(hardware="H100", shapes="(T([128, 768], f32), T([768], f32), T([768], f32), S([128, 1, 768]), S([128, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward computation with a direct row-layernorm kernel.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single float32 tensor with the final `[128, 768]` contiguous layout.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm.py")

    addmm_49, weight, bias, _shape0, shape1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        shape1,
        _contiguous_strides(shape1),
        device=addmm_49.device,
        dtype=addmm_49.dtype,
    )

    _layernorm_kernel[(ROWS,)](
        addmm_49,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
    )
    return out


def main() -> None:
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
