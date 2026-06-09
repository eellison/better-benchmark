"""
Oracle for var_mean_15b838f5d029

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured layernorm scope in one shape-specialized Triton row kernel, including the `[128,768] -> [128,1,768]` view semantics, fp32 `var_mean(..., dim=2, correction=0, keepdim=True)`, eps=1e-6 normalization, affine scale/bias, and final contiguous `[128,768]` view, whereas tuned Inductor already lowers this norm-template graph to a single fused row-normalization kernel for the same scope; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination, or recompute fusion because the remaining cost is one small-kernel launch plus the mandatory activation/affine reads, row reduction/rsqrt, and output store rather than an avoidable materialization or missed producer-consumer fusion; the fix is BANDWIDTH_BOUND: record this as a full-scope floor/diagnosis and only reopen if a broader norm-template or launch-overhead change beats the measured Triton floor.
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


if triton is not None:

    @triton.jit
    def _layernorm_affine_kernel(
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
        sq_mean = tl.sum(x_for_reduce * x_for_reduce, axis=0) / hidden
        variance = tl.maximum(sq_mean - mean * mean, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, weight, bias, view_shape, out_shape = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"input 0 must be a tensor, got {type(x)!r}")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise TypeError("weight and bias inputs must be tensors")

    if tuple(x.shape) != (ROWS, HIDDEN):
        raise ValueError(f"input shape {tuple(x.shape)} != {(ROWS, HIDDEN)}")
    if tuple(weight.shape) != (HIDDEN,) or tuple(bias.shape) != (HIDDEN,):
        raise ValueError(
            f"expected affine tensors with shape ({HIDDEN},), "
            f"got weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError(f"expected f32 tensors, got {x.dtype}, {weight.dtype}, {bias.dtype}")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if x.stride() != (HIDDEN, 1) or weight.stride() != (1,) or bias.stride() != (1,):
        raise ValueError(
            "expected contiguous inputs, got "
            f"x_stride={x.stride()} weight_stride={weight.stride()} bias_stride={bias.stride()}"
        )
    if list(view_shape) != [ROWS, 1, HIDDEN] or list(out_shape) != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected shape parameters: {view_shape!r}, {out_shape!r}")

    return x, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 768], f32), T([768], f32), T([768], f32), S([128, 1, 768]), S([128, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward layernorm affine computation.

    SCOPE INVARIANT: accepts the same five inputs as Repro.forward() and returns
    the same single f32 `[128,768]` tensor with contiguous `(768,1)` stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided((ROWS, HIDDEN), (HIDDEN, 1), device=x.device, dtype=x.dtype)
    _layernorm_affine_kernel[(ROWS,)](
        x,
        weight,
        bias,
        out,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK_N=triton.next_power_of_2(HIDDEN),
        num_warps=8,
        num_stages=3,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and expected.stride() == actual.stride()
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
