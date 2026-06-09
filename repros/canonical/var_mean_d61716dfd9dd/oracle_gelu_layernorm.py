"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full captured ALBERT tanh-approximate GELU producer and fp32 hidden-size-128 LayerNorm in one shape-specialized Triton row kernel, including the `[4096,128] -> [8,512,128]` view semantics, `0.5*x*(1+tanh(0.7978845608028654*(x+0.044715*x^3)))`, `var_mean(correction=0, keepdim=True)` with `eps=1e-12`, affine scale/bias, and final contiguous `[4096,128]` view, whereas Inductor already lowers the same full scope into the same launch and memory-traffic envelope for this fixed ALBERT shape; Inductor cannot materially improve this today because the remaining cost is dominated by required tanh-GELU math, one hidden-size-128 row reduction, affine reads, rsqrt, output stores, and launch overhead rather than an actionable missed scheduler fusion or algebraic rewrite; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope GELU-LayerNorm case unless broader norm-template math-codegen or launch-overhead work moves the family."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


ROWS = 4096
HIDDEN = 128
BLOCK_N = 128
EPS = 1.0e-12

if triton is not None:

    @triton.jit
    def _gelu_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        hidden: tl.constexpr,
        block_n: tl.constexpr,
        eps: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_n)
        mask = cols < hidden
        offsets = row * hidden + cols

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x3 = x * x * x
        tanh_arg = (x + x3 * 0.044715) * 0.7978845608028654
        gelu = (x * 0.5) * (libdevice.tanh(tanh_arg) + 1.0)

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
        raise ValueError(f"input shape {tuple(x.shape)} != {(ROWS, HIDDEN)}")
    if tuple(weight.shape) != (HIDDEN,) or tuple(bias.shape) != (HIDDEN,):
        raise ValueError(f"affine inputs must both have shape ({HIDDEN},)")
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError("all tensor inputs must be torch.float32")
    if not x.is_cuda or not weight.is_cuda or not bias.is_cuda:
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not x.is_contiguous() or not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("all tensor inputs must be contiguous")
    if list(shape0) != [8, 512, HIDDEN] or list(shape1) != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected shape parameters: {shape0!r}, {shape1!r}")

    return x, weight, bias


@oracle_impl(hardware="H100", shapes="(T([4096, 128], f32), T([128], f32), T([128], f32), S([8, 512, 128]), S([4096, 128]))")
def oracle_forward(inputs):
    """Run the full Repro.forward computation with a shape-specialized kernel.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
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
        hidden=HIDDEN,
        block_n=BLOCK_N,
        eps=EPS,
        num_warps=4,
    )
    return out


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
