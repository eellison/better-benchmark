"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full LLaMA complex RoPE broadcast multiply for both complex inputs in one Triton kernel, reusing the `complex64[32,32]` factor viewed as `[1,32,1,32]` and emitting both `complex64[32,32,8,32]` outputs directly with real/imag math, whereas Inductor currently warns that it does not support code generation for complex operators and leaves this shared-factor complex pointwise pattern without native fused codegen; Inductor cannot do this today because its scheduler/codegen lacks a complex-aware broadcast multiply pattern that decomposes the sibling complex outputs into real/imag loads, arithmetic, and stores in one kernel; the fix is NEW_PATTERN: add a native complex broadcast-multiply lowering for RoPE factors that emits both sibling outputs from one real/imag pointwise kernel."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _complex_rope_mul_kernel(
        factor_ri,
        x0_ri,
        x1_ri,
        out0_ri,
        out1_ri,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N

        # Repro views factor [32,32] as [1,32,1,32] before broadcasting.
        d1 = (offsets // 256) % 32
        d3 = offsets % 32
        factor_offsets = (d1 * 32 + d3) * 2
        tensor_offsets = offsets * 2

        fr = tl.load(factor_ri + factor_offsets, mask=mask, other=0.0).to(tl.float32)
        fi = tl.load(factor_ri + factor_offsets + 1, mask=mask, other=0.0).to(tl.float32)

        x0r = tl.load(x0_ri + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        x0i = tl.load(x0_ri + tensor_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        out0r = x0r * fr - x0i * fi
        out0i = x0r * fi + x0i * fr
        tl.store(out0_ri + tensor_offsets, out0r, mask=mask)
        tl.store(out0_ri + tensor_offsets + 1, out0i, mask=mask)

        x1r = tl.load(x1_ri + tensor_offsets, mask=mask, other=0.0).to(tl.float32)
        x1i = tl.load(x1_ri + tensor_offsets + 1, mask=mask, other=0.0).to(tl.float32)
        out1r = x1r * fr - x1i * fi
        out1i = x1r * fi + x1i * fr
        tl.store(out1_ri + tensor_offsets, out1r, mask=mask)
        tl.store(out1_ri + tensor_offsets + 1, out1i, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([32, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), T([32, 32, 8, 32], torch.complex64), S([1, 32, 1, 32]))")
def oracle_forward(inputs):
    """Run the oracle computation.

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

    slice_1, view_as_complex_14, view_as_complex_15, _shape_param_0 = inputs
    out0 = torch.empty_like(view_as_complex_14)
    out1 = torch.empty_like(view_as_complex_15)

    n = out0.numel()
    block_n = 256
    grid = (triton.cdiv(n, block_n),)
    _complex_rope_mul_kernel[grid](
        torch.view_as_real(slice_1),
        torch.view_as_real(view_as_complex_14),
        torch.view_as_real(view_as_complex_15),
        torch.view_as_real(out0),
        torch.view_as_real(out1),
        N=n,
        BLOCK_N=block_n,
        num_warps=4,
    )
    return (out0, out1)


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
