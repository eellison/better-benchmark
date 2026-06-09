"""
Oracle kernel for amax_sum_3ed297ef02cd: online softmax forward.

Pattern: bf16[8192, 262144] -> cast f32 -> amax -> sub -> exp -> sum -> div -> cast bf16
This is the Qwen3-0.6B softmax forward with large vocabulary (262144).

Strategy:
  - Online softmax (Milakov & Gimelshein 2018): single pass over the reduction
    dimension computing running max and sum_exp simultaneously, then a second
    pass to normalize and write output.
  - Each program instance handles one row (or a few rows via ROWS_PER_PROGRAM).
  - Use scalar accumulators for max and sum_exp within each tile iteration.
  - BLOCK_SIZE chosen to balance register pressure vs loop iterations.

Memory traffic:
  - Read: 8192 * 262144 * 2 bytes (bf16 input)
  - Write: 8192 * 262144 * 2 bytes (bf16 output)
  - Total: 8,589,934,592 bytes (~8 GB)
  - SOL at 3.35 TB/s (H100 HBM): ~2564 us
  - SOL at 3.8e6 us/byte formula: total_bytes / 3.8e6 = 2261 us
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

# Problem dimensions
M = 8192       # number of rows
N = 262144     # reduction dimension (vocab size)


# --- Triton Kernel ---

if triton is not None:

    @triton.jit
    def online_softmax_fwd_kernel(
        input_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Online softmax forward: fused max + sum_exp + normalize.

        Each program handles one row of the input matrix.
        Pass 1: compute running max and sum_exp (online algorithm)
        Pass 2: normalize and write output
        """
        row_idx = tl.program_id(0)
        row_start = row_idx * N

        # Pass 1: Online computation of max and sum_exp
        m_i = float("-inf")  # running max
        l_i = 0.0           # running sum of exp(x - m)

        for block_start in tl.range(0, N, BLOCK_N):
            cols = block_start + tl.arange(0, BLOCK_N)
            mask = cols < N
            # Load block of input, cast to f32
            x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(x, axis=0))
            # Rescale previous sum and add new contributions
            l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
            m_i = m_new

        # Pass 2: Normalize and write output
        for block_start in tl.range(0, N, BLOCK_N):
            cols = block_start + tl.arange(0, BLOCK_N)
            mask = cols < N
            x = tl.load(input_ptr + row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)
            # softmax(x) = exp(x - max) / sum_exp
            out = tl.exp(x - m_i) / l_i
            # Cast back to bf16
            tl.store(output_ptr + row_start + cols, out.to(tl.bfloat16), mask=mask)


def online_softmax_triton(x: torch.Tensor) -> torch.Tensor:
    """Launch the online softmax Triton kernel."""
    assert x.ndim == 2
    M_val, N_val = x.shape
    assert x.dtype == torch.bfloat16

    output = torch.empty_like(x)

    # Choose BLOCK_N: larger blocks mean fewer loop iterations but more registers.
    # For N=262144, BLOCK_N=8192 gives 32 iterations per pass.
    BLOCK_N = 8192

    grid = (M_val,)
    online_softmax_fwd_kernel[grid](
        x, output,
        N=N_val,
        BLOCK_N=BLOCK_N,
    )
    return output


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@oracle_impl(hardware="H100", shapes="(T([8192, 262144], bf16))")
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
        raise RuntimeError("triton is required for this oracle")
    # The repro takes a single bf16[M, N] tensor and returns the softmax output
    x = inputs[0]
    return online_softmax_triton(x)


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
