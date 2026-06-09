"""Gap diagnosis: this oracle computes the complete captured
`sum(tangents_1, [0], keepdim=True)` plus final `reshape([1000])` with one
shape-specialized Triton column-reduction kernel that reads the contiguous
`f32[128, 1000]` input once and writes the final contiguous `f32[1000]`
output directly, whereas Inductor already lowers this reshape-preserving
column sum to a single compiled reduction kernel in the same launch and
memory-traffic regime; Inductor cannot materially improve it today through
local scheduler fusion, scatter-reduce, cooperative split-K, algebraic
elimination, or a new pattern because there is no surrounding producer or
consumer to remove and the required input read, f32 accumulation, output
store, and one GPU launch dominate; the fix is BANDWIDTH_BOUND: record this
as at-floor/not-true-floor unless this full-scope oracle meaningfully beats
tuned Inductor on the same shape."""
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

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

M = 128
N = 1000
OUTPUT_SHAPE = (N,)
OUTPUT_STRIDE = (1,)

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _sum_128x1000_dim0_kernel(
        x_ptr,
        out_ptr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, 128)
        col_mask = cols < 1000
        values = tl.load(
            x_ptr + rows[:, None] * 1000 + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        totals = tl.sum(values, axis=0)
        tl.store(out_ptr + cols, totals, mask=col_mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    tangents_1, shape_param = inputs
    if not isinstance(tangents_1, torch.Tensor):
        raise TypeError(f"expected tensor input 0, got {type(tangents_1)!r}")
    if tangents_1.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {tangents_1.dtype}")
    if tuple(tangents_1.shape) != (M, N):
        raise ValueError(f"expected input shape {(M, N)}, got {tuple(tangents_1.shape)}")
    if not tangents_1.is_cuda:
        raise ValueError("oracle_sum.py expects CUDA inputs")
    if not tangents_1.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tangents_1.stride()}")
    if not isinstance(shape_param, (list, tuple)) or tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"expected reshape target {OUTPUT_SHAPE}, got {shape_param!r}")

    return tangents_1


@oracle_impl(hardware="H100", shapes="(T([128, 1000], f32), S([1000]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
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
        raise RuntimeError("Triton is required for oracle_sum.py")

    tangents_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=tangents_1.device,
        dtype=tangents_1.dtype,
    )
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _sum_128x1000_dim0_kernel[grid](tangents_1, out)
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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
