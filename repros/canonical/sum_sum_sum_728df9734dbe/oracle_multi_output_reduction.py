"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full GPT-J layernorm-backward multi-output region with row-local reductions feeding the transpose side output and three column reductions, whereas Inductor schedules the row reductions, transpose materialization, and later vector reductions as separate generic kernels; Inductor cannot do this today because its scheduler does not build a dependent multi-output reduction plan that reuses row-reduction scalars while materializing a side-layout output and reducing it by column; the fix is SCHEDULER_FUSION: teach Inductor to form a coordinated two-phase layernorm-backward reduction template with fused side-output stores and sibling column reductions."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


N_ROWS = 128
N_COLS = 4096


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _layernorm_side_kernel(
        mm_ptr,
        weight_ptr,
        yhat_ptr,
        scale_ptr,
        side_base_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)

        x = tl.load(mm_ptr + row * N + cols).to(tl.float32)
        weight = tl.load(weight_ptr + cols).to(tl.float32)
        yhat = tl.load(yhat_ptr + row * N + cols).to(tl.float32)
        scale = tl.load(scale_ptr + row).to(tl.float32)

        weighted = x * weight
        row_sum = tl.sum(weighted, axis=0)
        row_dot = tl.sum(weighted * yhat, axis=0)
        side = scale * (weighted * N - row_sum - yhat * row_dot)

        tl.store(side_base_ptr + row * N + cols, side)

    @triton.jit
    def _column_reduction_kernel(
        mm_ptr,
        yhat_ptr,
        side_base_ptr,
        out_xyhat_ptr,
        out_x_ptr,
        out_side_ptr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)[:, None]
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        offsets = rows * N + cols
        x = tl.load(mm_ptr + offsets).to(tl.float32)
        yhat = tl.load(yhat_ptr + offsets).to(tl.float32)
        side = tl.load(side_base_ptr + offsets).to(tl.float32)

        sum_xyhat = tl.sum(x * yhat, axis=0)
        sum_x = tl.sum(x, axis=0)
        sum_side = tl.sum(side, axis=0)

        out_cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        tl.store(out_xyhat_ptr + out_cols, sum_xyhat)
        tl.store(out_x_ptr + out_cols, sum_x)
        tl.store(out_side_ptr + out_cols, sum_side)


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

    mm, weight, yhat, scale, *_shape_params = inputs

    out_xyhat = torch.empty((N_COLS,), device=mm.device, dtype=torch.float32)
    out_x = torch.empty((N_COLS,), device=mm.device, dtype=torch.float32)
    side_base = torch.empty((N_ROWS, N_COLS), device=mm.device, dtype=torch.float32)
    out_side = torch.empty((N_COLS,), device=mm.device, dtype=torch.float32)

    _layernorm_side_kernel[(N_ROWS,)](
        mm,
        weight,
        yhat,
        scale,
        side_base,
        N=N_COLS,
        BLOCK_N=N_COLS,
        num_warps=8,
    )
    _column_reduction_kernel[(triton.cdiv(N_COLS, 16),)](
        mm,
        yhat,
        side_base,
        out_xyhat,
        out_x,
        out_side,
        N=N_COLS,
        BLOCK_M=N_ROWS,
        BLOCK_N=16,
        num_warps=8,
    )
    return (out_xyhat, out_x, side_base.permute(1, 0), out_side)


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
