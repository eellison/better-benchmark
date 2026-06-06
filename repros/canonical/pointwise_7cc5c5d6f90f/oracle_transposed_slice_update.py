"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete slice/update/copy_ scope by writing the transposed `arg0` tile into `arg1[:, :32]` and returning the mutated `arg1` storage, but under the required CUDAGraph/interleaved harness Inductor is already within measurement noise of this direct narrow update; Inductor does not need a scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recomputation, or new-pattern fix for this isolated repro because the remaining cost is the small required write plus launch overhead, so the fix is BANDWIDTH_BOUND: record this repro as at floor."""
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


ROWS = 128
COLS = 32000
UPDATE_COLS = 32


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1, "BLOCK_COLS": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2, "BLOCK_COLS": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_COLS": 32}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_COLS": 32}, num_warps=4, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _transposed_slice_update_kernel(
        base_ptr,
        update_ptr,
        N_ROWS: tl.constexpr,
        N_COLS: tl.constexpr,
        N_UPDATE_COLS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)[:, None]
        col_offsets = tl.arange(0, BLOCK_COLS)[None, :]
        mask = (row_offsets < N_ROWS) & (col_offsets < N_UPDATE_COLS)

        values = tl.load(update_ptr + col_offsets * N_ROWS + row_offsets, mask=mask, other=0.0)
        tl.store(base_ptr + row_offsets * N_COLS + col_offsets, values, mask=mask)


def _validate_inputs(inputs) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_transposed_slice_update.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects two inputs, got {len(inputs)}")

    arg1_1, arg0_1 = inputs
    if not isinstance(arg1_1, torch.Tensor) or not isinstance(arg0_1, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor inputs")
    if arg1_1.device.type != "cuda" or arg0_1.device.type != "cuda":
        raise ValueError(f"{REPRO_ID} expects CUDA tensor inputs")
    if arg1_1.dtype != torch.float32 or arg0_1.dtype != torch.float32:
        raise TypeError(f"{REPRO_ID} expects float32 inputs")
    if tuple(arg1_1.shape) != (ROWS, COLS):
        raise ValueError(f"unexpected base shape {tuple(arg1_1.shape)}")
    if tuple(arg0_1.shape) != (UPDATE_COLS, ROWS):
        raise ValueError(f"unexpected update shape {tuple(arg0_1.shape)}")
    if tuple(arg1_1.stride()) != (COLS, 1):
        raise ValueError(f"unexpected base stride {tuple(arg1_1.stride())}")
    if tuple(arg0_1.stride()) != (ROWS, 1):
        raise ValueError(f"unexpected update stride {tuple(arg0_1.stride())}")
    return arg1_1, arg0_1


def oracle_forward(inputs):
    """Run the full Repro.forward scope with a narrow transposed slice update."""
    arg1_1, arg0_1 = _validate_inputs(inputs)
    grid = lambda meta: (triton.cdiv(ROWS, meta["BLOCK_ROWS"]),)
    _transposed_slice_update_kernel[grid](
        arg1_1,
        arg0_1,
        N_ROWS=ROWS,
        N_COLS=COLS,
        N_UPDATE_COLS=UPDATE_COLS,
    )
    return arg1_1


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
