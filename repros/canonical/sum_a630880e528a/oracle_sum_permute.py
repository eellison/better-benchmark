"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete shared pointwise producer `arg126 * arg104 * (1 - arg104)`, writes the returned `[65, 1024]` permute-view backing storage, and accumulates the sibling `[65]` column sum in one Triton kernel, whereas Inductor currently schedules the materialized permute-side producer and the sibling column reduction as generic separate work over the same intermediate; Inductor cannot do this today because its scheduler does not fuse a layout-changing side output with a compatible sibling reduction consumer into one multi-output producer; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to keep small column reductions in the same fused kernel as a required permuted materialization when both outputs consume the same pointwise expression."""
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


ROWS = 1024
COLS = 65


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _sum_permute_kernel(
        arg104_ptr,
        arg126_ptr,
        out_base_ptr,
        out_sum_ptr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        col = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        offsets = rows * N + col

        a = tl.load(arg104_ptr + offsets).to(tl.float32)
        b = tl.load(arg126_ptr + offsets).to(tl.float32)
        value = b * a * (1.0 - a)

        tl.store(out_base_ptr + offsets, value)
        tl.store(out_sum_ptr + col, tl.sum(value, axis=0))


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 3:
        raise ValueError(f"expected 3 inputs, got {len(inputs)}")

    arg104_1, arg126_1, shape_param = inputs
    if not isinstance(arg104_1, torch.Tensor) or not isinstance(arg126_1, torch.Tensor):
        raise TypeError("first two repro inputs must be tensors")
    if arg104_1.device.type != "cuda" or arg126_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(arg104_1.shape) != (ROWS, COLS) or tuple(arg126_1.shape) != (ROWS, COLS):
        raise ValueError(
            f"unexpected tensor shapes: arg104={tuple(arg104_1.shape)} "
            f"arg126={tuple(arg126_1.shape)}"
        )
    if arg104_1.dtype != torch.float32 or arg126_1.dtype != torch.float32:
        raise TypeError("expected float32 tensor inputs")
    if not arg104_1.is_contiguous() or not arg126_1.is_contiguous():
        raise ValueError("oracle expects contiguous captured inputs")
    if list(shape_param) != [COLS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    return arg104_1, arg126_1


@oracle_impl(hardware="H100", shapes="(T([1024, 65], f32), T([1024, 65], f32), S([65]))")
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
    arg104_1, arg126_1 = _validate_inputs(inputs)
    out_base = torch.empty((ROWS, COLS), device=arg104_1.device, dtype=torch.float32)
    out_sum = torch.empty((COLS,), device=arg104_1.device, dtype=torch.float32)

    _sum_permute_kernel[(COLS,)](
        arg104_1,
        arg126_1,
        out_base,
        out_sum,
        N=COLS,
        BLOCK_M=ROWS,
        num_warps=8,
    )
    return out_base.permute(1, 0), out_sum


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
