"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete SELU-style pointwise producer once, writes the returned f32[512,1024] permute-view backing storage, and accumulates the sibling f32[512] column sum in one Triton kernel, whereas Inductor currently schedules the materialized permute-side producer and the reduction consumer as separate generic work over the same f32[1024,512] expression; Inductor cannot do this today because its scheduler does not fuse a layout-changing side output with a compatible sibling reduction over the same pointwise producer; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a multi-output pointwise-plus-column-reduction kernel when a required permuted materialization and a reduction consume the same producer."""
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


ROWS = 1024
COLS = 512


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 1}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 2}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=3),
        ],
        key=["N"],
    )
    @triton.jit
    def _selu_sum_permute_kernel(
        arg6_ptr,
        mm_ptr,
        out_base_ptr,
        sum_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_start = tl.program_id(0) * BLOCK_N
        rows = tl.arange(0, M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        valid = (rows < M) & (cols < N)
        offsets = rows * N + cols

        arg6 = tl.load(arg6_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        mm = tl.load(mm_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        neg = 1.7580993408473766 * mm * tl.exp(arg6)
        pos = 1.0507009873554805 * mm
        values = tl.where(arg6 <= 0.0, neg, pos)

        tl.store(out_base_ptr + offsets, values, mask=valid)
        col_sum = tl.sum(tl.where(valid, values, 0.0), axis=0)
        sum_cols = col_start + tl.arange(0, BLOCK_N)
        tl.store(sum_ptr + sum_cols, col_sum, mask=sum_cols < N)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_selu_sum_permute.py")
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    arg6_1, mm_8, shape_param = inputs
    for name, value in (("arg6_1", arg6_1), ("mm_8", mm_8)):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
        if value.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensor inputs")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
        if tuple(value.shape) != (ROWS, COLS):
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(ROWS, COLS)}")
        if tuple(value.stride()) != (COLS, 1):
            raise ValueError(f"{name} has stride {value.stride()}, expected {(COLS, 1)}")
    if arg6_1.device != mm_8.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if list(shape_param) != [COLS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    return arg6_1, mm_8


@oracle_impl(hardware="H100", shapes="(T([1024, 512], f32), T([1024, 512], f32), S([512]))")
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
    arg6_1, mm_8 = _validate_inputs(inputs)
    out_base = torch.empty((ROWS, COLS), device=arg6_1.device, dtype=torch.float32)
    col_sum = torch.empty((COLS,), device=arg6_1.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_N"]),)
    _selu_sum_permute_kernel[grid](
        arg6_1,
        mm_8,
        out_base,
        col_sum,
        M=ROWS,
        N=COLS,
    )
    return out_base.permute(1, 0), col_sum


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
