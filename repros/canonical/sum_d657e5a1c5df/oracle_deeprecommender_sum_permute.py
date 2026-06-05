"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeepRecommender gated SELU-style pointwise producer once, writes the returned f32[1024,1024] permute-view backing storage, and accumulates the sibling f32[1024] column sum in one Triton kernel, whereas Inductor currently schedules the required permuted side output and the reduction consumer as generic work over the same fused expression; Inductor cannot do this today because its scheduler does not fuse a layout-changing materialized side output with a compatible sibling reduction over that producer; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to emit a multi-output pointwise-plus-column-reduction kernel when a required permuted materialization and a reduction consume the same pointwise expression."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


ROWS = 1024
COLS = 1024


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
    def _deeprecommender_sum_permute_kernel(
        arg11_ptr,
        mm_ptr,
        arg10_ptr,
        out_base_ptr,
        out_sum_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        col_start = tl.program_id(0) * BLOCK_N
        rows = tl.arange(0, M)[:, None]
        cols = col_start + tl.arange(0, BLOCK_N)[None, :]
        valid = cols < N
        offsets = rows * N + cols

        gate = tl.load(arg11_ptr + offsets, mask=valid, other=0).to(tl.float32)
        mm = tl.load(mm_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
        arg10 = tl.load(arg10_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

        base = mm * gate * 5.000000000000001
        neg = base * 1.7580993408473766 * tl.exp(arg10)
        pos = base * 1.0507009873554805
        values = tl.where(arg10 <= 0.0, neg, pos)

        tl.store(out_base_ptr + offsets, values, mask=valid)
        col_sum = tl.sum(tl.where(valid, values, 0.0), axis=0)
        sum_cols = col_start + tl.arange(0, BLOCK_N)
        tl.store(out_sum_ptr + sum_cols, col_sum, mask=sum_cols < N)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_deeprecommender_sum_permute.py")
    if len(inputs) != 4:
        raise ValueError(f"{REPRO_ID} expects 4 inputs, got {len(inputs)}")

    arg11_1, mm_4, arg10_1, shape_param = inputs
    tensor_specs = (
        ("arg11_1", arg11_1, torch.bool),
        ("mm_4", mm_4, torch.float32),
        ("arg10_1", arg10_1, torch.float32),
    )
    for name, value, dtype in tensor_specs:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
        if value.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA tensor inputs")
        if value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype}, got {value.dtype}")
        if tuple(value.shape) != (ROWS, COLS):
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {(ROWS, COLS)}")
        if tuple(value.stride()) != (COLS, 1):
            raise ValueError(f"{name} has stride {value.stride()}, expected {(COLS, 1)}")
    if arg11_1.device != mm_4.device or arg11_1.device != arg10_1.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if list(shape_param) != [COLS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    return arg11_1, mm_4, arg10_1


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
    arg11_1, mm_4, arg10_1 = _validate_inputs(inputs)
    out_base = torch.empty((ROWS, COLS), device=mm_4.device, dtype=torch.float32)
    out_sum = torch.empty((COLS,), device=mm_4.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_N"]),)
    _deeprecommender_sum_permute_kernel[grid](
        arg11_1,
        mm_4,
        arg10_1,
        out_base,
        out_sum,
        M=ROWS,
        N=COLS,
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
