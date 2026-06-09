"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Repro.forward scope in one Triton column-tiled multi-output reduction kernel, writing both backing tensors for the returned transpose views while accumulating the `[1]` and `[1024]` sums, whereas Inductor schedules the masked row vector, broadcasted masked product side view, and sibling reductions as generic pointwise/reduction regions with extra launch and materialization overhead; Inductor cannot do this today because its scheduler does not fuse small broadcast producers, required side-view materialization, and sibling scalar/column reductions into one full-scope multi-output reduction template; the fix is SCHEDULER_FUSION: teach the scheduler/codegen to keep the broadcasted producer virtual across required transpose-view stores and co-accumulate all sibling reductions in the same generated kernel."""
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

ROWS = 2048
COLS = 1024
BLOCK_ROWS = 2048
BLOCK_COLS = 16
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_scope_kernel(
        mask_ptr,
        row_ptr,
        col_ptr,
        gate_ptr,
        row_base_out_ptr,
        row_sum_out_ptr,
        product_base_out_ptr,
        col_sum_out_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_COLS: tl.constexpr,
    ):
        col_tile = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        col_mask = cols < N_COLS

        row_mask = tl.load(mask_ptr + rows).to(tl.int1)
        row_values = tl.load(row_ptr + rows).to(tl.float32)
        masked_rows = tl.where(row_mask, 0.0, row_values)

        if col_tile == 0:
            tl.store(row_base_out_ptr + rows, masked_rows)
            tl.store(row_sum_out_ptr, tl.sum(masked_rows, axis=0))

        col_values = tl.load(col_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        gate = tl.load(
            gate_ptr + rows[:, None] * N_COLS + cols[None, :],
            mask=col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        product = masked_rows[:, None] * col_values[None, :]
        product = tl.where(gate <= 0.0, 0.0, product)

        tl.store(
            product_base_out_ptr + rows[:, None] * N_COLS + cols[None, :],
            product,
            mask=col_mask[None, :],
        )
        tl.store(col_sum_out_ptr + cols, tl.sum(product, axis=0), mask=col_mask)


def _shape_tuple(shape_param):
    return tuple(int(dim) for dim in shape_param)


def _validate_inputs(inputs) -> None:
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 5:
        raise ValueError(f"expected 5 Repro.forward inputs, got {len(inputs)}")

    arg52_1, arg54_1, arg13_1, arg51_1, shape_param_0 = inputs
    expected = {
        "arg52_1": (arg52_1, torch.bool, (ROWS, 1)),
        "arg54_1": (arg54_1, torch.float32, (ROWS, 1)),
        "arg13_1": (arg13_1, torch.float32, (1, COLS)),
        "arg51_1": (arg51_1, torch.float32, (ROWS, COLS)),
    }
    for name, (tensor, dtype, shape) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} expected dtype={dtype} shape={list(shape)}, "
                f"got dtype={tensor.dtype} shape={list(tensor.shape)}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the canonical oracle")

    if _shape_tuple(shape_param_0) != (COLS,):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0}")


@oracle_impl(hardware="H100", shapes="(T([2048, 1], b8), T([2048, 1], f32), T([1, 1024], f32), T([2048, 1024], f32), S([1024]))")
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
    _validate_inputs(inputs)
    arg52_1, arg54_1, arg13_1, arg51_1, shape_param_0 = inputs

    row_base = torch.empty((ROWS, 1), device=arg54_1.device, dtype=torch.float32)
    row_sum_base = torch.empty((1, 1), device=arg54_1.device, dtype=torch.float32)
    product_base = torch.empty((ROWS, COLS), device=arg54_1.device, dtype=torch.float32)
    col_sum_base = torch.empty((1, COLS), device=arg54_1.device, dtype=torch.float32)

    _full_scope_kernel[(triton.cdiv(COLS, BLOCK_COLS),)](
        arg52_1,
        arg54_1,
        arg13_1,
        arg51_1,
        row_base,
        row_sum_base,
        product_base,
        col_sum_base,
        BLOCK_M=BLOCK_ROWS,
        BLOCK_N=BLOCK_COLS,
        N_COLS=COLS,
        num_warps=8,
    )

    return (
        row_base.permute(1, 0),
        row_sum_base.view(1),
        product_base.permute(1, 0),
        col_sum_base.view(_shape_tuple(shape_param_0)),
    )


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
