"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full view-to-view scaled stable softmax in one shape-specialized Triton row kernel, whereas Inductor currently emits a generic fused reduction/pointwise kernel for the amax, exp, sum, and normalize chain; Inductor cannot do this today because its scheduler/codegen does not recognize the paired amax/sum softmax idiom with an intervening scale as a dedicated small-last-dimension template; the fix is NEW_PATTERN: add a scaled softmax pattern that lowers this view-preserving reduction chain to a specialized row-wise kernel."""
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


INPUT_SHAPE = (256, 32, 33)
VIEW_SHAPE = (32, 8, 32, 33)
NUM_ROWS = 256 * 32
NUM_COLS = 33
BLOCK_COLS = 64
INV_TEMPERATURE_LOG2E = 0.18033688011112042  # log2(e) / 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value):
    return tuple(int(dim) for dim in value)


def _validate_scope(bmm_14, shape_param_0, shape_param_1, shape_param_2):
    if tuple(bmm_14.shape) != INPUT_SHAPE:
        raise ValueError(f"expected input shape {INPUT_SHAPE}, got {tuple(bmm_14.shape)}")
    if bmm_14.dtype != torch.float32:
        raise TypeError(f"expected input dtype torch.float32, got {bmm_14.dtype}")
    if _shape_tuple(shape_param_0) != VIEW_SHAPE:
        raise ValueError(f"expected first view shape {VIEW_SHAPE}, got {_shape_tuple(shape_param_0)}")
    if _shape_tuple(shape_param_1) != VIEW_SHAPE:
        raise ValueError(f"expected expand shape {VIEW_SHAPE}, got {_shape_tuple(shape_param_1)}")
    if _shape_tuple(shape_param_2) != INPUT_SHAPE:
        raise ValueError(f"expected final view shape {INPUT_SHAPE}, got {_shape_tuple(shape_param_2)}")


def _aten_fallback(bmm_14, shape_param_0, shape_param_1, shape_param_2):
    view_default = torch.ops.aten.view.default(bmm_14, shape_param_0)
    mul_tensor = torch.ops.aten.mul.Tensor(view_default, 1)
    amax_default = torch.ops.aten.amax.default(mul_tensor, [-1], True)
    sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, amax_default)
    div_tensor = torch.ops.aten.div.Tensor(sub_tensor, 8.0)
    exp_default = torch.ops.aten.exp.default(div_tensor)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True)
    div_tensor_1 = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list)
    expand_default = torch.ops.aten.expand.default(div_tensor_1, shape_param_1)
    return torch.ops.aten.view.default(expand_default, shape_param_2)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=1, num_stages=4),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=2, num_stages=4),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=4, num_stages=4),
        ],
        key=[],
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        n_rows: tl.constexpr,
        n_cols: tl.constexpr,
        block_cols: tl.constexpr,
        inv_temperature_log2e: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        col_offsets = tl.arange(0, block_cols)
        mask = (row_offsets[:, None] < n_rows) & (col_offsets[None, :] < n_cols)
        offsets = row_offsets[:, None] * n_cols + col_offsets[None, :]

        vals = tl.load(input_ptr + offsets, mask=mask, other=-float("inf")).to(tl.float32)
        row_max = tl.max(vals, axis=1)
        numer = tl.exp2((vals - row_max[:, None]) * inv_temperature_log2e)
        denom = tl.sum(numer, axis=1)
        out_vals = numer / denom[:, None]
        tl.store(output_ptr + offsets, out_vals, mask=mask)


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
    bmm_14, shape_param_0, shape_param_1, shape_param_2 = inputs
    _validate_scope(bmm_14, shape_param_0, shape_param_1, shape_param_2)

    if triton is None or not bmm_14.is_cuda or not bmm_14.is_contiguous():
        return _aten_fallback(bmm_14, shape_param_0, shape_param_1, shape_param_2)

    output = torch.empty_like(bmm_14)
    grid = lambda meta: (triton.cdiv(NUM_ROWS, meta["BLOCK_ROWS"]),)
    oracle_kernel[grid](
        bmm_14,
        output,
        n_rows=NUM_ROWS,
        n_cols=NUM_COLS,
        block_cols=BLOCK_COLS,
        inv_temperature_log2e=INV_TEMPERATURE_LOG2E,
    )
    return output


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
