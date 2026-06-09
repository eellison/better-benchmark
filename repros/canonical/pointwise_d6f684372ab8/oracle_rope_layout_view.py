"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full rope layout materialization from Repro.forward, viewing the contiguous `float32[32,32,8,32,2]` input as `[32,32,8,64]`, applying the `[0,2,1,3]` permute, preserving the no-op expand, cloning into contiguous `[32,8,32,64]` storage, and returning the final contiguous `[256,32,64]` view with the same values, dtype, shape, and stride, whereas Inductor already lowers this pure layout materialization to the same required read/reordered-write traffic envelope; Inductor cannot remove the copy because the returned tensor is the newly materialized contiguous clone after a real dimension reorder, so the fix is BANDWIDTH_BOUND: treat this as an at-floor layout-copy case unless generic copy scheduling or launch overhead improves."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 1}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 2}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 4}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
        ],
        key=["N_ROWS"],  # re-tune when N_ROWS changes
    )
    @triton.jit
    def oracle_kernel(
        input_ptr,
        output_ptr,
        N_ROWS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        """Materialize view([32,32,8,64]).permute(0,2,1,3).contiguous()."""
        pid = tl.program_id(0)
        rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, 64)

        # Final output is [256,32,64], row = first_dim * 32 + c.
        first_dim = rows // 32
        c = rows - first_dim * 32
        a = first_dim // 8
        b = first_dim - a * 8

        src = a[:, None] * 16384 + c[:, None] * 512 + b[:, None] * 64 + cols[None, :]
        dst = rows[:, None] * 64 + cols[None, :]
        mask = rows[:, None] < N_ROWS
        values = tl.load(input_ptr + src, mask=mask, other=0.0)
        tl.store(output_ptr + dst, values, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([32, 32, 8, 32, 2], f32), S([32, 32, 8, 64]), S([32, 8, 32, 64]), S([256, 32, 64]))")
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
    x, _shape_param_0, _shape_param_1, _shape_param_2 = inputs

    if triton is None or not x.is_cuda:
        view_default = torch.ops.aten.view.default(x, _shape_param_0)
        permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3])
        expand_default = torch.ops.aten.expand.default(permute_default, _shape_param_1)
        clone_default = torch.ops.aten.clone.default(
            expand_default, memory_format=torch.contiguous_format
        )
        return torch.ops.aten.view.default(clone_default, _shape_param_2)

    output = torch.empty(tuple(_shape_param_2), device=x.device, dtype=x.dtype)
    n_rows = 256 * 32
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_ROWS"]),)
    oracle_kernel[grid](x, output, N_ROWS=n_rows)
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
