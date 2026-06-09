"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full permute.clone._unsafe_view.view chain as one 192-wide Triton row copy from contiguous [B,H,S,D] storage into the final contiguous [B*S,H*D] output, whereas Inductor currently lowers this clone/view scope as a generic elementwise layout-copy kernel with scalar logical-index decoding for the permuted dimensions; Inductor cannot do this today because its pointwise scheduler/codegen lacks a dedicated permute-contiguous-flatten layout-copy pattern that vectorizes the static 64-wide head dimension and writes final storage order directly; the fix is NEW_PATTERN: add a specialized layout-copy lowering for permute(0,2,1,3).clone().view(B*S,H*D) that emits vectorized row stores over the final contiguous output."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_ROWS", "SEQ"],
    )
    @triton.jit
    def _permute_clone_view_kernel(
        input_ptr,
        output_ptr,
        TOTAL_ROWS: tl.constexpr,
        SEQ: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        dim = tl.arange(0, BLOCK_D)
        row_mask = rows[:, None] < TOTAL_ROWS

        batch = rows // SEQ
        seq_idx = rows - batch * SEQ

        input_base = batch[:, None] * (3 * SEQ * BLOCK_D) + seq_idx[:, None] * BLOCK_D + dim[None, :]
        output_base = rows[:, None] * (3 * BLOCK_D) + dim[None, :]

        head0 = tl.load(input_ptr + input_base, mask=row_mask, other=0.0)
        head1 = tl.load(input_ptr + input_base + SEQ * BLOCK_D, mask=row_mask, other=0.0)
        head2 = tl.load(input_ptr + input_base + 2 * SEQ * BLOCK_D, mask=row_mask, other=0.0)

        tl.store(output_ptr + output_base, head0, mask=row_mask)
        tl.store(output_ptr + output_base + BLOCK_D, head1, mask=row_mask)
        tl.store(output_ptr + output_base + 2 * BLOCK_D, head2, mask=row_mask)


def _shape_tuple(value, name):
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(inputs):
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x, shape_param = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected arg84_1 to be a tensor, got {type(x)!r}")
    if x.ndim != 4:
        raise ValueError(f"expected rank-4 input, got shape={tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise ValueError(f"expected f32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_layout_clone.py expects CUDA inputs")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")

    batch, heads, seq, dim = (int(size) for size in x.shape)
    if heads != 3 or dim != 64:
        raise ValueError(f"expected captured [B,3,S,64] layout, got shape={tuple(x.shape)}")

    hidden = heads * dim
    total_rows = batch * seq
    expected_shape = (total_rows, hidden)
    actual_shape = _shape_tuple(shape_param, "_shape_param_0")
    if actual_shape != expected_shape:
        raise ValueError(f"unexpected shape parameter {actual_shape}, expected {expected_shape}")

    return x, total_rows, seq, hidden, dim


@oracle_impl(hardware="H100", shapes="(T([128, 3, 197, 64], f32), S([25216, 192]))")
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

    x, total_rows, seq, hidden, dim = _validate_inputs(tuple(inputs))
    output = torch.empty_strided(
        (total_rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda META: (triton.cdiv(total_rows, META["BLOCK_ROWS"]),)
    _permute_clone_view_kernel[grid](
        x,
        output,
        TOTAL_ROWS=total_rows,
        SEQ=seq,
        BLOCK_D=dim,
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
