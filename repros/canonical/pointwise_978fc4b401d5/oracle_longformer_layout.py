"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Longformer head/sequence layout materialization by lowering the captured view/permute/view/permute/clone/view chain to one Triton tiled copy that writes the final contiguous `[4096, 768]` output directly, whereas Inductor currently lowers the same layout-only graph as a generic pointwise clone-permute copy with more general index decoding for the reshape/transpose pattern; Inductor cannot do this today because its scheduler/codegen does not recognize this fixed head-major to sequence-major layout-copy pattern and emit a 64-wide vectorized copy over the head dimension; the fix is NEW_PATTERN: add a guarded layout-copy lowering for `[A, B, C, D] -> [B*C, A*D]` materializations with contiguous `D` lanes."""
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
            triton.Config({"BLOCK_ROWS": 4, "BLOCK_HEADS": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_HEADS": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_ROWS": 8, "BLOCK_HEADS": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16, "BLOCK_HEADS": 4}, num_warps=8, num_stages=3),
        ],
        key=["ROWS", "HEADS", "D"],
    )
    @triton.jit
    def _longformer_layout_kernel(
        input_ptr,
        output_ptr,
        ROWS: tl.constexpr,
        HEADS: tl.constexpr,
        D: tl.constexpr,
        INPUT_HEAD_STRIDE: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        head_lane = tl.program_id(1) * (BLOCK_HEADS * D) + tl.arange(0, BLOCK_HEADS * D)
        head = head_lane // D
        lane = head_lane - head * D

        mask = (row[:, None] < ROWS) & (head[None, :] < HEADS)
        input_offsets = head[None, :] * INPUT_HEAD_STRIDE + row[:, None] * D + lane[None, :]
        output_offsets = row[:, None] * HIDDEN + head[None, :] * D + lane[None, :]

        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _expect_shape_param(name: str, actual, expected: tuple[int, ...]) -> None:
    if tuple(actual) != expected:
        raise ValueError(f"unexpected {name}: got {actual}, expected {list(expected)}")


def _parse_layout(inputs):
    x, shape0, shape1, shape2, shape3, shape4 = inputs
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input, got shape={tuple(x.shape)}")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(x.stride())}")

    heads, batch_chunks, seq, singleton, dim = (int(v) for v in shape0)
    if singleton != 1:
        raise ValueError(f"expected singleton layout dimension, got {singleton}")

    rows = batch_chunks * seq
    hidden = heads * dim
    _expect_shape_param("shape1", shape1, (heads, batch_chunks, seq, dim))
    _expect_shape_param("shape2", shape2, (1, heads, rows, dim))
    _expect_shape_param("shape3", shape3, (rows, 1, hidden))
    _expect_shape_param("shape4", shape4, (rows, hidden))

    expected_input = (heads * batch_chunks, seq, dim)
    if tuple(x.shape) != expected_input:
        raise ValueError(f"unexpected input shape {tuple(x.shape)}, expected {expected_input}")

    return x, rows, heads, dim, hidden


def _check_output_layout(output: torch.Tensor, rows: int, hidden: int) -> bool:
    return (
        tuple(output.shape) == (rows, hidden)
        and tuple(output.stride()) == (hidden, 1)
        and output.storage_offset() == 0
    )


@oracle_impl(hardware="H100", shapes="(T([192, 256, 64], f16), S([12, 16, 256, 1, 64]), S([12, 16, 256, 64]), S([1, 12, 4096, 64]), S([4096, 1, 768]), S([4096, 768]))")
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

    x, rows, heads, dim, hidden = _parse_layout(inputs)
    output = torch.empty_strided(
        (rows, hidden),
        (hidden, 1),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda META: (
        triton.cdiv(rows, META["BLOCK_ROWS"]),
        triton.cdiv(heads, META["BLOCK_HEADS"]),
    )
    _longformer_layout_kernel[grid](
        x,
        output,
        ROWS=rows,
        HEADS=heads,
        D=dim,
        INPUT_HEAD_STRIDE=rows * dim,
        HIDDEN=hidden,
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
        x, rows, heads, dim, hidden = _parse_layout(inputs)
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_output_layout(layout_output, rows, hidden)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={list(layout_output.stride())})"
        )
        ok = ok and layout_ok
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
