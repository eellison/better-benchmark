"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete generated-position embedding scope by folding the iota, expand, constant add, and embedding gather into one shape-specialized Triton broadcast-copy kernel that writes the fresh contiguous [B,512,H] output, whereas Inductor lowers the same semantic operation through generic indirect-index embedding code for a generated index tensor; Inductor cannot emit this lean form today because it lacks a guarded generated-position embedding template that proves the constant-offset row range and reuses source row tiles across the batch broadcast; the fix is NEW_PATTERN: add an embedding-plus-constant-position lowering that bypasses per-element index construction/assert overhead and writes the final layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

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

SEQ_LEN = 512
ROW_OFFSET = 2


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256, "BLOCK_B": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512, "BLOCK_B": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024, "BLOCK_B": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512, "BLOCK_B": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024, "BLOCK_B": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512, "BLOCK_B": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 1024, "BLOCK_B": 4}, num_warps=8, num_stages=3),
        ],
        key=["SOURCE_TOTAL", "BATCH", "HIDDEN"],
    )
    @triton.jit
    def _generated_position_embedding_kernel(
        table_ptr,
        out_ptr,
        SOURCE_TOTAL: tl.constexpr,
        BATCH: tl.constexpr,
        HIDDEN: tl.constexpr,
        OFFSET: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_B: tl.constexpr,
    ):
        source_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        source_mask = source_offsets < SOURCE_TOTAL
        token = source_offsets // HIDDEN
        col = source_offsets - token * HIDDEN
        values = tl.load(
            table_ptr + (token + OFFSET) * HIDDEN + col,
            mask=source_mask,
            other=0.0,
        )

        batch_offsets = tl.program_id(1) * BLOCK_B + tl.arange(0, BLOCK_B)
        out_offsets = batch_offsets[:, None] * SOURCE_TOTAL + source_offsets[None, :]
        out_mask = (batch_offsets[:, None] < BATCH) & source_mask[None, :]
        tl.store(out_ptr + out_offsets, values[None, :], mask=out_mask)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, int, int]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_generated_position_embedding.py")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    table, shape_param = inputs
    if not isinstance(table, torch.Tensor):
        raise TypeError("first repro input must be the embedding table tensor")
    if table.device.type != "cuda":
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if table.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {table.dtype}")
    if table.ndim != 2:
        raise ValueError(f"expected rank-2 embedding table, got shape={tuple(table.shape)}")
    if not table.is_contiguous():
        raise ValueError(f"embedding table must be contiguous, got stride={table.stride()}")
    if table.shape[0] < SEQ_LEN + ROW_OFFSET:
        raise ValueError(
            f"embedding table has too few rows: got={table.shape[0]} "
            f"required={SEQ_LEN + ROW_OFFSET}"
        )

    if not isinstance(shape_param, (list, tuple)) or len(shape_param) != 2:
        raise TypeError(f"shape parameter must be a length-2 list/tuple, got {shape_param!r}")
    batch = int(shape_param[0])
    shape_seq = int(shape_param[1])
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if shape_seq not in (-1, SEQ_LEN):
        raise ValueError(f"unexpected sequence shape parameter: {shape_param!r}")

    return table, batch, int(table.shape[1])


@oracle_impl(hardware="H100", shapes="(T([1026, 768], f32), S([4, -1]))")
def oracle_forward(inputs):
    """Run the full iota/expand/add/embedding scope from Repro.forward."""
    table, batch, hidden = _validate_inputs(inputs)
    source_total = SEQ_LEN * hidden
    output = torch.empty_strided(
        (batch, SEQ_LEN, hidden),
        (source_total, hidden, 1),
        device=table.device,
        dtype=table.dtype,
    )
    grid = lambda meta: (
        triton.cdiv(source_total, meta["BLOCK_N"]),
        triton.cdiv(batch, meta["BLOCK_B"]),
    )
    _generated_position_embedding_kernel[grid](
        table,
        output,
        SOURCE_TOTAL=source_total,
        BATCH=batch,
        HIDDEN=hidden,
        OFFSET=ROW_OFFSET,
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
