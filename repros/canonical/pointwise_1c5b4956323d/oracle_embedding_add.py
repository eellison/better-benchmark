"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete captured unsqueeze-plus-two embedding scope as a shape-specialized Triton gather that folds the metadata-only `[1024] -> [1,1024]` view, int64 index add, and f32 table load into one fresh contiguous `[1,1024,H]` output, whereas Inductor fuses the same ops but lowers them through generic pointwise indirect-indexing code with per-element embedding bounds asserts and generic scheduling overhead; Inductor cannot emit this lean form today because it has no guarded constant-offset embedding-gather template or range-proof path that can replace the generic assert-heavy indirect-index lowering for known-valid token ids; the fix is NEW_PATTERN: add an embedding-plus-constant-offset gather template that proves or cheaply guards index bounds once, then emits direct dense gather/copy code for this layout."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


SEQ_LEN = 1024
VOCAB = 1026
INDEX_OFFSET = 2


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
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["HIDDEN"],
    )
    @triton.jit
    def _embedding_add_kernel(
        ids_ptr,
        table_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        HIDDEN: tl.constexpr,
        ROW_OFFSET: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        token = offsets // HIDDEN
        col = offsets - token * HIDDEN
        row = tl.load(ids_ptr + token, mask=mask, other=0) + ROW_OFFSET
        values = tl.load(table_ptr + row * HIDDEN + col, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, values, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_embedding_add.py")
    if len(inputs) != 2:
        raise ValueError(f"expected 2 inputs, got {len(inputs)}")

    ids, table = inputs
    if not isinstance(ids, torch.Tensor) or not isinstance(table, torch.Tensor):
        raise TypeError("oracle expects tensor inputs")
    if ids.device.type != "cuda" or table.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if ids.dtype != torch.int64:
        raise TypeError(f"expected int64 ids, got {ids.dtype}")
    if table.dtype != torch.float32:
        raise TypeError(f"expected fp32 embedding table, got {table.dtype}")
    if tuple(ids.shape) != (SEQ_LEN,):
        raise ValueError(f"unexpected ids shape: {tuple(ids.shape)}")
    if table.ndim != 2 or table.shape[0] != VOCAB:
        raise ValueError(f"unexpected embedding table shape: {tuple(table.shape)}")
    if not ids.is_contiguous() or not table.is_contiguous():
        raise ValueError("oracle expects contiguous captured inputs")

    return ids, table


def oracle_forward(inputs):
    """Run the full Repro.forward scope: unsqueeze, add-two, embedding gather."""
    ids, table = _validate_inputs(inputs)
    hidden = table.shape[1]
    total = SEQ_LEN * hidden
    output = torch.empty_strided(
        (1, SEQ_LEN, hidden),
        (total, hidden, 1),
        device=table.device,
        dtype=table.dtype,
    )
    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    _embedding_add_kernel[grid](
        ids,
        table,
        output,
        TOTAL=total,
        HIDDEN=hidden,
        ROW_OFFSET=INDEX_OFFSET,
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
