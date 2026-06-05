"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Qwen MoE routing payload by fusing bf16 row softmax and top-8 token selection in Triton, preserving the PyTorch sort boundary for the flattened token ids, and using a Triton sorted-row copy epilogue for the returned bf16 gather plus int32 sorted-index output, whereas Inductor lowers the decomposed softmax/topk/sort/index/where graph as generic reductions and library selection/sort/gather stages; Inductor cannot do this today because its scheduler/codegen pattern library has no full routing-topk lowering that combines row-softmax selection with the post-sort row-materialization contract; the fix is NEW_PATTERN: add a routing-topk pattern that emits a row softmax+topk producer and a sorted-token gather epilogue around the required sort boundary."""
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
VOCAB = 128
TOPK = 8
TOTAL_TOPK = ROWS * TOPK
HIDDEN = 2048


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _softmax_topk_kernel(
        mm_ptr,
        topk_values_ptr,
        flat_indices_ptr,
        BLOCK_N: tl.constexpr,
        K: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(mm_ptr + row * BLOCK_N + cols).to(tl.float32)

        row_max = tl.max(x, axis=0)
        numer = tl.exp(x - row_max)
        denom = tl.sum(numer, axis=0)
        probs = numer / denom

        work = x
        for k in tl.static_range(0, 8):
            best = tl.max(work, axis=0)
            # PyTorch topk's equal-value set for this shape chooses the lowest
            # token ids at the selection boundary; equal probabilities make the
            # returned values insensitive to the exact within-tie ordering.
            candidate_idx = tl.where(work == best, cols, BLOCK_N)
            best_idx = tl.min(candidate_idx, axis=0)
            best_prob = tl.sum(tl.where(cols == best_idx, probs, 0.0), axis=0)

            out_offset = row * K + k
            tl.store(topk_values_ptr + out_offset, best_prob)
            tl.store(flat_indices_ptr + out_offset, best_idx)
            work = tl.where(cols == best_idx, -float("inf"), work)

    @triton.jit
    def _cast_sorted_indices_kernel(
        sorted_i64_ptr,
        sorted_i32_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(sorted_i64_ptr + offsets, mask=mask, other=0)
        tl.store(sorted_i32_ptr + offsets, values.to(tl.int32), mask=mask)

    @triton.jit
    def _sorted_row_gather_kernel(
        view_ptr,
        sorted_i64_ptr,
        sort_positions_ptr,
        out_ptr,
        n_cols: tl.constexpr,
        k: tl.constexpr,
        valid_tokens: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        col_mask = cols < n_cols

        token = tl.load(sorted_i64_ptr + row)
        position = tl.load(sort_positions_ptr + row)
        source_row = position // k

        values = tl.load(
            view_ptr + source_row * n_cols + cols,
            mask=col_mask & (token < valid_tokens),
            other=0.0,
        )
        values = tl.where(token < valid_tokens, values, 0.0)
        tl.store(out_ptr + row * n_cols + cols, values, mask=col_mask)


def _validate_inputs(mm_19: torch.Tensor, view_72: torch.Tensor) -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not mm_19.is_cuda or not view_72.is_cuda:
        raise RuntimeError("CUDA inputs are required for this oracle")
    if mm_19.dtype != torch.bfloat16:
        raise TypeError(f"expected mm_19 dtype torch.bfloat16, got {mm_19.dtype}")
    if view_72.dtype != torch.bfloat16:
        raise TypeError(f"expected view_72 dtype torch.bfloat16, got {view_72.dtype}")
    if tuple(mm_19.shape) != (ROWS, VOCAB):
        raise ValueError(f"unexpected mm_19 shape: {tuple(mm_19.shape)}")
    if tuple(view_72.shape) != (ROWS, HIDDEN):
        raise ValueError(f"unexpected view_72 shape: {tuple(view_72.shape)}")
    if tuple(mm_19.stride()) != (VOCAB, 1):
        raise ValueError(f"unexpected mm_19 stride: {tuple(mm_19.stride())}")
    if tuple(view_72.stride()) != (HIDDEN, 1):
        raise ValueError(f"unexpected view_72 stride: {tuple(view_72.stride())}")


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
    mm_19, view_72 = inputs
    _validate_inputs(mm_19, view_72)

    topk_values = torch.empty((ROWS, TOPK), device=mm_19.device, dtype=torch.float32)
    flat_indices = torch.empty((TOTAL_TOPK,), device=mm_19.device, dtype=torch.int64)
    _softmax_topk_kernel[(ROWS,)](
        mm_19,
        topk_values,
        flat_indices,
        BLOCK_N=VOCAB,
        K=TOPK,
        num_warps=4,
    )

    sorted_indices_i64, sort_positions = torch.sort(flat_indices)
    sorted_indices = torch.empty((TOTAL_TOPK,), device=mm_19.device, dtype=torch.int32)
    where_self = torch.empty((TOTAL_TOPK, HIDDEN), device=mm_19.device, dtype=torch.bfloat16)

    _cast_sorted_indices_kernel[(triton.cdiv(TOTAL_TOPK, 1024),)](
        sorted_indices_i64,
        sorted_indices,
        n_elements=TOTAL_TOPK,
        BLOCK=1024,
        num_warps=4,
    )
    _sorted_row_gather_kernel[(TOTAL_TOPK, triton.cdiv(HIDDEN, 1024))](
        view_72,
        sorted_indices_i64,
        sort_positions,
        where_self,
        n_cols=HIDDEN,
        k=TOPK,
        valid_tokens=VOCAB,
        BLOCK_COLS=1024,
        num_warps=4,
    )
    return where_self, sorted_indices, topk_values


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
