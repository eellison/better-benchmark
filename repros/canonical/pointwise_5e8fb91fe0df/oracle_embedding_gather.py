"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete embedding gather plus multiply-by-one scope as one Triton gather-copy kernel that directly materializes the required contiguous `[batch, sequence, hidden]` output for all local embedding shapes, whereas Inductor already lowers the captured `aten.embedding.default` and no-op `aten.mul.Tensor(..., 1.0)` region to the same mandatory indexed table reads and output stores; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the only remaining work is the required random row gather and dense output write; the fix is BANDWIDTH_BOUND: record this as an at-floor embedding-gather case unless broader gather memory-throughput or launch-overhead work moves both implementations."""
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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 1, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 2, "BLOCK_D": 1024}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 4, "BLOCK_D": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 8, "BLOCK_D": 512}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 16, "BLOCK_D": 256}, num_warps=8, num_stages=3),
        ],
        key=["N_ROWS", "HIDDEN"],
    )
    @triton.jit
    def _embedding_gather_kernel(
        weight_ptr,
        index_ptr,
        out_ptr,
        N_ROWS: tl.constexpr,
        HIDDEN: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        row_mask = row_offsets < N_ROWS
        col_mask = col_offsets < HIDDEN
        mask = row_mask[:, None] & col_mask[None, :]

        token_ids = tl.load(index_ptr + row_offsets, mask=row_mask, other=0)
        values = tl.load(
            weight_ptr + token_ids[:, None] * HIDDEN + col_offsets[None, :],
            mask=mask,
            other=0.0,
        )
        tl.store(out_ptr + row_offsets[:, None] * HIDDEN + col_offsets[None, :], values, mask=mask)


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    weight, indices = inputs
    if not isinstance(weight, torch.Tensor) or not isinstance(indices, torch.Tensor):
        raise TypeError(f"{REPRO_ID} expects tensor weight and index inputs")
    if not weight.is_cuda or not indices.is_cuda:
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if weight.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected fp16/fp32 embedding table, got {weight.dtype}")
    if indices.dtype != torch.int64:
        raise TypeError(f"expected int64 embedding indices, got {indices.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 embedding table, got shape {tuple(weight.shape)}")
    if indices.ndim != 2:
        raise ValueError(f"expected rank-2 embedding indices, got shape {tuple(indices.shape)}")
    if not weight.is_contiguous() or not indices.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects captured contiguous input layouts")
    return weight, indices


@oracle_impl(hardware="H100", shapes="(T([50265, 1024], f32), T([8, 1024], i64, gen=Index(50265)))")
def oracle_forward(inputs):
    """Run the full embedding gather scope for the same inputs as Repro.forward()."""
    weight, indices = _validate_inputs(inputs)
    batch, seq = indices.shape
    hidden = weight.shape[1]
    n_rows = batch * seq

    output = torch.empty_strided(
        (batch, seq, hidden),
        (seq * hidden, hidden, 1),
        device=weight.device,
        dtype=weight.dtype,
    )
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_M"]), triton.cdiv(hidden, meta["BLOCK_D"]))
    _embedding_gather_kernel[grid](
        weight,
        indices,
        output,
        N_ROWS=n_rows,
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
