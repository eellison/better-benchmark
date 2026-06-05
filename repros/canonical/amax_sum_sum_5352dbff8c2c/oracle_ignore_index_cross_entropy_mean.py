"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete T5 ignore-index cross-entropy mean returned by Repro.forward, including the [8192, 32128] to [8, 1024, 32128] to [-1, 32128] views, stable online row logsumexp, safe target gather through the input full_5 scalar for ignored labels, ignored-row fill contribution from the input full scalar, valid-count reduction, loss reduction, and final average scalar in Triton, whereas Inductor currently lowers the decomposed view/amax/sub/exp/sum/log/sub/ne/where/gather/squeeze/neg/where/sum/count/div graph as generic row reductions, pointwise kernels, gather work, and scalar reductions that materialize the full log-softmax-sized intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize ignore-index cross-entropy mean with explicit safe-index and ignored-loss fill inputs into a fused online row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor lowering for log_softmax plus masked gather plus loss/count mean that emits an online cross-entropy row kernel and a small final reduction directly."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _xent_rows_kernel(
        logits_ptr,
        labels_ptr,
        safe_index_ptr,
        fill_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        row_start = row * n_cols

        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_index = tl.load(safe_index_ptr)
        gather_index = tl.where(is_valid, label, safe_index)
        target_logit = tl.load(logits_ptr + row_start + gather_index).to(tl.float32)
        ignored_fill = tl.load(fill_ptr).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + tl.arange(0, block_n)
            mask = cols < n_cols
            logits = tl.load(
                logits_ptr + row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)

            block_max = tl.max(logits, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(logits - new_max),
                axis=0,
            )
            row_max = new_max

        loss = row_max + tl.log(denom) - target_logit
        tl.store(loss_ptr + row, tl.where(is_valid, loss, ignored_fill))
        tl.store(valid_ptr + row, tl.where(is_valid, 1.0, 0.0))

    @triton.jit
    def _mean_reduce_kernel(
        loss_ptr,
        valid_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        block_m: tl.constexpr,
    ):
        offsets = tl.arange(0, block_m)
        mask = offsets < n_rows
        losses = tl.load(loss_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        valid = tl.load(valid_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        total_loss = tl.sum(losses, axis=0)
        total_valid = tl.sum(valid, axis=0)
        tl.store(out_ptr, total_loss / total_valid)


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _effective_block_n(block_n: int, n_cols: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    _check_power_of_two(block_n, "block_n")
    return min(block_n, triton.next_power_of_2(n_cols))


def _launch_oracle(
    logits_2d: torch.Tensor,
    labels_1d: torch.Tensor,
    safe_index: torch.Tensor,
    ignored_fill: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not logits_2d.is_cuda or not labels_1d.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if not safe_index.is_cuda or not ignored_fill.is_cuda:
        raise RuntimeError("CUDA scalar inputs are required for this Triton oracle")
    if logits_2d.dtype != torch.float32 or logits_2d.ndim != 2:
        raise TypeError("expected f32 logits with shape [rows, vocab]")
    if labels_1d.dtype != torch.int64 or labels_1d.ndim != 1:
        raise TypeError("expected i64 labels with shape [rows]")
    if safe_index.shape != () or safe_index.dtype != torch.int64:
        raise TypeError("expected scalar i64 safe-index input")
    if ignored_fill.shape != () or ignored_fill.dtype != torch.float32:
        raise TypeError("expected scalar f32 ignored-loss fill input")

    n_rows, n_cols = logits_2d.shape
    if labels_1d.shape != (n_rows,):
        raise ValueError("labels must flatten to the logits row count")
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("temporary buffers must have one entry per row")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError("output must be a scalar f32 tensor")

    _xent_rows_kernel[(n_rows,)](
        logits_2d,
        labels_1d,
        safe_index,
        ignored_fill,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        block_n=_effective_block_n(block_n, n_cols),
        num_warps=num_warps,
    )

    block_m = triton.next_power_of_2(n_rows)
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=block_m,
        num_warps=8,
    )
    return out


def oracle_ignore_index_cross_entropy_mean(
    mm_96: torch.Tensor,
    arg52_1: torch.Tensor,
    full_5: torch.Tensor,
    full: torch.Tensor,
    shape_3d,
    shape_2d,
    *,
    block_n: int = 8192,
    num_warps: int = 16,
) -> torch.Tensor:
    logits_2d = mm_96.view(shape_3d).view(shape_2d)
    labels_1d = arg52_1.view(-1)
    n_rows = logits_2d.shape[0]

    loss_per_row = torch.empty((n_rows,), device=mm_96.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm_96.device, dtype=torch.float32)
    out = torch.empty((), device=mm_96.device, dtype=torch.float32)
    return _launch_oracle(
        logits_2d,
        labels_1d,
        full_5,
        full,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
    """
    return oracle_ignore_index_cross_entropy_mean(*inputs)


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
