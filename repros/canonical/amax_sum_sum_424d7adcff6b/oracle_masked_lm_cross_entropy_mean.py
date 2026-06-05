"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete masked-LM ignore-index cross-entropy mean returned by Repro.forward, including the sliced logits layout, stable row logsumexp, safe masked target gather, ignored-row fill value, valid-count reduction, loss reduction, and final scalar division in Triton, whereas Inductor currently lowers the decomposed view/amax/sub/exp/sum/log/sub/ne/where/gather/squeeze/neg/where/sum/count/div graph as separate generic row reductions, pointwise kernels, gather work, and scalar reductions that materialize the full log-softmax intermediate; Inductor cannot do this today because its scheduler/codegen pattern library does not canonicalize full ignore-index cross-entropy mean with an explicit sliced-vocab layout and sibling valid-count reduction into a fused online row-reduction plus scalar epilogue; the fix is NEW_PATTERN: add an Inductor lowering for log_softmax plus masked gather plus loss/count mean that emits an online cross-entropy row kernel and a small final reduction directly."""
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
    def _masked_lm_xent_rows_kernel(
        logits_ptr,
        labels_ptr,
        fill_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        logits_row_stride: tl.constexpr,
        logits_col_stride: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)

        row_start = row * logits_row_stride
        target = tl.load(
            logits_ptr + row_start + safe_label * logits_col_stride,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)
        cols_base = tl.arange(0, block_n)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            logits = tl.load(
                logits_ptr + row_start + cols * logits_col_stride,
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

        fill = tl.load(fill_ptr).to(tl.float32)
        loss = row_max + tl.log(denom) - target
        loss = tl.where(is_valid, loss, fill)

        tl.store(loss_ptr + row, loss)
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


def _require_triton_cuda() -> None:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def _effective_block_n(block_n: int, n_cols: int) -> int:
    _check_power_of_two(block_n, "block_n")
    return min(block_n, triton.next_power_of_2(n_cols))


def _launch_oracle(
    logits: torch.Tensor,
    labels_1d: torch.Tensor,
    fill: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    *,
    n_cols: int,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    _require_triton_cuda()
    if logits.ndim != 2 or logits.dtype != torch.float32:
        raise ValueError(f"expected f32 2D logits, got shape={tuple(logits.shape)} dtype={logits.dtype}")
    if labels_1d.ndim != 1 or labels_1d.dtype != torch.int64:
        raise ValueError(f"expected i64 flattened labels, got shape={tuple(labels_1d.shape)} dtype={labels_1d.dtype}")
    if fill.shape != () or fill.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 fill, got shape={tuple(fill.shape)} dtype={fill.dtype}")

    n_rows = labels_1d.numel()
    if logits.shape[0] != n_rows or logits.shape[1] < n_cols:
        raise ValueError(f"logits shape {tuple(logits.shape)} does not cover rows={n_rows}, cols={n_cols}")
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("temporary buffer shapes do not match row count")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError(f"expected scalar f32 output, got shape={tuple(out.shape)} dtype={out.dtype}")

    _masked_lm_xent_rows_kernel[(n_rows,)](
        logits,
        labels_1d,
        fill,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        logits_row_stride=logits.stride(0),
        logits_col_stride=logits.stride(1),
        block_n=_effective_block_n(block_n, n_cols),
        num_warps=num_warps,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return out


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
    addmm_73, arg205_1, full_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    n_cols = int(_shape_param_1[1])
    logits = addmm_73[:, :n_cols]
    labels_1d = arg205_1.view(-1)
    n_rows = labels_1d.numel()

    loss_per_row = torch.empty((n_rows,), device=addmm_73.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=addmm_73.device, dtype=torch.float32)
    out = torch.empty((), device=addmm_73.device, dtype=torch.float32)
    return _launch_oracle(
        logits,
        labels_1d,
        full_1,
        loss_per_row,
        valid_per_row,
        out,
        n_cols=n_cols,
        block_n=8192,
        num_warps=16,
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
