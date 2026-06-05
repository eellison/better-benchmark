"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete MobileBERT masked-LM ignore-index cross-entropy mean returned by Repro.forward, including the logits slice, vocabulary bias add, returned [256, 128, 30522] biased-logits layout, stable online row logsumexp, safe masked target gather, valid-loss sum, valid-count sum, and final scalar division in Triton, whereas Inductor currently lowers the decomposed slice/add/view/amax/sub/exp/sum/log/gather/mask/sum/count/div graph as generic reductions and pointwise/layout kernels that materialize and reread full log-softmax-sized intermediates; Inductor cannot do this today because its pattern library does not canonicalize biased ignore-index cross entropy with a required sibling logits output into a fused online row-reduction plus side-output store; the fix is NEW_PATTERN: add a guarded biased log_softmax plus masked-gather mean lowering that emits this online cross-entropy row kernel and scalar epilogue directly."""
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
# Do not add custom benchmark functions. bench_oracle() owns timing so graph
# capture, GPU locking, and interleaved oracle/compile measurement are preserved.
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


BLOCK_N = 2048
NUM_WARPS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _biased_xent_rows_kernel(
        labels_ptr,
        mm_ptr,
        bias_ptr,
        logits_out_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        mm_row_stride: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols_base = tl.arange(0, block_n)
        mm_row_start = row * mm_row_stride
        out_row_start = row * n_cols

        label = tl.load(labels_ptr + row)
        is_valid = label != -100
        safe_label = tl.where(is_valid, label, 0)
        target = tl.load(
            mm_ptr + mm_row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)
        target += tl.load(
            bias_ptr + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

        row_max = tl.full([], -float("inf"), tl.float32)
        denom = tl.full([], 0.0, tl.float32)

        for block_start in tl.range(0, n_cols, block_n):
            cols = block_start + cols_base
            mask = cols < n_cols
            x = tl.load(
                mm_ptr + mm_row_start + cols,
                mask=mask,
                other=-float("inf"),
            ).to(tl.float32)
            b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            x = x + b
            tl.store(logits_out_ptr + out_row_start + cols, x, mask=mask)

            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            denom = denom * tl.exp(row_max - new_max) + tl.sum(
                tl.exp(x - new_max),
                axis=0,
            )
            row_max = new_max

        loss = row_max + tl.log(denom) - target
        tl.store(loss_ptr + row, tl.where(is_valid, loss, 0.0))
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
    labels_1d: torch.Tensor,
    mm: torch.Tensor,
    bias: torch.Tensor,
    logits_out: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    mean_out: torch.Tensor,
    *,
    block_n: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_cuda()

    if labels_1d.dtype != torch.int64 or labels_1d.ndim != 1:
        raise TypeError("expected i64 labels with shape [rows]")
    if mm.dtype != torch.float32 or mm.ndim != 2:
        raise TypeError("expected f32 mm with shape [rows, vocab + 2]")
    if bias.dtype != torch.float32 or bias.ndim != 1:
        raise TypeError("expected f32 bias with shape [vocab]")
    if not labels_1d.is_cuda or not mm.is_cuda or not bias.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")

    n_rows = labels_1d.numel()
    n_cols = bias.numel()
    if mm.shape[0] != n_rows or mm.shape[1] < n_cols:
        raise ValueError(
            f"mm shape {tuple(mm.shape)} does not cover labels={n_rows}, bias={n_cols}"
        )
    if logits_out.numel() != n_rows * n_cols or logits_out.dtype != torch.float32:
        raise ValueError("logits_out must hold the full returned biased logits")
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("temporary buffers must have one entry per row")
    if mean_out.shape != () or mean_out.dtype != torch.float32:
        raise ValueError("mean_out must be a scalar f32 tensor")

    _biased_xent_rows_kernel[(n_rows,)](
        labels_1d,
        mm,
        bias,
        logits_out,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        mm_row_stride=mm.stride(0),
        block_n=_effective_block_n(block_n, n_cols),
        num_warps=num_warps,
    )
    _mean_reduce_kernel[(1,)](
        loss_per_row,
        valid_per_row,
        mean_out,
        n_rows=n_rows,
        block_m=triton.next_power_of_2(n_rows),
        num_warps=8,
    )
    return mean_out, logits_out


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: accepts the exact inputs from Repro.forward() and returns
    the same output tuple: scalar mean loss and returned biased logits view.
    """
    labels, mm, bias, shape_3d, shape_2d, output_shape = inputs
    n_rows = labels.numel()
    n_cols = bias.numel()

    if tuple(shape_2d) != (n_rows, n_cols):
        raise ValueError(f"unexpected 2D shape parameter {shape_2d}")
    if tuple(output_shape) != tuple(shape_3d):
        raise ValueError(f"unexpected output shape parameter {output_shape}")
    output_numel = 1
    for dim in output_shape:
        output_numel *= dim
    if n_rows * n_cols != output_numel:
        raise ValueError("shape parameters do not match labels and bias")

    logits_out = torch.empty(tuple(output_shape), device=mm.device, dtype=torch.float32)
    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    mean_out = torch.empty((), device=mm.device, dtype=torch.float32)

    return _launch_oracle(
        labels.view(-1),
        mm,
        bias,
        logits_out,
        loss_per_row,
        valid_per_row,
        mean_out,
        block_n=BLOCK_N,
        num_warps=NUM_WARPS,
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
