"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete shifted-label ignore-index cross-entropy mean returned by Repro.forward, including the constant_pad/slice/clone/view label shift, bf16/f16 logits view and fp32 conversion, stable online row logsumexp, safe masked target gather, valid-loss sum, valid-count sum, and final scalar division in Triton, whereas Inductor currently lowers the decomposed pad/slice/view/amax/sub/exp/sum/log/log-softmax/gather/mask/sum/count/div graph as generic row reductions plus pointwise and gather work that materializes full log-softmax-sized intermediates; Inductor cannot do this today because its pattern library does not canonicalize shifted-label ignore-index cross-entropy mean into a fused online row-reduction with scalar epilogue; the fix is NEW_PATTERN: add an online cross-entropy lowering that recognizes label-shift producers, masked target gather, and sibling valid-count reduction."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    get_shape_key,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _shifted_xent_rows_kernel(
        labels_ptr,
        logits_ptr,
        loss_ptr,
        valid_ptr,
        n_cols: tl.constexpr,
        seq_len: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        pos = row % seq_len
        row_start = row * n_cols

        shifted_label = tl.load(
            labels_ptr + row + 1,
            mask=pos < seq_len - 1,
            other=-100,
        )
        is_valid = shifted_label != -100
        safe_label = tl.where(is_valid, shifted_label, 0)
        target_logit = tl.load(
            logits_ptr + row_start + safe_label,
            mask=is_valid,
            other=0.0,
        ).to(tl.float32)

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


def _block_n_for(n_cols: int) -> tuple[int, int]:
    if n_cols <= 32768:
        return 2048, 8
    if n_cols <= 65536:
        return 4096, 8
    return 8192, 16


def _launch_oracle(
    labels: torch.Tensor,
    logits_2d: torch.Tensor,
    out: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    *,
    seq_len: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not labels.is_cuda or not logits_2d.is_cuda:
        raise RuntimeError("CUDA tensors are required for this Triton oracle")
    if labels.dtype != torch.int64 or labels.ndim != 2:
        raise TypeError("expected i64 labels with shape [batch, seq_len]")
    if logits_2d.dtype not in (torch.float16, torch.bfloat16) or logits_2d.ndim != 2:
        raise TypeError("expected f16/bf16 logits with shape [rows, vocab]")
    if not labels.is_contiguous() or not logits_2d.is_contiguous():
        raise ValueError("oracle expects canonical contiguous repro inputs")

    n_rows, n_cols = logits_2d.shape
    if labels.numel() != n_rows or labels.shape[1] != seq_len:
        raise ValueError("labels must flatten to the logits row count")
    if out.shape != () or out.dtype != torch.float32:
        raise ValueError("output must be a scalar f32 tensor")
    if loss_per_row.shape != (n_rows,) or valid_per_row.shape != (n_rows,):
        raise ValueError("temporary buffers must have one entry per row")

    block_n, num_warps = _block_n_for(n_cols)
    _shifted_xent_rows_kernel[(n_rows,)](
        labels,
        logits_2d,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
        seq_len=seq_len,
        block_n=block_n,
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


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the canonical inputs."""
    labels, logits, shape_3d, shape_2d = inputs
    logits_2d = logits.view(shape_3d).view(shape_2d)
    n_rows = logits_2d.shape[0]

    out = torch.empty((), device=logits.device, dtype=torch.float32)
    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    return _launch_oracle(
        labels,
        logits_2d,
        out,
        loss_per_row,
        valid_per_row,
        seq_len=labels.shape[1],
    )


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
