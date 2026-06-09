"""
Oracle for amax_sum_sum_4c98004c3aa3: online softmax cross-entropy mean.

Repro pattern:
  labels: i64[16, 512] -> shifted next-token labels with -100 in the last slot
  logits: f32[8192, 29056]
  logits -> amax/sub/exp/sum/log -> log_softmax -> gather(label)
         -> neg -> mask(label != -100) -> sum(losses) / sum(valid labels)

Gap diagnosis (classification: NEW_PATTERN): this oracle reads each logits row
once, using scalar online max and denominator accumulators plus one target-logit
load to compute the per-row cross-entropy directly, then reduces per-row loss
and valid-count vectors to the final scalar mean; Inductor currently lowers the
decomposed graph as generic amax/exp/sum/log/log_softmax/gather/mask/reduction
work, which materializes and rereads large intermediates instead of recognizing
that the gathered log-softmax value is just `x_label - max - log(sum_exp)`.
Inductor cannot do this today because the scheduler/template matcher does not
canonicalize the shifted-label, ignore-index cross-entropy idiom into an online
softmax row template with a scalar epilogue and sibling valid-count reduction.
The fix is a NEW_PATTERN Inductor lowering for cross-entropy-style
log_softmax+gather+masked-mean that emits this online accumulator kernel and
the small final scalar reduction.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_sum_4c98004c3aa3"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ_LEN = 512
N_COLS = 29056
M_ROWS = BATCH * SEQ_LEN

COMPILE_CONFIGS = [
    ("coordinate_descent_tuning", {"coordinate_descent_tuning": True}),
    (
        "combo_looped_cd",
        {
            "combo_kernels": True,
            "combo_kernel_per_subkernel_blocks": True,
            "coordinate_descent_tuning": True,
            "benchmark_combo_kernel": True,
            "triton.multi_kernel": 3,
        },
    ),
]


@triton.jit
def _online_xent_rows_kernel(
    tokens_ptr,
    logits_ptr,
    loss_ptr,
    valid_ptr,
    n_cols: tl.constexpr,
    seq_len: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    pos = row % seq_len
    is_last_token = pos == (seq_len - 1)

    label = tl.load(tokens_ptr + row + 1, mask=~is_last_token, other=-100)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    row_start = row * n_cols
    x_label = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)

    row_max = tl.full([], -float("inf"), tl.float32)
    denom = tl.full([], 0.0, tl.float32)

    for block_start in tl.range(0, n_cols, block_n):
        cols = block_start + tl.arange(0, block_n)
        col_mask = cols < n_cols
        x = tl.load(
            logits_ptr + row_start + cols,
            mask=col_mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        denom = denom * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max

    loss = row_max + tl.log(denom) - x_label
    loss = tl.where(is_valid, loss, 0.0)

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


def _shifted_labels(tokens: torch.Tensor) -> torch.Tensor:
    labels = torch.empty_like(tokens)
    labels[:, :-1] = tokens[:, 1:]
    labels[:, -1] = -100
    return labels.reshape(-1)


def _reference(tokens: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    labels = _shifted_labels(tokens)
    valid = labels != -100
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    row_max = logits.amax(dim=1, keepdim=True)
    shifted = logits - row_max
    log_denom = torch.log(torch.exp(shifted).sum(dim=1, keepdim=True))
    log_probs = shifted - log_denom

    gathered = log_probs.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    losses = torch.where(valid, -gathered, torch.zeros_like(gathered))
    return losses.sum() / valid.sum().to(torch.float32)


def _launch_oracle(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert tokens.is_cuda and logits.is_cuda
    assert tokens.dtype == torch.int64
    assert logits.dtype == torch.float32 and logits.ndim == 2
    assert tokens.ndim == 2
    n_rows, n_cols = logits.shape
    batch, seq_len = tokens.shape
    assert n_rows == batch * seq_len
    assert loss_per_row.shape == (n_rows,) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (n_rows,) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    _online_xent_rows_kernel[(n_rows,)](
        tokens,
        logits,
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


def oracle_online_softmax_xent_mean(
    tokens: torch.Tensor,
    logits: torch.Tensor,
    block_n: int = 4096,
    num_warps: int = 8,
) -> torch.Tensor:
    n_rows = logits.shape[0]
    loss_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=logits.device, dtype=torch.float32)
    out = torch.empty((), device=logits.device, dtype=torch.float32)
    return _launch_oracle(
        tokens,
        logits,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


def _compile_inputs(
    module,
    batch: int,
    seq_len: int,
    n_cols: int,
    seed: int,
) -> tuple:
    if (batch, seq_len, n_cols) == (BATCH, SEQ_LEN, N_COLS):
        inputs = module.make_inputs()
        return tuple(x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs)

    tokens, logits = _make_inputs(batch, seq_len, n_cols, seed=seed)
    return tokens, logits, [batch, seq_len, n_cols], [-1, n_cols]


@oracle_impl(hardware="H100", shapes="(T([16, 512], i64), T([8192, 29056], f32), S([16, 512, 29056]), S([-1, 29056]))")
def oracle_forward(inputs):
    return oracle_online_softmax_xent_mean(*inputs)


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

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
