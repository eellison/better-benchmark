"""
Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full
ignore-index cross-entropy mean by folding the `arg2_1` bias add into a
single-pass online logsumexp over each `[2048, 8008]` row, eagerly loading the
target logit, emitting only per-row loss and valid-count scalars, and reducing
those to the final scalar division. Inductor currently lowers the decomposed
add/amax/sub/exp/sum/log/log-softmax/gather/mask/sum/count/div graph as generic
row reductions plus pointwise/gather work, so it materializes and rereads the
large log-softmax-sized intermediate instead of recognizing that the gathered
negative log-probability can be computed as `max + log(sum_exp) - x_target`.
The fix is a NEW_PATTERN lowering for biased log_softmax+gather+masked-mean
cross entropy that emits an online accumulator row template with the scalar
loss/count epilogue.
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
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "amax_sum_sum_b9ac8700504c"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 16
SEQ_LEN = 128
N_ROWS = BATCH * SEQ_LEN
N_COLS = 8008

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
def _biased_online_xent_rows_kernel(
    labels_ptr,
    logits_ptr,
    bias_ptr,
    loss_ptr,
    valid_ptr,
    n_cols: tl.constexpr,
    block_n: tl.constexpr,
):
    row = tl.program_id(0)
    label = tl.load(labels_ptr + row)
    is_valid = label != -100
    safe_label = tl.where(is_valid, label, 0)

    row_start = row * n_cols
    x_label = tl.load(
        logits_ptr + row_start + safe_label,
        mask=is_valid,
        other=0.0,
    ).to(tl.float32)
    x_label += tl.load(bias_ptr + safe_label, mask=is_valid, other=0.0).to(tl.float32)

    row_max = tl.full([], -float("inf"), tl.float32)
    denom = tl.full([], 0.0, tl.float32)

    for block_start in tl.range(0, n_cols, block_n):
        cols = block_start + tl.arange(0, block_n)
        mask = cols < n_cols
        x = tl.load(
            logits_ptr + row_start + cols,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + b

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


def _launch_oracle(
    arg0_1: torch.Tensor,
    mm: torch.Tensor,
    arg2_1: torch.Tensor,
    loss_per_row: torch.Tensor,
    valid_per_row: torch.Tensor,
    out: torch.Tensor,
    block_n: int,
    num_warps: int,
) -> torch.Tensor:
    assert arg0_1.is_cuda and mm.is_cuda and arg2_1.is_cuda
    assert arg0_1.dtype == torch.int64 and arg0_1.numel() == mm.shape[0]
    assert mm.dtype == torch.float32 and mm.ndim == 2
    assert arg2_1.dtype == torch.float32 and arg2_1.numel() == mm.shape[1]
    assert loss_per_row.shape == (mm.shape[0],) and loss_per_row.dtype == torch.float32
    assert valid_per_row.shape == (mm.shape[0],) and valid_per_row.dtype == torch.float32
    assert out.shape == () and out.dtype == torch.float32

    n_rows, n_cols = mm.shape
    _biased_online_xent_rows_kernel[(n_rows,)](
        arg0_1,
        mm,
        arg2_1,
        loss_per_row,
        valid_per_row,
        n_cols=n_cols,
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
    mm: torch.Tensor,
    arg2_1: torch.Tensor,
    arg3_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    *,
    block_n: int = 2048,
    num_warps: int = 8,
) -> torch.Tensor:
    n_rows = mm.shape[0]
    loss_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    valid_per_row = torch.empty((n_rows,), device=mm.device, dtype=torch.float32)
    out = torch.empty((), device=mm.device, dtype=torch.float32)
    return _launch_oracle(
        arg3_1,
        mm,
        arg2_1,
        loss_per_row,
        valid_per_row,
        out,
        block_n=block_n,
        num_warps=num_warps,
    )


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
