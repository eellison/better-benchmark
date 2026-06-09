"""
Canonical-local measured oracle floor for repro amax_sum_sum_dc96a87ba8db.

Repro pattern:
    x_bf16[M, N] -> softmax(x.float(), dim=-1).to(bf16).sum()

This oracle avoids materializing the full bf16 softmax output. It computes each
row's rounded-bf16 softmax sum with an online softmax pass, stores one fp32 row
sum per row, then reduces row sums to a scalar.
"""
from __future__ import annotations

import argparse
import sys
import csv
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.testing import do_bench

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

REPRO_ID = "amax_sum_sum_dc96a87ba8db"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"



@triton.jit
def _softmax_row_sums_kernel(x_ptr, row_sums_ptr, n_cols: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0)
    row_start = row * n_cols

    m_i = tl.full([], -float("inf"), tl.float32)
    l_i = tl.full([], 0.0, tl.float32)

    for off in tl.range(0, n_cols, block):
        cols = off + tl.arange(0, block)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
        m_ij = tl.max(x, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        l_i = l_i * tl.exp(m_i - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m_i = m_new

    row_sum = tl.full([], 0.0, tl.float32)
    for off in tl.range(0, n_cols, block):
        cols = off + tl.arange(0, block)
        mask = cols < n_cols
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=-float("inf")).to(tl.float32)
        p = tl.exp(x - m_i) / l_i
        # Match repro's bf16 materialization before sum as closely as practical.
        p_bf16_as_f32 = p.to(tl.bfloat16).to(tl.float32)
        row_sum += tl.sum(tl.where(mask, p_bf16_as_f32, 0.0), axis=0)

    tl.store(row_sums_ptr + row, row_sum)


@triton.jit
def _sum_rows_kernel(row_sums_ptr, out_ptr, n_rows: tl.constexpr, block: tl.constexpr):
    offsets = tl.arange(0, block)
    mask = offsets < n_rows
    vals = tl.load(row_sums_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(vals, axis=0)
    tl.store(out_ptr, total.to(tl.bfloat16))


def oracle_softmax_sum(x: torch.Tensor, block: int = 8192, warps: int = 8) -> torch.Tensor:
    assert x.is_cuda and x.dtype is torch.bfloat16 and x.ndim == 2
    n_rows, n_cols = x.shape
    row_sums = torch.empty((n_rows,), device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.bfloat16)
    _softmax_row_sums_kernel[(n_rows,)](x, row_sums, n_cols, block=block, num_warps=warps)
    pow2_rows = triton.next_power_of_2(n_rows)
    _sum_rows_kernel[(1,)](row_sums, out, n_rows, block=pow2_rows, num_warps=8)
    return out


def eager_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(torch.bfloat16).sum()


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_baseline_row() -> dict[str, str]:
    path = Path("investigation_results/sol_gap_candidates.csv")
    if not path.exists():
        return {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["repro_id"] == "amax_sum_sum_dc96a87ba8db":
                return row
    return {}


@oracle_impl(hardware="H100", shapes="(T([8192, 262144], bf16))")
def oracle_forward(inputs):
    return oracle_softmax_sum(*inputs)


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
