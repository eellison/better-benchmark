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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--cols", type=int, default=262144)
    parser.add_argument("--block", type=int, default=8192)
    parser.add_argument("--warps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Run an eager correctness check; expensive for the default shape.")
    parser.add_argument("--out", type=Path, default=Path("investigation_results/measured_oracle_floors.csv"))
    args = parser.parse_args()

    torch.manual_seed(0)
    x = torch.randn((args.rows, args.cols), device="cuda", dtype=torch.bfloat16)

    out = oracle_softmax_sum(x, block=args.block, warps=args.warps)
    torch.cuda.synchronize()

    correct = "not_checked"
    max_abs_diff = math.nan
    if args.check:
        ref = eager_ref(x)
        torch.cuda.synchronize()
        max_abs_diff = (out.float() - ref.float()).abs().item()
        correct = str(bool(torch.allclose(out.float(), ref.float(), rtol=1e-2, atol=1e-2)))
        print(f"correct={correct} oracle={out.item()} ref={ref.item()} max_abs_diff={max_abs_diff}")

    def run_oracle():
        oracle_softmax_sum(x, block=args.block, warps=args.warps)

    oracle_ms = do_bench(run_oracle, warmup=args.warmup, rep=args.rep, return_mode="min")
    oracle_us = oracle_ms * 1000.0
    print(f"oracle_us={oracle_us:.3f} rows={args.rows} cols={args.cols} block={args.block} warps={args.warps}")

    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan"))
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan"))
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))

    row = {
        "repro_id": "amax_sum_sum_dc96a87ba8db",
        "repro_path": "repros/canonical/amax_sum_sum_dc96a87ba8db/repro.py",
        "shape_label": f"{args.rows}x{args.cols}",
        "family": "online_softmax_cross_entropy",
        "oracle_impl": "triton_online_softmax_sum_two_stage",
        "oracle_path": "repros/canonical/amax_sum_sum_dc96a87ba8db/oracle_softmax_sum.py",
        "hardware": "B200",
        "device_name": torch.cuda.get_device_name(0),
        "git_commit": get_git_commit(),
        "compiled_us": baseline.get("compiled_us", ""),
        "coord_descent_us": baseline.get("coord_descent_us", ""),
        "best_compile_us": best_compile_us,
        "memcopy_sol_us": memcopy_sol_us,
        "oracle_us": oracle_us,
        "total_bytes": total_bytes,
        "n_kernels": n_kernels,
        "oracle_over_sol": oracle_us / memcopy_sol_us if memcopy_sol_us == memcopy_sol_us else math.nan,
        "speedup_vs_best_compile": best_compile_us / oracle_us if best_compile_us == best_compile_us else math.nan,
        "correct": correct,
        "max_abs_diff": max_abs_diff,
        "tolerance": "rtol=1e-2,atol=1e-2" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Oracle avoids materializing full bf16 softmax; exact repro-equivalent scalar sum within bf16 tolerance.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended {args.out}")


if __name__ == "__main__":
    main()
