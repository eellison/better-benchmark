"""
Canonical-local oracle scaffold for repro var_mean_765fb8f2c85e.

Repro pattern core:
    x[N, C, H, W] -> BN training stats -> running stats update -> affine -> ReLU

The captured repro continues with max-pool and avg-pool.  This file focuses on
the P1 norm_template_canonicalization opportunity: fusing training batch-norm
statistics, normalization, affine, and ReLU.  The pure PyTorch implementation is
the correctness reference; the Triton path is a straightforward three-kernel
floor scaffold that avoids ATen decomposition overhead but still leaves room for
a production tiled/reduction implementation.

TODOs for an exact production oracle:
  * Replace the simple one-program-per-channel reductions with a two-level tiled
    reduction for larger N*H*W and more stable occupancy.
  * Fuse stats finalization and affine/ReLU in a persistent channel tile when C
    is small enough, or use warp-specialized reductions for common BN shapes.
  * Extend this scaffold to include the following max-pool/avg-pool if measuring
    the entire captured repro instead of the BN+ReLU norm-template floor.
"""
from __future__ import annotations

import argparse
import csv
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
    from triton.testing import do_bench
except Exception:  # pragma: no cover - keeps --impl torch usable without Triton.
    triton = None
    tl = None
    do_bench = None


REPRO_ID = "var_mean_765fb8f2c85e"
REPRO_PATH = "repros/canonical/var_mean_765fb8f2c85e/repro.py"
ORACLE_PATH = "repros/canonical/var_mean_765fb8f2c85e/oracle_bn_training_forward.py"
DEFAULT_N = 128
DEFAULT_C = 192
DEFAULT_H = 71
DEFAULT_W = 71
DEFAULT_EPS = 0.001
DEFAULT_MOMENTUM = 0.1


if triton is not None:

    @triton.jit
    def _stats_kernel(x_ptr, mean_ptr, var_ptr, elems_per_channel: tl.constexpr, block: tl.constexpr):
        channel = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < elems_per_channel

        # x is contiguous NCHW.  For a fixed channel c, element k maps to:
        # n = k // (H*W), hw = k % (H*W), flat = n*C*H*W + c*H*W + hw.
        # The caller passes elems_per_channel=N*H*W and hw_size=H*W via C math.
        # H*W is recovered as elems_per_channel / N only in Python by supplying
        # flat stride constants through constexpr arguments in the final kernel.
        # This simple scaffold assumes block >= elems_per_channel for default
        # repro shape.  TODO: split this into partial reductions for larger M.
        vals = tl.load(x_ptr + channel * elems_per_channel + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        tl.store(mean_ptr + channel, mean)
        tl.store(var_ptr + channel, tl.maximum(var, 0.0))


    @triton.jit
    def _stats_nchw_kernel(
        x_ptr,
        mean_ptr,
        var_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        block: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < elems_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        flat_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
        vals = tl.load(x_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        tl.store(mean_ptr + channel, mean)
        tl.store(var_ptr + channel, tl.maximum(var, 0.0))


    @triton.jit
    def _running_stats_kernel(
        mean_ptr,
        var_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_running_mean_ptr,
        out_running_var_ptr,
        channels: tl.constexpr,
        elems_per_channel: tl.constexpr,
        momentum: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.arange(0, block)
        mask = offsets < channels
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        correction = elems_per_channel / (elems_per_channel - 1.0)
        tl.store(out_running_mean_ptr + offsets, old_mean * (1.0 - momentum) + mean * momentum, mask=mask)
        tl.store(out_running_var_ptr + offsets, old_var * (1.0 - momentum) + var * correction * momentum, mask=mask)


    @triton.jit
    def _affine_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        var_ptr,
        y_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        program = tl.program_id(0)
        offsets = program * block + tl.arange(0, block)
        mask = offsets < total
        channel = (offsets // hw_size) % channels
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * tl.rsqrt(var + eps) * weight + bias
        tl.store(y_ptr + offsets, tl.maximum(y, 0.0), mask=mask)


def torch_bn_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation of the BN training forward + affine + ReLU floor."""
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    y = torch.relu((x - mean) * inv_std * weight[None, :, None, None] + bias[None, :, None, None])

    elems_per_channel = x.shape[0] * x.shape[2] * x.shape[3]
    correction = elems_per_channel / (elems_per_channel - 1)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    new_running_mean = running_mean * (1.0 - momentum) + mean_1d * momentum
    new_running_var = running_var * (1.0 - momentum) + var_1d * correction * momentum
    return y, new_running_mean, new_running_var


def triton_bn_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
    stats_block: int | None = None,
    affine_block: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton scaffold for BN training forward + running stats + affine + ReLU."""
    if triton is None:
        raise RuntimeError("Triton is not available; use --impl torch")
    if not x.is_cuda or x.dtype != torch.float32 or x.ndim != 4 or not x.is_contiguous():
        raise ValueError("triton_bn_relu expects contiguous CUDA float32 NCHW input")

    n_batches, channels, height, width = x.shape
    hw_size = height * width
    elems_per_channel = n_batches * hw_size
    block = stats_block or triton.next_power_of_2(elems_per_channel)
    if block < elems_per_channel:
        raise ValueError("stats_block must cover N*H*W in this simple scaffold")

    mean = torch.empty((channels,), device=x.device, dtype=torch.float32)
    var = torch.empty((channels,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    new_running_mean = torch.empty_like(running_mean)
    new_running_var = torch.empty_like(running_var)

    _stats_nchw_kernel[(channels,)](
        x,
        mean,
        var,
        channels=channels,
        hw_size=hw_size,
        elems_per_channel=elems_per_channel,
        block=block,
        num_warps=8,
    )
    _running_stats_kernel[(1,)](
        mean,
        var,
        running_mean,
        running_var,
        new_running_mean,
        new_running_var,
        channels=channels,
        elems_per_channel=elems_per_channel,
        momentum=momentum,
        block=triton.next_power_of_2(channels),
        num_warps=8,
    )
    grid = (triton.cdiv(x.numel(), affine_block),)
    _affine_relu_kernel[grid](
        x,
        weight,
        bias,
        mean,
        var,
        y,
        total=x.numel(),
        channels=channels,
        hw_size=hw_size,
        eps=eps,
        block=affine_block,
        num_warps=4,
    )
    return y, new_running_mean, new_running_var


def make_inputs(
    n_batches: int,
    channels: int,
    height: int,
    width: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn((n_batches, channels, height, width), device=device, dtype=torch.float32)
    running_mean = torch.randn((channels,), device=device, dtype=torch.float32)
    running_var = torch.rand((channels,), device=device, dtype=torch.float32) + 0.5
    weight = torch.randn((channels,), device=device, dtype=torch.float32)
    bias = torch.randn((channels,), device=device, dtype=torch.float32)
    return x, running_mean, running_var, weight, bias


def assert_close(
    got: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rtol: float,
    atol: float,
) -> tuple[bool, float]:
    max_abs_diff = 0.0
    ok = True
    for got_tensor, ref_tensor in zip(got, ref):
        diff = (got_tensor.float() - ref_tensor.float()).abs().max().item()
        max_abs_diff = max(max_abs_diff, diff)
        ok = ok and bool(torch.allclose(got_tensor.float(), ref_tensor.float(), rtol=rtol, atol=atol))
    return ok, max_abs_diff


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
            if row.get("repro_id") == REPRO_ID:
                return row
    return {}


def benchmark(fn: Callable[[], object], warmup: int, rep: int) -> float:
    if do_bench is not None and torch.cuda.is_available():
        return float(do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0)

    import time

    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - start) * 1_000_000.0 / rep


def append_result(args: argparse.Namespace, oracle_us: float, correct: str, max_abs_diff: float) -> None:
    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan"))
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan"))
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))

    row = {
        "repro_id": REPRO_ID,
        "repro_path": REPRO_PATH,
        "shape_label": f"{args.n}x{args.c}x{args.h}x{args.w}",
        "family": "batch_norm_training_forward_affine_relu",
        "oracle_impl": args.impl,
        "oracle_path": ORACLE_PATH,
        "hardware": args.hardware,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
        "tolerance": f"rtol={args.rtol},atol={args.atol}" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "BN training stats+running-stat update+affine+ReLU floor; captured repro pool tail intentionally excluded.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out.exists()
    with args.out.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BN training forward oracle scaffold for var_mean_765fb8f2c85e")
    parser.add_argument("--impl", choices=("torch", "triton"), default="torch")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--c", type=int, default=DEFAULT_C)
    parser.add_argument("--h", type=int, default=DEFAULT_H)
    parser.add_argument("--w", type=int, default=DEFAULT_W)
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument("--momentum", type=float, default=DEFAULT_MOMENTUM)
    parser.add_argument("--stats-block", type=int, default=None)
    parser.add_argument("--affine-block", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--no-append", action="store_true", help="Run without appending measured_oracle_floors.csv")
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--out", type=Path, default=Path("investigation_results/measured_oracle_floors.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu --impl torch for a CPU smoke test")
    if args.impl == "triton" and args.device != "cuda":
        raise RuntimeError("--impl triton requires --device cuda")

    inputs = make_inputs(args.n, args.c, args.h, args.w, args.device, args.seed)

    if args.impl == "torch":
        fn = lambda: torch_bn_relu(*inputs, eps=args.eps, momentum=args.momentum)
    else:
        fn = lambda: triton_bn_relu(
            *inputs,
            eps=args.eps,
            momentum=args.momentum,
            stats_block=args.stats_block,
            affine_block=args.affine_block,
        )

    out = fn()
    if args.device == "cuda":
        torch.cuda.synchronize()

    correct = "not_checked"
    max_abs_diff = math.nan
    if args.check:
        ref = torch_bn_relu(*inputs, eps=args.eps, momentum=args.momentum)
        if args.device == "cuda":
            torch.cuda.synchronize()
        ok, max_abs_diff = assert_close(out, ref, args.rtol, args.atol)
        correct = str(ok)
        print(f"correct={correct} max_abs_diff={max_abs_diff:.6g}")

    oracle_us = benchmark(fn, args.warmup, args.rep)
    if args.device == "cuda":
        torch.cuda.synchronize()
    print(
        "oracle_us="
        f"{oracle_us:.3f} impl={args.impl} shape={args.n}x{args.c}x{args.h}x{args.w} "
        f"eps={args.eps} momentum={args.momentum}"
    )

    if not args.no_append:
        append_result(args, oracle_us, correct, max_abs_diff)


if __name__ == "__main__":
    main()
