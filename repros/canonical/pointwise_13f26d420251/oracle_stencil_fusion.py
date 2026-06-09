"""
Oracle for pointwise_13f26d420251: fused BatchNorm-affine + ReLU + MaxPool + AvgPool.

Pattern: BN-affine (sub, rsqrt, mul, mul, add) -> ReLU -> max_pool -> avg_pool
Shape: [128, 192, 71, 71] f32 (main tensor), BN params [192]

Optimization: the compiled code materializes:
  1. BN-affine output [128, 192, 71, 71] f32
  2. ReLU output [128, 192, 71, 71] f32
  3. MaxPool output [128, 192, 35, 35] f32 (consumed by avg_pool)

The oracle fuses BN-affine + ReLU into the MaxPool read (eliminating intermediates 1 & 2),
then the AvgPool reads the MaxPool output. The big win is eliminating the two large
[128,192,71,71] writes/reads (~2 * 128*192*71*71*4 = ~992 MB of traffic).

Outputs:
  - pool offsets: i8[128, 192, 35, 35]
  - avg_pool result: f32[128, 192, 35, 35]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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



REPRO_ID = "pointwise_13f26d420251"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"



@triton.jit
def fused_bn_relu_maxpool_kernel(
    input_ptr,         # [B, C, H, W] f32 contiguous
    mean_ptr,          # [C] f32
    inv_std_ptr,       # [C] f32 (precomputed 1/sqrt(var+eps))
    weight_ptr,        # [C] f32
    bias_ptr,          # [C] f32
    out_val_ptr,       # [B, C, H_out, W_out] f32
    out_idx_ptr,       # [B, C, H_out, W_out] i8
    B: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """Fused BN + ReLU + MaxPool kernel. Each program handles one (b, c, oh) row."""
    pid = tl.program_id(0)
    n_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
    bc_oh = pid // n_ow_blocks
    ow_block = pid % n_ow_blocks

    bc = bc_oh // H_out
    oh = bc_oh % H_out
    b = bc // C
    c = bc % C

    # Load BN parameters for this channel
    mean = tl.load(mean_ptr + c)
    inv_std = tl.load(inv_std_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    # Output column offsets
    ow_offs = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < W_out

    # Stride-2 pooling, kernel=3, padding=0, dilation=1, ceil_mode=False
    ih_start = oh * 2
    iw_starts = ow_offs * 2

    # Input base for this (b, c) plane
    input_base = b * (C * H_in * W_in) + c * (H_in * W_in)

    best_val = tl.full([BLOCK_OW], -float('inf'), dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_OW], dtype=tl.int8)

    for kh in tl.static_range(3):
        ih = ih_start + kh
        ih_valid = ih < H_in
        for kw in tl.static_range(3):
            iw = iw_starts + kw
            iw_valid = iw < W_in
            valid = ow_mask & ih_valid & iw_valid

            offset = input_base + ih * W_in + iw
            val = tl.load(input_ptr + offset, mask=valid, other=0.0)

            # Fused BN-affine + ReLU
            val = (val - mean) * inv_std * weight + bias
            val = tl.maximum(val, 0.0)

            is_better = val > best_val
            best_val = tl.where(is_better, val, best_val)
            idx_val = tl.cast(kh * 3 + kw, tl.int8)
            best_idx = tl.where(is_better, idx_val, best_idx)

    # Store outputs
    out_base = b * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out
    tl.store(out_val_ptr + out_base + ow_offs, best_val, mask=ow_mask)
    tl.store(out_idx_ptr + out_base + ow_offs, best_idx, mask=ow_mask)


def triton_fused_oracle(
    arg22_1: torch.Tensor,       # mean [192]
    convolution_4: torch.Tensor, # input [128, 192, 71, 71]
    arg23_1: torch.Tensor,       # var [192]
    arg24_1: torch.Tensor,       # weight [192]
    arg25_1: torch.Tensor,       # bias [192]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused BN-affine + ReLU + MaxPool, then AvgPool on the result."""
    B, C, H_in, W_in = convolution_4.shape
    # MaxPool: kernel=3, stride=2, padding=0, dilation=1, ceil_mode=False
    H_out = (H_in - 3) // 2 + 1  # (71-3)//2 + 1 = 35
    W_out = (W_in - 3) // 2 + 1  # 35

    # Precompute inv_std
    inv_std = torch.reciprocal(torch.sqrt(arg23_1 + 0.001))

    pool_val = torch.empty(B, C, H_out, W_out, device=convolution_4.device, dtype=torch.float32)
    pool_idx = torch.empty(B, C, H_out, W_out, device=convolution_4.device, dtype=torch.int8)

    BLOCK_OW = 32
    n_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
    grid = (B * C * H_out * n_ow_blocks,)

    fused_bn_relu_maxpool_kernel[grid](
        convolution_4, arg22_1, inv_std, arg24_1, arg25_1,
        pool_val, pool_idx,
        B, C, H_in, W_in, H_out, W_out,
        BLOCK_OW=BLOCK_OW,
    )

    # AvgPool on top of the max-pooled output: kernel=3, stride=1, padding=1
    # This is a separate small kernel; the big savings are from eliminating the
    # BN+ReLU intermediates above.
    avg_out = torch.nn.functional.avg_pool2d(pool_val, kernel_size=3, stride=1, padding=1)

    return pool_idx, avg_out


def torch_direct_oracle(
    arg22_1: torch.Tensor,
    convolution_4: torch.Tensor,
    arg23_1: torch.Tensor,
    arg24_1: torch.Tensor,
    arg25_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation (no fusion, for correctness reference)."""
    inv_std = torch.reciprocal(torch.sqrt(arg23_1 + 0.001))
    x = (convolution_4 - arg22_1[None, :, None, None]) * inv_std[None, :, None, None]
    x = x * arg24_1[None, :, None, None] + arg25_1[None, :, None, None]
    x = torch.relu(x)
    result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        x, [3, 3], [2, 2], [0, 0], [1, 1], False
    )
    pool_val = result[0]
    pool_idx = result[1]
    avg_out = torch.nn.functional.avg_pool2d(pool_val, kernel_size=3, stride=1, padding=1)
    return pool_idx, avg_out


def make_inputs(device: torch.device) -> tuple:
    module = _load_repro_module()
    inputs = module.make_inputs()
    moved = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    diffs = []
    for a, e in zip(actual, expected):
        if a.is_floating_point():
            diffs.append((a.float() - e.float()).abs().max().item())
        else:
            diffs.append((a.long() - e.long()).abs().max().item())
    return max(diffs)


def allclose_check(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    for a, e in zip(actual, expected):
        if a.is_floating_point():
            if not torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol):
                return False
        else:
            if not torch.equal(a, e):
                return False
    return True


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def append_csv(
    path: Path,
    impl: str,
    device: torch.device,
    oracle_us: float,
    correct: str,
    diff: float,
    args: argparse.Namespace,
) -> None:
    row = {
        "repro_id": REPRO_ID,
        "repro_path": str(REPRO_PATH.relative_to(REPO_ROOT)),
        "family": "layout_indexing_stencil_fusion",
        "oracle_impl": impl,
        "oracle_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "hardware": args.hardware,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "git_commit": get_git_commit(),
        "oracle_us": oracle_us,
        "correct": correct,
        "max_abs_diff": diff,
        "tolerance": f"rtol={args.rtol},atol={args.atol}" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Fused BN-affine+ReLU+MaxPool avoids [128,192,71,71] intermediates; AvgPool separate.",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", choices=("triton-fused", "torch-direct"), default="triton-fused")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Compare oracle output to captured repro.py.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-bench", action="store_true", help="Skip timing and CSV append.")
    parser.add_argument("--no-append", action="store_true", help="Do not append measured_oracle_floors.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    return parser.parse_args()


@oracle_impl(hardware="H100", shapes="(T([192], f32), T([128, 192, 71, 71], f32), T([192], f32), T([192], f32), T([192], f32))")
def oracle_forward(inputs):
    return triton_fused_oracle(*inputs)


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
