"""
Oracle for pointwise_71e3a6c09140: fused BatchNorm-affine + ReLU + MaxPool2d.

Pattern: BN-affine (sub, rsqrt, mul, mul, add) -> ReLU -> max_pool_with_offsets
Shape: [128, 64, 147, 147] f32 (main tensor), BN params [64]

Optimization: the compiled code materializes the full BN-affine output [128,64,147,147]
and the ReLU output [128,64,147,147] before feeding into the pooling stencil. The oracle
fuses the entire chain: for each pooling output element, it reads 3x3 input elements,
applies the BN-affine transform and ReLU in registers, then computes the max.

This eliminates two [128,64,147,147] f32 intermediate writes (the BN output and the ReLU
output), saving ~2 * 128*64*147*147*4 = ~1.4 GB of memory traffic.

Output: (pooled f32[128,64,73,73], offsets i8[128,64,73,73])
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


REPRO_ID = "pointwise_71e3a6c09140"
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
    """Each program handles one (b, c, oh) row of output, tiled over ow."""
    pid = tl.program_id(0)
    # pid indexes into [B * C * H_out * ceil(W_out / BLOCK_OW)]
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

            # Fused BN-affine: (val - mean) * inv_std * weight + bias
            val = (val - mean) * inv_std * weight + bias
            # Fused ReLU
            val = tl.maximum(val, 0.0)

            is_better = val > best_val
            best_val = tl.where(is_better, val, best_val)
            idx_val = tl.cast(kh * 3 + kw, tl.int8)
            best_idx = tl.where(is_better, idx_val, best_idx)

    # Store outputs
    out_base = b * (C * H_out * W_out) + c * (H_out * W_out) + oh * W_out
    tl.store(out_val_ptr + out_base + ow_offs, best_val, mask=ow_mask)
    tl.store(out_idx_ptr + out_base + ow_offs, best_idx, mask=ow_mask)


def triton_fused_bn_relu_maxpool(
    arg12_1: torch.Tensor,       # mean [64]
    convolution_2: torch.Tensor, # input [128, 64, 147, 147]
    arg13_1: torch.Tensor,       # var [64]
    arg14_1: torch.Tensor,       # weight [64]
    arg15_1: torch.Tensor,       # bias [64]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused BN-affine + ReLU + MaxPool2d."""
    B, C, H_in, W_in = convolution_2.shape
    # Pool: kernel=3, stride=2, padding=0, dilation=1, ceil_mode=False
    H_out = (H_in - 3) // 2 + 1  # (147-3)//2 + 1 = 73
    W_out = (W_in - 3) // 2 + 1  # 73

    # Precompute inv_std = 1/sqrt(var + eps)
    inv_std = torch.reciprocal(torch.sqrt(arg13_1 + 0.001))

    out_val = torch.empty(B, C, H_out, W_out, device=convolution_2.device, dtype=torch.float32)
    out_idx = torch.empty(B, C, H_out, W_out, device=convolution_2.device, dtype=torch.int8)

    BLOCK_OW = 32
    n_ow_blocks = (W_out + BLOCK_OW - 1) // BLOCK_OW
    grid = (B * C * H_out * n_ow_blocks,)

    fused_bn_relu_maxpool_kernel[grid](
        convolution_2, arg12_1, inv_std, arg14_1, arg15_1,
        out_val, out_idx,
        B, C, H_in, W_in, H_out, W_out,
        BLOCK_OW=BLOCK_OW,
    )

    return out_val, out_idx


def torch_direct_oracle(
    arg12_1: torch.Tensor,
    convolution_2: torch.Tensor,
    arg13_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg15_1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation (no fusion, for correctness reference)."""
    inv_std = torch.reciprocal(torch.sqrt(arg13_1 + 0.001))
    # BN affine
    x = (convolution_2 - arg12_1[None, :, None, None]) * inv_std[None, :, None, None]
    x = x * arg14_1[None, :, None, None] + arg15_1[None, :, None, None]
    x = torch.relu(x)
    result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        x, [3, 3], [2, 2], [0, 0], [1, 1], False
    )
    return result[0], result[1]


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        "notes": "Fused BN-affine+ReLU+MaxPool avoids materializing [128,64,147,147] intermediates.",
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = make_inputs(device)

    if args.impl == "triton-fused":
        oracle_fn = lambda *xs: triton_fused_bn_relu_maxpool(*xs)
    else:
        oracle_fn = lambda *xs: torch_direct_oracle(*xs)

    with torch.no_grad():
        out = oracle_fn(*inputs)
        synchronize(device)

        correct = "not_checked"
        diff = math.nan
        if args.check:
            module = _load_repro_module()
            ref_model = module.Repro().to(device)
            ref = ref_model(*inputs)
            synchronize(device)
            diff = max_abs_diff(out, ref)
            correct = str(allclose_check(out, ref, rtol=args.rtol, atol=args.atol))
            print(f"correct={correct} max_abs_diff={diff:.6g}")

        if args.no_bench:
            return

        oracle_us = benchmark(lambda: oracle_fn(*inputs), device, args.warmup, args.rep)
        print(f"oracle_us={oracle_us:.3f} impl={args.impl} device={device} warmup={args.warmup} rep={args.rep}")

        if not args.no_append:
            append_csv(args.out, args.impl, device, oracle_us, correct, diff, args)


if __name__ == "__main__":
    main()
