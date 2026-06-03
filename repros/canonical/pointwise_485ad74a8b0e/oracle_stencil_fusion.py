"""
Oracle for pointwise_485ad74a8b0e: fused ReLU + MaxPool2d.

Pattern: relu -> _low_memory_max_pool_with_offsets (kernel_size=3, stride=2, ceil_mode=True)
Shape: [512, 64, 111, 111] f16

Optimization: the compiled code materializes the ReLU output as a full [512,64,111,111]
f16 tensor, then reads it back for the pooling stencil. The oracle fuses ReLU into the
pooling kernel so the intermediate is never written to global memory.

This is a Triton kernel that reads the input once, applies ReLU element-wise in registers,
then computes the 3x3 max-pool with stride 2 (ceil_mode=True) producing:
  - pooled values: f16[512, 64, 55, 55]
  - pool offsets (argmax within 3x3 window): i8[512, 64, 55, 55]
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


REPRO_ID = "pointwise_485ad74a8b0e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def fused_relu_maxpool_kernel(
    input_ptr,
    out_val_ptr,
    out_idx_ptr,
    N: tl.constexpr,  # batch * channels
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program handles one (n, oh, ow) output element."""
    pid = tl.program_id(0)
    # pid indexes into [N * H_out * W_out]
    n = pid // (H_out * W_out)
    rem = pid % (H_out * W_out)
    oh = rem // W_out
    ow = rem % W_out

    # Stride-2 pooling with kernel 3, ceil_mode=True, no padding
    ih_start = oh * 2
    iw_start = ow * 2

    # Base pointer for this (n) plane
    base_offset = n * H_in * W_in

    # Compute max over the 3x3 window with ReLU fusion
    best_val = tl.cast(-65504.0, tl.float16)  # f16 min
    best_idx = tl.cast(0, tl.int8)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = ih_start + kh
            iw = iw_start + kw
            valid = (ih < H_in) & (iw < W_in)
            offset = base_offset + ih * W_in + iw
            val = tl.load(input_ptr + offset, mask=valid, other=0.0)
            # Fuse ReLU: max(val, 0)
            val = tl.maximum(val, tl.cast(0.0, tl.float16))
            is_better = val > best_val
            best_val = tl.where(is_better, val, best_val)
            best_idx = tl.where(is_better, tl.cast(kh * 3 + kw, tl.int8), best_idx)

    # Store outputs
    out_offset = n * H_out * W_out + oh * W_out + ow
    tl.store(out_val_ptr + out_offset, best_val)
    tl.store(out_idx_ptr + out_offset, best_idx)


def triton_fused_relu_maxpool(convolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused ReLU + MaxPool2d via Triton, avoiding the relu intermediate."""
    assert convolution.is_contiguous()
    B, C, H_in, W_in = convolution.shape
    # ceil_mode with kernel=3, stride=2, padding=0, dilation=1
    H_out = (H_in - 1 + 2) // 2  # = ceil((H_in - 3 + 1) / 2) for ceil_mode
    W_out = (W_in - 1 + 2) // 2

    # For ceil_mode: H_out = ceil((H_in - kernel_size) / stride) + 1
    # = ceil((111 - 3)/2) + 1 = ceil(54) + 1 = 55
    H_out = math.ceil((H_in - 3) / 2) + 1
    W_out = math.ceil((W_in - 3) / 2) + 1

    N = B * C
    out_val = torch.empty(B, C, H_out, W_out, device=convolution.device, dtype=convolution.dtype)
    out_idx = torch.empty(B, C, H_out, W_out, device=convolution.device, dtype=torch.int8)

    # Flatten batch and channel dimensions for the kernel
    input_flat = convolution.view(N, H_in, W_in)
    out_val_flat = out_val.view(N, H_out, W_out)
    out_idx_flat = out_idx.view(N, H_out, W_out)

    n_elements = N * H_out * W_out
    grid = (n_elements,)

    fused_relu_maxpool_kernel[grid](
        input_flat,
        out_val_flat,
        out_idx_flat,
        N, H_in, W_in, H_out, W_out,
        BLOCK_SIZE=1,
    )

    return out_val, out_idx


def torch_direct_oracle(convolution: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch oracle: fuse ReLU into pool via torch operations."""
    relu_out = torch.relu(convolution)
    # Use the same low_memory_max_pool_with_offsets as the repro
    result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu_out, [3, 3], [2, 2], [0, 0], [1, 1], True
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
        "notes": "Fused ReLU+MaxPool avoids materializing [512,64,111,111] f16 relu intermediate.",
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
    parser.add_argument("--rtol", type=float, default=1e-3)
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
        oracle_fn = lambda *xs: triton_fused_relu_maxpool(*xs)
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
