"""
Oracle for pointwise_8793407401ab: fused BatchNorm-affine + ReLU + bilinear upsample + pad + cat.

Pattern:
  1. BN-affine (sub, rsqrt, mul, mul, add) on [8, 64, 320, 479]
  2. ReLU
  3. Bilinear interpolation upsampling the result to [8, 64, 640, 958]
     (using precomputed index/weight tensors for both axes)
  4. Constant pad to [8, 64, 640, 959] (pad right by 1)
  5. Cat with relu_1 [8, 64, 640, 959] along dim=1 -> [8, 128, 640, 959]

Optimization: the compiled code materializes:
  - BN-affine output [8, 64, 320, 479] f32 (~39 MB)
  - ReLU output [8, 64, 320, 479] f32 (~39 MB)
  - Four _unsafe_index intermediate tensors [8, 64, 640, 958] f32 (~125 MB each)
  - Several intermediate sub/mul/add results during interpolation

The oracle fuses BN-affine + ReLU into the bilinear interpolation gather, so for each
output element of the interpolation we:
  1. Gather 4 source elements from the convolution input
  2. Apply BN-affine + ReLU to each in registers
  3. Compute the bilinear blend
  4. Write the final padded+concatenated output directly

This eliminates all intermediate materializations between BN and the final cat output.
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


REPRO_ID = "pointwise_8793407401ab"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

sys.path.insert(0, str(REPO_ROOT))


@triton.jit
def fused_bn_relu_bilinear_kernel(
    # Inputs
    conv_ptr,          # [B, C, H_in, W_in] f32
    mean_ptr,          # [C] f32
    inv_std_ptr,       # [C] f32
    weight_ptr,        # [C] f32
    bias_ptr,          # [C] f32
    relu_1_ptr,        # [B, C, H_out, W_out_final] f32 (first half of cat)
    # Outputs
    out_ptr,           # [B, 2*C, H_out, W_out_final] f32
    # Dimensions
    B: tl.constexpr,
    C: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out_interp: tl.constexpr,  # 958 (interpolation width before pad)
    W_out_final: tl.constexpr,   # 959 (after pad)
    # Interpolation scale factors
    h_scale: tl.constexpr,
    w_scale: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Fused BN + ReLU + bilinear interpolation + pad + cat (second half only).

    Each program handles one (b, c, oh) row of the interpolated output.
    The first C channels of the output come from relu_1 (pass-through copy),
    handled by a separate simple kernel.
    """
    pid = tl.program_id(0)
    n_w_blocks = (W_out_final + BLOCK_W - 1) // BLOCK_W
    bc_oh = pid // n_w_blocks
    w_block = pid % n_w_blocks

    bc = bc_oh // H_out
    oh = bc_oh % H_out
    b = bc // C
    c = bc % C

    # Load BN parameters
    mean = tl.load(mean_ptr + c)
    inv_std = tl.load(inv_std_ptr + c)
    w_bn = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    # Output column indices
    ow_offs = w_block * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < W_out_final

    # For the padded region (ow >= W_out_interp), output is 0.0
    interp_mask = ow_offs < W_out_interp

    # Compute source coordinates for bilinear interpolation
    # Row: src_h = oh * h_scale, clamped
    src_h_f = tl.cast(oh, tl.float32) * h_scale
    src_h_f = tl.maximum(src_h_f, 0.0)
    ih_lo = tl.cast(src_h_f, tl.int32)
    ih_hi = tl.minimum(ih_lo + 1, H_in - 1)
    h_frac = src_h_f - tl.cast(ih_lo, tl.float32)
    h_frac = tl.maximum(tl.minimum(h_frac, 1.0), 0.0)

    # Column: src_w = ow * w_scale, clamped
    src_w_f = tl.cast(ow_offs, tl.float32) * w_scale
    src_w_f = tl.maximum(src_w_f, 0.0)
    iw_lo = tl.cast(src_w_f, tl.int32)
    iw_hi = tl.minimum(iw_lo + 1, W_in - 1)
    w_frac = src_w_f - tl.cast(iw_lo, tl.float32)
    w_frac = tl.maximum(tl.minimum(w_frac, 1.0), 0.0)

    # Input base for (b, c)
    conv_base = b * (C * H_in * W_in) + c * (H_in * W_in)

    # Load 4 corners from convolution input and apply BN+ReLU
    # top-left (ih_hi, iw_hi) -- note: the repro uses clamp_max(floor+1, max) for "hi"
    off_hh_wh = conv_base + ih_hi * W_in + iw_hi
    off_hh_wl = conv_base + ih_hi * W_in + iw_lo
    off_hl_wh = conv_base + ih_lo * W_in + iw_hi
    off_hl_wl = conv_base + ih_lo * W_in + iw_lo

    v_hh_wh = tl.load(conv_ptr + off_hh_wh, mask=interp_mask & ow_mask, other=0.0)
    v_hh_wl = tl.load(conv_ptr + off_hh_wl, mask=interp_mask & ow_mask, other=0.0)
    v_hl_wh = tl.load(conv_ptr + off_hl_wh, mask=interp_mask & ow_mask, other=0.0)
    v_hl_wl = tl.load(conv_ptr + off_hl_wl, mask=interp_mask & ow_mask, other=0.0)

    # Apply BN-affine + ReLU to each
    v_hh_wh = tl.maximum((v_hh_wh - mean) * inv_std * w_bn + bias, 0.0)
    v_hh_wl = tl.maximum((v_hh_wl - mean) * inv_std * w_bn + bias, 0.0)
    v_hl_wh = tl.maximum((v_hl_wh - mean) * inv_std * w_bn + bias, 0.0)
    v_hl_wl = tl.maximum((v_hl_wl - mean) * inv_std * w_bn + bias, 0.0)

    # Bilinear interpolation
    # The repro computes:
    #   interp_w along columns: lerp(lo, hi, w_frac) per row
    #   interp_h along rows: lerp(lo_row, hi_row, h_frac)
    # top row (ih_hi): lerp between (ih_hi, iw_lo) and (ih_hi, iw_hi) by w_frac
    top = v_hh_wl + (v_hh_wh - v_hh_wl) * w_frac
    # bottom row (ih_lo): lerp between (ih_lo, iw_lo) and (ih_lo, iw_hi) by w_frac
    bot = v_hl_wl + (v_hl_wh - v_hl_wl) * w_frac
    # vertical lerp
    val = bot + (top - bot) * h_frac

    # Apply padding: 0 for ow >= W_out_interp
    val = tl.where(interp_mask, val, 0.0)

    # Write to second half of the cat output (channels C..2C)
    out_stride_b = 2 * C * H_out * W_out_final
    out_stride_c = H_out * W_out_final
    out_offset = b * out_stride_b + (C + c) * out_stride_c + oh * W_out_final + ow_offs
    tl.store(out_ptr + out_offset, val, mask=ow_mask)


@triton.jit
def copy_first_half_kernel(
    relu_1_ptr,   # [B, C, H, W] f32
    out_ptr,      # [B, 2*C, H, W] f32
    B: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy relu_1 into the first C channels of the output (cat dim=1)."""
    pid = tl.program_id(0)
    # pid indexes into [B * C * ceil(HW / BLOCK)]
    n_hw_blocks = (HW + BLOCK - 1) // BLOCK
    bc = pid // n_hw_blocks
    hw_block = pid % n_hw_blocks

    b = bc // C
    c = bc % C

    hw_offs = hw_block * BLOCK + tl.arange(0, BLOCK)
    hw_mask = hw_offs < HW

    # Read from relu_1
    src_offset = b * (C * HW) + c * HW + hw_offs
    val = tl.load(relu_1_ptr + src_offset, mask=hw_mask)

    # Write to first half of output
    dst_offset = b * (2 * C * HW) + c * HW + hw_offs
    tl.store(out_ptr + dst_offset, val, mask=hw_mask)


def triton_fused_oracle(
    arg93_1: torch.Tensor,       # mean [64]
    convolution_15: torch.Tensor, # input [8, 64, 320, 479]
    arg94_1: torch.Tensor,       # var [64]
    arg95_1: torch.Tensor,       # weight [64]
    arg96_1: torch.Tensor,       # bias [64]
    relu_1: torch.Tensor,        # [8, 64, 640, 959]
    _shape_param_0: torch.Tensor,  # [640, 1] (unused, shape param for view)
) -> torch.Tensor:
    """Fused BN-affine + ReLU + bilinear upsample + pad + cat."""
    B, C, H_in, W_in = convolution_15.shape  # 8, 64, 320, 479
    _, _, H_out, W_out_final = relu_1.shape   # 8, 64, 640, 959
    W_out_interp = 958  # bilinear output width before padding

    # Precompute inv_std
    inv_std = torch.reciprocal(torch.sqrt(arg94_1 + 1e-05))

    # Allocate output: [B, 2*C, H_out, W_out_final]
    out = torch.empty(B, 2 * C, H_out, W_out_final, device=convolution_15.device, dtype=torch.float32)

    # Scale factors matching the repro's iota * scale pattern
    # h: iota(640) * 0.49921752738654146 -> clamp(0) -> floor+1 -> clamp(319)
    # This is equivalent to: src_h = oh * h_scale where h_scale = 0.49921752738654146
    h_scale: float = 0.49921752738654146
    w_scale: float = 0.4994775339602926

    # Launch the fused BN+ReLU+bilinear+pad kernel for the second half
    BLOCK_W = 64
    n_w_blocks = (W_out_final + BLOCK_W - 1) // BLOCK_W
    grid_fused = (B * C * H_out * n_w_blocks,)

    fused_bn_relu_bilinear_kernel[grid_fused](
        convolution_15, arg93_1, inv_std, arg95_1, arg96_1, relu_1,
        out,
        B, C, H_in, W_in, H_out, W_out_interp, W_out_final,
        h_scale, w_scale,
        BLOCK_W=BLOCK_W,
    )

    # Copy relu_1 into the first half (channels 0..C-1)
    HW = H_out * W_out_final
    BLOCK_COPY = 1024
    n_hw_blocks = (HW + BLOCK_COPY - 1) // BLOCK_COPY
    grid_copy = (B * C * n_hw_blocks,)

    copy_first_half_kernel[grid_copy](
        relu_1, out,
        B, C, HW,
        BLOCK=BLOCK_COPY,
    )

    return out


def torch_direct_oracle(
    arg93_1: torch.Tensor,
    convolution_15: torch.Tensor,
    arg94_1: torch.Tensor,
    arg95_1: torch.Tensor,
    arg96_1: torch.Tensor,
    relu_1: torch.Tensor,
    _shape_param_0: torch.Tensor,
) -> torch.Tensor:
    """Direct PyTorch formulation matching the repro exactly (for correctness)."""
    # BN affine
    inv_std = torch.reciprocal(torch.sqrt(arg94_1 + 1e-05))
    x = (convolution_15 - arg93_1[None, :, None, None]) * inv_std[None, :, None, None]
    x = x * arg95_1[None, :, None, None] + arg96_1[None, :, None, None]
    x = torch.relu(x)

    # Bilinear interpolation using the same index arithmetic as the repro
    device = x.device
    H_out = 640
    W_out_interp = 958
    H_in = 320
    W_in = 479

    # Row indices
    h_coords = torch.arange(H_out, device=device, dtype=torch.float32) * 0.49921752738654146
    h_coords = torch.clamp(h_coords, min=0.0)
    h_lo = h_coords.to(torch.int64)
    h_hi = torch.clamp(h_lo + 1, max=H_in - 1)
    h_frac = torch.clamp(h_coords - h_lo.float(), 0.0, 1.0).view(-1, 1)

    # Col indices
    w_coords = torch.arange(W_out_interp, device=device, dtype=torch.float32) * 0.4994775339602926
    w_coords = torch.clamp(w_coords, min=0.0)
    w_lo = w_coords.to(torch.int64)
    w_hi = torch.clamp(w_lo + 1, max=W_in - 1)
    w_frac = torch.clamp(w_coords - w_lo.float(), 0.0, 1.0)

    # Gather 4 corners
    v_hh_wh = x[:, :, h_hi][:, :, :, w_hi]
    v_hh_wl = x[:, :, h_hi][:, :, :, w_lo]
    v_hl_wh = x[:, :, h_lo][:, :, :, w_hi]
    v_hl_wl = x[:, :, h_lo][:, :, :, w_lo]

    # Bilinear blend
    top = v_hh_wl + (v_hh_wh - v_hh_wl) * w_frac
    bot = v_hl_wl + (v_hl_wh - v_hl_wl) * w_frac
    interp = bot + (top - bot) * h_frac

    # Pad right by 1
    padded = torch.nn.functional.pad(interp, [0, 1, 0, 0], value=0.0)

    # Cat
    return torch.cat([relu_1, padded], dim=1)


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


def max_abs_diff(actual, expected) -> float:
    if isinstance(actual, tuple):
        return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))
    return (actual.float() - expected.float()).abs().max().item()


def allclose_check(actual, expected, rtol: float, atol: float) -> bool:
    if isinstance(actual, tuple):
        return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))
    return torch.allclose(actual.float(), expected.float(), rtol=rtol, atol=atol)


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
        "notes": "Fused BN+ReLU+bilinear upsample+pad+cat avoids [8,64,640,958] intermediate materializations.",
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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
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
        oracle_fn = lambda *xs: triton_fused_oracle(*xs)
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
