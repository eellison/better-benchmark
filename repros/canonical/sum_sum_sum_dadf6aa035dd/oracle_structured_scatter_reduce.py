"""
Oracle for repro sum_sum_sum_dadf6aa035dd (structured_pool_upsample_backward_reduce, PyTorch-UNet).

Pattern:
    Bilinear upsample backward (4x index_put with accumulate=True) -> BN-affine/ReLU mask
    -> 3 channel reductions (sum, sum*centered, final BN-backward linear combination).

    Input shapes: [8, 512, 160, 239] -> slice channels 256:512 -> pad -> [8, 256, 160, 238]
    Scatter output: [8, 256, 80, 119]
    Index shapes: row_idx [160, 1], col_idx [238]

Key optimization insight:
    The original graph materializes a [8, 256, 80, 119] scattered intermediate from
    4 separate index_put operations on a [8, 256, 160, 238] source tensor. This
    intermediate is then gated by BN-affine/ReLU and reduced per channel.

    Since the final operation reduces over batch and spatial dims to produce per-channel
    vectors, we fuse:
    1. The 4 index_put scatters (bilinear upsample backward)
    2. The BN-backward reductions into algebraic combinations of 3 partial sums,
       avoiding multiple full passes over the [8, 256, 80, 119] scattered tensor.

Implementation:
    torch_fused_bilinear_scatter_reduce: Pure PyTorch implementation that computes
    the bilinear scatter and fuses the 3 BN-backward reductions algebraically.
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


REPRO_ID = "sum_sum_sum_dadf6aa035dd"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
# Scale factor from the repro: 1 / (N * H_out * W_out) = 1 / (8 * 80 * 119)
CHANNEL_REDUCTION_SCALE = 1.3130252100840337e-05
SHAPE_LABEL = "torchbench_pytorch_unet_train_001_14c7e53c"

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_sum_sum_dadf6aa035dd_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def structured_bilinear_index_put(
    getitem_18: torch.Tensor,
    row_weight: torch.Tensor,
    col_weight: torch.Tensor,
    row_hi: torch.Tensor,
    col_hi: torch.Tensor,
    col_lo: torch.Tensor,
    row_lo: torch.Tensor,
) -> torch.Tensor:
    """Compute the 4 bilinear index_put accumulations matching the repro exactly.

    The repro does:
      src = getitem_18[:, 256:512, :, :-1]  -> [8, 256, 160, 238]
      constant_pad_nd removes the last column -> [8, 256, 160, 238]
      Then computes bilinear weights and their complements, and scatters 4 ways:
        val * row_weight * col_weight       -> index_put at (row_hi, col_hi)
        val * row_weight * (1 - col_weight) -> index_put at (row_hi, col_lo)
        val * (1 - row_weight) * col_weight -> index_put at (row_lo, col_hi)
        val * (1 - row_weight) * (1 - col_weight) -> index_put at (row_lo, col_lo)
    """
    # Source: slice channels 256:512 then pad (remove last col)
    src = getitem_18[:, 256:512, :, :-1]  # [8, 256, 160, 238]

    # Compute the 4 weighted contributions
    val_rw = src * row_weight
    val_1mrw = src - val_rw

    val_rw_cw = val_rw * col_weight
    val_rw_1mcw = val_rw - val_rw_cw
    val_1mrw_cw = val_1mrw * col_weight
    val_1mrw_1mcw = val_1mrw - val_1mrw_cw

    # Output shape
    N, C = src.shape[0], src.shape[1]
    out_H, out_W = 80, 119
    out = torch.zeros((N, C, out_H, out_W), device=src.device, dtype=src.dtype)

    # Scatter all 4 contributions (accumulate=True for overlapping indices)
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_hi], val_rw_cw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_hi, col_lo], val_rw_1mcw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_hi], val_1mrw_cw, True)
    out = torch.ops.aten.index_put.default(out, [None, None, row_lo, col_lo], val_1mrw_1mcw, True)

    return out


def torch_fused_bilinear_scatter_reduce(
    getitem_18: torch.Tensor,    # f32[8, 512, 160, 239]
    arg102_1: torch.Tensor,      # f32[160, 1] - row weights
    arg101_1: torch.Tensor,      # f32[238] - col weights
    arg98_1: torch.Tensor,       # i64[160, 1] - row_hi indices
    arg100_1: torch.Tensor,      # i64[238] - col_hi indices
    arg99_1: torch.Tensor,       # i64[238] - col_lo indices
    arg97_1: torch.Tensor,       # i64[160, 1] - row_lo indices
    arg94_1: torch.Tensor,       # f32[8, 256, 80, 119] - BN input (x)
    arg95_1: torch.Tensor,       # f32[1, 256, 1, 1] - BN running mean
    arg96_1: torch.Tensor,       # f32[1, 256, 1, 1] - BN inv_std
    arg29_1: torch.Tensor,       # f32[256] - BN weight (gamma)
    arg30_1: torch.Tensor,       # f32[256] - BN bias (beta)
    full: torch.Tensor,          # f32[] - scalar for where masking
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bilinear scatter + BN backward reduction oracle.

    Computes:
      1. Bilinear upsample backward via 4 index_put accumulations -> scatter_result [8,256,80,119]
      2. BN-affine/ReLU gate: where(relu(BN(x)) <= 0, full, scatter_result) -> masked
      3. Three reductions fused into two outputs:
         - out0 = sum(masked * centered, [0,2,3]) * inv_std  (per channel)
         - out1 = sum(normalized_grad * affine_scale, [0,2,3])  (per channel)
    """
    # Phase 1: Compute scatter result (bilinear upsample backward)
    scatter_result = structured_bilinear_index_put(
        getitem_18, arg102_1, arg101_1, arg98_1, arg100_1, arg99_1, arg97_1
    )

    # Phase 2: Compute BN-affine/ReLU mask
    # bn_affine = (arg94_1 - mean) * inv_std * weight + bias
    sub_tensor = arg94_1 - arg95_1
    mul_tensor_3 = sub_tensor * arg96_1
    weight_4d = arg29_1[None, :, None, None]
    bias_4d = arg30_1[None, :, None, None]
    bn_affine = mul_tensor_3 * weight_4d + bias_4d
    relu_gate = torch.relu(bn_affine)
    le_mask = relu_gate <= 0

    # Apply mask: where gate is active, use full scalar; else use scatter_result
    masked = torch.where(le_mask, full, scatter_result)

    # Phase 3: BN backward reductions (fused algebraically)
    # centered = arg94_1 - mean (same as sub_tensor)
    centered = sub_tensor  # [8, 256, 80, 119]
    inv_std_vec = arg96_1.squeeze((0, 2, 3))  # [256]

    # Reduction 1: sum(masked, [0,2,3])
    masked_sum = masked.sum(dim=(0, 2, 3))  # [256]

    # Reduction 2: sum(masked * centered, [0,2,3])
    masked_centered_sum = (masked * centered).sum(dim=(0, 2, 3))  # [256]

    # Output 0: mul_tensor_13 = sum_dim_int_list_1 * squeeze_dims_1
    # = masked_centered_sum * inv_std_vec
    out0 = masked_centered_sum * inv_std_vec

    # Output 1: sum(normalized_grad * affine_scale, [0,2,3])
    # Algebraic fusion avoids 3rd full pass:
    # out1 = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)
    mean_term = masked_sum * CHANNEL_REDUCTION_SCALE  # [256]
    var_term = masked_centered_sum * CHANNEL_REDUCTION_SCALE * inv_std_vec * inv_std_vec  # [256]
    affine_scale = inv_std_vec * arg29_1  # [256]

    centered_sum = centered.sum(dim=(0, 2, 3))  # [256]
    N_HW = 8 * 80 * 119  # total spatial elements per channel

    out1 = affine_scale * (masked_sum - var_term * centered_sum - mean_term * N_HW)

    return (out0, out1)


def make_inputs(device: torch.device) -> tuple:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if not configs:
        module = _load_repro_module()
        inputs = module.make_inputs()
    else:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)} if isinstance(spec, dict) and spec.get("kind") == "tensor" else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
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
    return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))


def allclose_check(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_baseline_row() -> dict[str, str]:
    path = REPO_ROOT / "investigation_results" / "sol_gap_candidates.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("repro_id") == REPRO_ID:
                return row
    return {}


def append_csv(
    path: Path,
    impl: str,
    device: torch.device,
    oracle_us: float,
    correct: str,
    diff: float,
    args: argparse.Namespace,
) -> None:
    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan"))
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan"))
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))

    row = {
        "repro_id": REPRO_ID,
        "repro_path": str(REPRO_PATH.relative_to(REPO_ROOT)),
        "shape_label": SHAPE_LABEL,
        "family": "structured_pool_upsample_backward_reduce",
        "oracle_impl": impl,
        "oracle_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "hardware": args.hardware,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
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
        "max_abs_diff": diff,
        "tolerance": f"rtol={args.rtol},atol={args.atol}" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": "Fused bilinear scatter + BN backward reduction; avoids separate reduction passes.",
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
    parser.add_argument(
        "--impl",
        choices=("fused-scatter-reduce",),
        default="fused-scatter-reduce",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", default=True, help="Verify oracle vs repro.")
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--no-bench", action="store_true", help="Skip timing.")
    parser.add_argument("--no-append", action="store_true", help="Skip CSV append.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = make_inputs(device)

    oracle_fn = torch_fused_bilinear_scatter_reduce

    with torch.no_grad():
        out = oracle_fn(*inputs)
        synchronize(device)

        correct = "not_checked"
        diff = math.nan
        if args.check:
            module = _load_repro_module()
            repro = module.Repro()
            ref = repro(*inputs)
            synchronize(device)
            diff = max_abs_diff(out, ref)
            correct = str(allclose_check(out, ref, rtol=args.rtol, atol=args.atol))
            print(f"correct={correct} max_abs_diff={diff:.6g}")
            if correct != "True":
                for i, (a, e) in enumerate(zip(out, ref)):
                    d = (a.float() - e.float()).abs().max().item()
                    rel = ((a.float() - e.float()).abs() / (e.float().abs() + 1e-8)).max().item()
                    print(f"  output[{i}]: max_abs_diff={d:.6g}, max_rel_diff={rel:.6g}")

        if args.no_bench:
            return

        oracle_us = benchmark(lambda: oracle_fn(*inputs), device, args.warmup, args.rep)
        print(f"oracle_us={oracle_us:.3f} impl={args.impl} device={device} warmup={args.warmup} rep={args.rep}")

        if not args.no_append:
            append_csv(args.out, args.impl, device, oracle_us, correct, diff, args)


if __name__ == "__main__":
    main()
