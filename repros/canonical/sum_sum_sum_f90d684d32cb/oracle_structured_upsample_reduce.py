"""
Canonical-local oracle scaffold for repro sum_sum_sum_f90d684d32cb.

Repro pattern:
    Structured bilinear/index_put upsample contributions are gated by a
    BN-affine/ReLU mask, then reduced per channel for the batch-norm backward
    style outputs in ``repro.py``.

This file is intentionally local to the canonical repro so candidate fused
structured upsample + reduction kernels can be developed and measured without
changing the captured repro.  The runnable implementation below is a direct
PyTorch formulation that keeps the bilinear scatter and BN/ReLU channel
reductions explicit.  The Triton entry point is a scaffold with detailed TODOs
because an exact performant kernel needs to fuse duplicate-index accumulation,
masking, and two dependent channel reductions.
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


REPRO_ID = "sum_sum_sum_f90d684d32cb"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
CHANNEL_REDUCTION_SCALE = 3.268828451882845e-06

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_sum_sum_f90d684d32cb_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def structured_bilinear_index_put(
    getitem_12: torch.Tensor,
    row_weight: torch.Tensor,
    col_weight: torch.Tensor,
    row_hi: torch.Tensor,
    col_hi: torch.Tensor,
    col_lo: torch.Tensor,
    row_lo: torch.Tensor,
) -> torch.Tensor:
    """Direct PyTorch version of the four accumulated bilinear index_puts."""
    src = getitem_12[:, 128:256, :, :-1]
    row_weight = row_weight.to(dtype=src.dtype)
    col_weight = col_weight.to(dtype=src.dtype)
    row_complement = 1.0 - row_weight
    col_complement = 1.0 - col_weight

    out_shape = (src.shape[0], src.shape[1], src.shape[2] // 2, src.shape[3] // 2)
    base = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    hi_hi = torch.ops.aten.index_put.default(base, [None, None, row_hi, col_hi], src * row_weight * col_weight, True)
    hi_lo = torch.ops.aten.index_put.default(base, [None, None, row_hi, col_lo], src * row_weight * col_complement, True)
    lo_hi = torch.ops.aten.index_put.default(base, [None, None, row_lo, col_hi], src * row_complement * col_weight, True)
    lo_lo = torch.ops.aten.index_put.default(base, [None, None, row_lo, col_lo], src * row_complement * col_complement, True)
    return hi_hi + hi_lo + lo_hi + lo_lo


def torch_direct_oracle(
    getitem_12: torch.Tensor,
    arg115_1: torch.Tensor,
    arg114_1: torch.Tensor,
    arg111_1: torch.Tensor,
    arg113_1: torch.Tensor,
    arg112_1: torch.Tensor,
    arg110_1: torch.Tensor,
    arg107_1: torch.Tensor,
    arg108_1: torch.Tensor,
    arg109_1: torch.Tensor,
    arg34_1: torch.Tensor,
    arg35_1: torch.Tensor,
    full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation of the repro's returned reductions.

    The graph has three logical phases:

    * scatter four structured bilinear ``index_put(..., accumulate=True)``
      contributions from channels 128:256 of the encoder feature map;
    * compute the BN-affine + ReLU gate from the skip tensor and use it to mask
      the scattered values;
    * reduce the masked upsample per channel for BN-backward-style outputs.
    """
    upsample = structured_bilinear_index_put(
        getitem_12,
        arg115_1,
        arg114_1,
        arg111_1,
        arg113_1,
        arg112_1,
        arg110_1,
    )

    inv_std_4d = arg109_1
    weight_4d = arg34_1[None, :, None, None]
    bias_4d = arg35_1[None, :, None, None]
    bn_affine = (arg107_1 - arg108_1) * inv_std_4d * weight_4d + bias_4d
    relu_gate = torch.relu(bn_affine)
    masked = torch.where(relu_gate <= 0, full.to(dtype=upsample.dtype, device=upsample.device), upsample)

    centered = arg107_1 - arg108_1
    masked_sum = masked.sum(dim=(0, 2, 3))
    masked_centered_sum = (masked * centered).sum(dim=(0, 2, 3))
    inv_std = arg109_1.squeeze((0, 2, 3))

    mean_term = masked_sum * CHANNEL_REDUCTION_SCALE
    variance_term = masked_centered_sum * CHANNEL_REDUCTION_SCALE * inv_std * inv_std
    normalized_grad = masked - centered * variance_term[None, :, None, None] - mean_term[None, :, None, None]
    affine_scale = inv_std * arg34_1

    out_running_mean_like = masked_centered_sum * inv_std
    out_input_like = (normalized_grad * affine_scale[None, :, None, None]).sum(dim=(0, 2, 3))
    return out_running_mean_like, out_input_like


def triton_fused_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Placeholder for the intended fused Triton implementation.

    TODO for a P0 structured upsample + BN reduction kernel:
    1. Launch over channel tiles and destination H/W tiles for the fixed UNet
       geometry ``N=8, C=128, H=160, W=239`` while reading source channels
       ``128:256`` from the ``320x478`` cropped tensor.
    2. Reconstruct the four bilinear ``index_put(..., accumulate=True)`` paths
       with explicit duplicate-index accumulation.  The index tensors are
       structured resize maps, but correctness must match PyTorch's accumulate
       semantics for repeated row/column indices.
    3. Compute the BN-affine/ReLU predicate from ``arg107_1``, ``arg108_1``,
       ``arg109_1``, ``arg34_1``, and ``arg35_1`` before adding each candidate
       upsample value into the per-channel partials.
    4. The second returned reduction depends on both ``sum(masked)`` and
       ``sum(masked * centered)``; an exact fused version likely needs a
       two-phase reduction or persistent channel programs that retain these
       scalars before producing the final channel sums.
    5. Compare against ``torch_direct_oracle`` with fp32 tolerances, then record
       measured floors through the CSV plumbing below.

    The direct Torch oracle remains runnable so this scaffold can be used for
    correctness checks, timing structure, and result logging today.
    """
    raise NotImplementedError(
        "Triton fused structured upsample + reduction kernel is a TODO; "
        "use --impl torch-direct for the runnable scaffold."
    )


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


def allclose(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
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
        "shape_label": "torchbench_pytorch_unet_train_001_8bfb0b5d",
        "family": "structured_upsample_reduce",
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
        "notes": "Torch direct structured upsample + BN/ReLU reduction scaffold; Triton exact kernel TODO.",
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
    parser.add_argument("--impl", choices=("torch-direct", "triton-todo"), default="torch-direct")
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

    if args.impl == "torch-direct":
        oracle_fn = torch_direct_oracle
    else:
        oracle_fn = triton_fused_oracle

    with torch.no_grad():
        out = oracle_fn(*inputs)
        synchronize(device)

        correct = "not_checked"
        diff = math.nan
        if args.check:
            module = _load_repro_module()
            if device.type != "cuda":
                module.device = lambda *unused_args, **unused_kwargs: device
            ref = module.Repro().to(device)(*inputs)
            synchronize(device)
            diff = max_abs_diff(out, ref)
            correct = str(allclose(out, ref, rtol=args.rtol, atol=args.atol))
            print(f"correct={correct} max_abs_diff={diff:.6g}")

        if args.no_bench:
            return

        oracle_us = benchmark(lambda: oracle_fn(*inputs), device, args.warmup, args.rep)
        print(f"oracle_us={oracle_us:.3f} impl={args.impl} device={device} warmup={args.warmup} rep={args.rep}")

        if not args.no_append:
            append_csv(args.out, args.impl, device, oracle_us, correct, diff, args)


if __name__ == "__main__":
    main()
