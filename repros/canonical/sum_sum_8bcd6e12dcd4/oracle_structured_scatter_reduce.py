"""
Oracle for repro sum_sum_8bcd6e12dcd4 (structured_pool_upsample_backward_reduce, SqueezeNet).

Pattern:
    max_pool backward (scatter_add) -> channel slice -> where mask -> sum reduction.

Key optimization insight:
    The original graph materializes a large [65536, 3025] scatter_add output, then
    slices channels, applies boolean masks, and reduces per channel.  Since the final
    reduction sums over batch and spatial dims, we can fuse the scatter with the mask
    check and accumulate directly into per-channel partial sums -- avoiding the
    65536*3025 = ~200M element intermediate entirely.

    For each source value at (batch, channel, src_spatial), the index tells us which
    output spatial position it lands at.  We check the mask at that position; if not
    masked, we add the value to the channel accumulator.  This turns the scatter +
    reduce into a simple gather-mask-reduce over the source tensor.

Implementation:
    1. torch_fused_scatter_reduce: Pure PyTorch using advanced indexing (no Triton).
       Avoids the large intermediate via direct gather from mask.
    2. Correctness verified against the original repro.
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


REPRO_ID = "sum_sum_8bcd6e12dcd4"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_sum_8bcd6e12dcd4_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_fused_scatter_reduce(
    getitem_54: "f32[512, 128, 27, 27]",
    arg33_1: "i8[512, 128, 27, 27]",
    arg61_1: "b8[512, 64, 55, 55]",
    full: "f32[]",
    arg62_1: "b8[512, 64, 55, 55]",
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused scatter-reduce oracle that avoids materializing the scatter output.

    Strategy:
        Instead of scatter_add -> slice -> where -> sum, we:
        1. Compute max pool indices (same as original)
        2. Reshape indices to [512, 128, 27, 27] = [batch, channels, src_h, src_w]
           Each index value is a position in the 55*55 = 3025 spatial grid.
        3. For the first 64 channels (used by slice_tensor_1 -> where_self with arg61_1):
           - Decode each index to (out_h, out_w) in the 55x55 grid
           - Check if arg61_1[batch, channel, out_h, out_w] is True (masked to zero)
           - If not masked, accumulate the source value into channel sum
        4. Same for channels 64:128 with arg62_1.

    This avoids the [65536, 3025] intermediate entirely.
    """
    N, C, src_H, src_W = getitem_54.shape  # [512, 128, 27, 27]
    out_H, out_W = 55, 55

    # Compute max pool indices from the low-memory offsets
    # _low_memory_max_pool_offsets_to_indices converts i8 offsets to i64 absolute indices
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg33_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]
    )  # [512, 128, 27, 27] with values in [0, 3025)

    # Each index value is a flat position in the 55x55 output grid.
    # Decode to (out_h, out_w)
    idx_h = indices_abs // out_W  # [512, 128, 27, 27]
    idx_w = indices_abs % out_W   # [512, 128, 27, 27]

    # For slice_tensor_1 (channels 64:128 of the 128-channel scatter output):
    # where_self = where(arg61_1, full, slice_tensor_1)
    # -> zeros where arg61_1 is True, keeps scatter value where False
    # Then sum over [0, 2, 3] (batch, h, w) -> [64]
    #
    # The scatter puts getitem_54[:, 64:128, :, :] values at the indexed positions
    # in channels 64:128 of the output. After slicing, channel c of slice_tensor_1
    # = channel (c+64) of view_default_2.
    #
    # For sum_dim_int_list (the first output):
    #   Uses channels 64:128 from getitem_54, mask arg61_1
    # For sum_dim_int_list_1 (the second output):
    #   Uses channels 0:64 from getitem_54, mask arg62_1

    full_val = full.item()

    # The where operation: where(mask, full_val, scatter_result)
    # - Where mask is True: contributes full_val to the sum
    # - Where mask is False: contributes the scatter_result (accumulated source values)
    #
    # The total sum per channel c has two parts:
    # Part A: full_val * (number of True positions in mask for channel c)
    # Part B: sum of source values whose destination position has mask=False
    #
    # Part A accounts for all 512*55*55 positions in each channel where mask is True.
    # Part B only needs to iterate over the 512*729 source positions.

    # --- Output 0: channels 64:128, mask arg61_1 ---
    # Part A: count of True positions per channel in arg61_1
    mask_true_count_1 = arg61_1.sum(dim=(0, 2, 3)).float()  # [64]
    part_a_0 = full_val * mask_true_count_1  # [64]

    # Part B: gather mask at scatter destinations for source values
    src_1 = getitem_54[:, 64:128, :, :]  # [512, 64, 27, 27]
    idx_h_1 = idx_h[:, 64:128, :, :]     # [512, 64, 27, 27]
    idx_w_1 = idx_w[:, 64:128, :, :]     # [512, 64, 27, 27]

    # Flatten the mask spatially: [512, 64, 55*55]
    mask_1_flat = arg61_1.reshape(N, 64, out_H * out_W)  # [512, 64, 3025]

    # Compute flat indices into the spatial dim
    flat_idx_1 = (idx_h_1 * out_W + idx_w_1).long()  # [512, 64, 27, 27]
    flat_idx_1_reshaped = flat_idx_1.reshape(N, 64, src_H * src_W)  # [512, 64, 729]

    # Gather mask values at destination positions
    mask_at_dst_1 = torch.gather(mask_1_flat, 2, flat_idx_1_reshaped)  # [512, 64, 729]

    # Zero out source values whose destination is masked (will be replaced by full_val already counted in Part A)
    src_1_flat = src_1.reshape(N, 64, src_H * src_W)  # [512, 64, 729]
    # Where mask is True at dst, source value doesn't contribute (full_val is used instead)
    # Where mask is False at dst, source value contributes to the scatter result at that position
    unmasked_src_1 = torch.where(mask_at_dst_1, torch.zeros_like(src_1_flat), src_1_flat)
    part_b_0 = unmasked_src_1.sum(dim=(0, 2))  # [64]

    out_0 = part_a_0 + part_b_0

    # --- Output 1: channels 0:64, mask arg62_1 ---
    mask_true_count_0 = arg62_1.sum(dim=(0, 2, 3)).float()  # [64]
    part_a_1 = full_val * mask_true_count_0  # [64]

    src_0 = getitem_54[:, 0:64, :, :]    # [512, 64, 27, 27]
    idx_h_0 = idx_h[:, 0:64, :, :]       # [512, 64, 27, 27]
    idx_w_0 = idx_w[:, 0:64, :, :]       # [512, 64, 27, 27]

    mask_0_flat = arg62_1.reshape(N, 64, out_H * out_W)  # [512, 64, 3025]
    flat_idx_0 = (idx_h_0 * out_W + idx_w_0).long()
    flat_idx_0_reshaped = flat_idx_0.reshape(N, 64, src_H * src_W)  # [512, 64, 729]

    mask_at_dst_0 = torch.gather(mask_0_flat, 2, flat_idx_0_reshaped)  # [512, 64, 729]

    src_0_flat = src_0.reshape(N, 64, src_H * src_W)  # [512, 64, 729]
    unmasked_src_0 = torch.where(mask_at_dst_0, torch.zeros_like(src_0_flat), src_0_flat)
    part_b_1 = unmasked_src_0.sum(dim=(0, 2))  # [64]

    out_1 = part_a_1 + part_b_1

    return (out_0, out_1)


def torch_naive_scatter_reduce(
    getitem_54, arg33_1, arg61_1, full, arg62_1,
    _shape_param_0, _shape_param_1, _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Naive implementation matching repro.py exactly (for correctness comparison)."""
    full_default = torch.ops.aten.full.default(
        [65536, 3025], 0, dtype=torch.float32, layout=torch.strided,
        device=getitem_54.device, pin_memory=False,
    )
    view_default = torch.ops.aten.view.default(getitem_54, _shape_param_0)
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg33_1, [3, 3], [55, 55], [2, 2], [0, 0], [1, 1]
    )
    view_default_1 = torch.ops.aten.view.default(indices_abs, _shape_param_1)
    scatter_add_default = torch.ops.aten.scatter_add.default(
        full_default, 1, view_default_1, view_default,
    )
    view_default_2 = torch.ops.aten.view.default(scatter_add_default, _shape_param_2)
    slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 1, 0, 64)
    slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 1, 64, 128)
    where_self = torch.ops.aten.where.self(arg61_1, full, slice_tensor_1)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
    where_self_1 = torch.ops.aten.where.self(arg62_1, full, slice_tensor)
    sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(where_self_1, [0, 2, 3])
    return (sum_dim_int_list, sum_dim_int_list_1)


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
        "shape_label": "torchbench_squeezenet1_1_train_001_c1f46ce0",
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
        "notes": "Fused scatter-reduce oracle: bypasses [65536,3025] intermediate via gather-mask-reduce.",
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
        choices=("fused-scatter-reduce", "naive"),
        default="fused-scatter-reduce",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--check", action="store_true", help="Verify oracle vs repro.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--no-bench", action="store_true", help="Skip timing.")
    parser.add_argument("--no-append", action="store_true", help="Skip CSV append.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = make_inputs(device)

    if args.impl == "fused-scatter-reduce":
        oracle_fn = torch_fused_scatter_reduce
    else:
        oracle_fn = torch_naive_scatter_reduce

    with torch.no_grad():
        out = oracle_fn(*inputs)
        synchronize(device)

        correct = "not_checked"
        diff = math.nan
        if args.check:
            ref = torch_naive_scatter_reduce(*inputs)
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
