"""
Oracle for repro sum_3ee4028cab37 (structured_pool_upsample_backward_reduce, VGG16).

Gap diagnosis (classification: SCATTER_REDUCE): this repro is a structured
max-pool-backward scatter_add feeding a boolean mask and one channel reduction.
Inductor currently materializes the [16384, 12544] scatter_add output before
applying where and sum as generic consumers; the missing optimization is
SCATTER_REDUCE support that recognizes the pool-backward scatter indices and
accumulates the masked per-channel sum from the source values without building
the full scatter matrix.

Pattern:
    max_pool backward (scatter_add) -> where mask -> sum reduction over [batch, H, W].

    Original computation:
      1. full_default = zeros [16384, 12544]     # (batch*channels) x (out_H * out_W)
      2. scatter_add (pool backward) -> [16384, 12544]
      3. reshape to [128, 128, 112, 112]
      4. where(mask, 0, scatter_result) -> zero where mask is True
      5. sum over [0, 2, 3] -> [128]  (per-channel reduction)

Key optimization insight:
    The scatter_add materializes a [16384, 12544] (~200M element, ~800MB) intermediate
    that is immediately reduced via where+sum. Since the final reduction sums over
    batch and spatial dims, we can avoid materializing the scatter output entirely:

    For each source value at (batch, channel, src_h, src_w):
      - Compute the destination spatial index (position in the 112x112 output grid)
      - Check the boolean mask at (batch, channel, dst_h, dst_w)
      - If mask is False (not masked out), accumulate the value into channel sum

    This turns the scatter + reduce into a gather-mask-reduce over the source tensor,
    reducing memory traffic from ~800MB write + ~800MB read to just reading the source
    (~100MB) + mask lookups.

    Additionally, since where(mask, 0, val) zeros out masked positions before summing,
    any source value whose destination is masked contributes nothing. We simply skip it.

Implementation:
    torch_fused_scatter_reduce: Pure PyTorch using advanced indexing (no Triton).
    Avoids the large intermediate via direct gather from mask.
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


REPRO_ID = "sum_3ee4028cab37"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"



def _load_repro_module():
    spec = importlib.util.spec_from_file_location("sum_3ee4028cab37_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_fused_scatter_reduce(
    getitem_24: "f32[128, 128, 56, 56]",
    arg22_1: "i8[128, 128, 56, 56]",
    arg45_1: "b8[128, 128, 112, 112]",
    full: "f32[]",
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> torch.Tensor:
    """Fused scatter-reduce oracle that avoids materializing the scatter output.

    Strategy:
        Instead of scatter_add -> where -> sum, we:
        1. Compute max pool indices (same as original)
        2. For each source value, compute its destination spatial position in 112x112
        3. Check if the mask is False at that position (meaning value contributes)
        4. Accumulate unmasked values per channel

    Since where(mask, 0, scatter_result) zeros masked positions, and the final sum
    reduces batch and spatial dims, we only need to sum source values whose
    destination positions have mask=False.

    The full scalar is 0 (from the repro: full.default([16384, 12544], 0)), and
    where(mask, full=0, val) means masked positions contribute 0. So the sum is
    simply: sum of source values at unmasked destination positions, per channel.
    """
    N, C, src_H, src_W = getitem_24.shape  # [128, 128, 56, 56]
    out_H, out_W = 112, 112
    full_val = full.item()

    # Compute max pool indices from low-memory offsets
    # kernel_size=[2,2], output_size=[112,112], stride=[2,2], padding=[0,0], dilation=[1,1]
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg22_1, [2, 2], [112, 112], [2, 2], [0, 0], [1, 1]
    )  # [128, 128, 56, 56] with values in [0, 12544) = [0, 112*112)

    # Each index value is a flat position in the 112x112 output grid.
    # Decode to (out_h, out_w) for mask lookup
    idx_h = indices_abs // out_W  # [128, 128, 56, 56]
    idx_w = indices_abs % out_W   # [128, 128, 56, 56]

    # The where operation: where(mask, full_val, scatter_result)
    # - Where mask is True: position contributes full_val (=0) to sum
    # - Where mask is False: position contributes scatter_result value
    #
    # But wait - the scatter_add accumulates source values at destination positions.
    # Multiple source positions may map to the same destination. The final sum
    # reduces over all spatial positions, so duplicate contributions at the same
    # destination all get summed together anyway. The only thing that matters is
    # whether a source value's destination is masked or not.
    #
    # However, we also need to account for full_val contributions at positions that
    # have mask=True. Since full_val=0, those contribute nothing.
    #
    # Actually, if full_val != 0 we need:
    #   Part A: full_val * count(mask==True per channel, across batch and spatial)
    #   Part B: sum of source values whose destination has mask==False
    #
    # For this repro, full_val comes from aten.full.default with value 0, so Part A = 0.
    # But let's be general:

    if full_val == 0.0:
        # Only Part B matters: sum source values at unmasked destinations
        # Look up mask at each source value's destination position
        # mask shape: [128, 128, 112, 112], need to gather at (batch, channel, idx_h, idx_w)

        # Use advanced indexing to look up mask values
        batch_idx = torch.arange(N, device=getitem_24.device).reshape(N, 1, 1, 1).expand_as(idx_h)
        chan_idx = torch.arange(C, device=getitem_24.device).reshape(1, C, 1, 1).expand_as(idx_h)

        # Gather mask at destination positions
        mask_at_dst = arg45_1[batch_idx, chan_idx, idx_h, idx_w]  # [128, 128, 56, 56] bool

        # Where mask is True at destination, source value doesn't contribute
        # Where mask is False at destination, source value contributes
        masked_src = torch.where(mask_at_dst, torch.zeros_like(getitem_24), getitem_24)

        # Sum over batch(0) and spatial(2,3) -> [128] per channel
        result = masked_src.sum(dim=(0, 2, 3))  # [128]
    else:
        # General case with non-zero full_val
        # Part A: full_val * number of True mask positions per channel
        mask_true_count = arg45_1.sum(dim=(0, 2, 3)).float()  # [128]
        part_a = full_val * mask_true_count

        # Part B: sum of source values at unmasked destinations
        batch_idx = torch.arange(N, device=getitem_24.device).reshape(N, 1, 1, 1).expand_as(idx_h)
        chan_idx = torch.arange(C, device=getitem_24.device).reshape(1, C, 1, 1).expand_as(idx_h)
        mask_at_dst = arg45_1[batch_idx, chan_idx, idx_h, idx_w]  # [128, 128, 56, 56] bool

        masked_src = torch.where(mask_at_dst, torch.zeros_like(getitem_24), getitem_24)
        part_b = masked_src.sum(dim=(0, 2, 3))  # [128]

        result = part_a + part_b

    return result


def torch_naive_reference(
    getitem_24, arg22_1, arg45_1, full,
    _shape_param_0, _shape_param_1, _shape_param_2,
) -> torch.Tensor:
    """Naive implementation matching repro.py exactly (for correctness comparison)."""
    full_default = torch.ops.aten.full.default(
        [16384, 12544], 0, dtype=torch.float32, layout=torch.strided,
        device=getitem_24.device, pin_memory=False,
    )
    view_default = torch.ops.aten.view.default(getitem_24, _shape_param_0)
    indices_abs = torch.ops.prims._low_memory_max_pool_offsets_to_indices(
        arg22_1, [2, 2], [112, 112], [2, 2], [0, 0], [1, 1]
    )
    view_default_1 = torch.ops.aten.view.default(indices_abs, _shape_param_1)
    scatter_add_default = torch.ops.aten.scatter_add.default(
        full_default, 1, view_default_1, view_default,
    )
    view_default_2 = torch.ops.aten.view.default(scatter_add_default, _shape_param_2)
    where_self = torch.ops.aten.where.self(arg45_1, full, view_default_2)
    sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3])
    return sum_dim_int_list


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
        "shape_label": "torchbench_vgg16_train_001_26bbe98b",
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
        "notes": "Fused scatter-reduce oracle: bypasses [16384x12544] intermediate via gather-mask-reduce over source.",
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
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Verify oracle vs repro.")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    bench_group = parser.add_mutually_exclusive_group()
    bench_group.add_argument("--bench", dest="no_bench", action="store_false", help="Run timing.")
    bench_group.add_argument("--no-bench", dest="no_bench", action="store_true", help="Skip timing.")
    parser.set_defaults(no_bench=False)
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
        oracle_fn = torch_naive_reference

    with torch.no_grad():
        out = oracle_fn(*inputs)
        synchronize(device)

        correct = "not_checked"
        diff = math.nan
        if args.check:
            ref = torch_naive_reference(*inputs)
            synchronize(device)
            diff = (out.float() - ref.float()).abs().max().item()
            correct = str(torch.allclose(out.float(), ref.float(), rtol=args.rtol, atol=args.atol))
            print(f"correct={correct} max_abs_diff={diff:.6g}")

        if args.no_bench:
            return

        oracle_us = benchmark(lambda: oracle_fn(*inputs), device, args.warmup, args.rep)
        print(f"oracle_us={oracle_us:.3f} impl={args.impl} device={device} warmup={args.warmup} rep={args.rep}")

        if not args.no_append:
            append_csv(args.out, args.impl, device, oracle_us, correct, diff, args)


if __name__ == "__main__":
    main()
