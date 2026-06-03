"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the adaptive-average-pool backward broadcast and SiLU-gradient channel sum as a direct per-channel reduction from the `[128, 2304]` pooled gradient and `[128, 2304, 7, 7]` activation tensor, whereas Inductor currently lowers the `full -> as_strided_scatter -> as_strided -> expand -> div -> pointwise -> sum([0, 2, 3])` chain as ordinary tensor producers and consumers; Inductor cannot do this today because its scatter/view scheduler does not recognize a zero-fill structured scatter/expand whose only real consumer is a channel reduction and cannot lower that pattern to an output-centric reduction; the fix is SCATTER_REDUCE: add an FX/post-grad rewrite for zero-fill structured scatter/expand feeding pointwise channel reductions and lower it to a direct channel-reduction template."""
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

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - allows CPU-only syntax checks.
    triton = None
    tl = None


REPRO_ID = "sum_b699ca824f8e"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
SHAPE_LABEL = "timm_nfnet_l0_train_369f714a"
SPATIAL_SIZE = 49.0

sys.path.insert(0, str(REPO_ROOT))


def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def torch_direct_avgpool_backward_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    """Direct reduction for the avgpool-backward expand feeding a channel sum."""
    sigmoid = torch.reciprocal(torch.exp(torch.neg(convolution_80)) + 1.0)
    silu_backward = sigmoid * (convolution_80 * (1.0 - sigmoid) + 1.0)
    spatial_reduced = silu_backward.sum(dim=(2, 3))
    return (mm * spatial_reduced).sum(dim=0) / SPATIAL_SIZE


if triton is not None:

    @triton.jit
    def _avgpool_backward_reduce_kernel(
        mm_ptr,
        conv_ptr,
        out_ptr,
        stride_conv_n: tl.constexpr,
        stride_conv_c: tl.constexpr,
        stride_conv_h: tl.constexpr,
        stride_conv_w: tl.constexpr,
        n_size: tl.constexpr,
        c_size: tl.constexpr,
        block_c: tl.constexpr,
        block_n: tl.constexpr,
        block_rows: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_n = tl.program_id(1)
        c_offsets = pid_c * block_c + tl.arange(0, block_c)
        row_offsets = tl.arange(0, block_rows)
        n_offsets = pid_n * block_n + row_offsets // 64
        spatial_offsets = row_offsets % 64
        h_offsets = spatial_offsets // 7
        w_offsets = spatial_offsets - h_offsets * 7
        valid_spatial = spatial_offsets < 49
        mask = (
            (n_offsets[:, None] < n_size)
            & valid_spatial[:, None]
            & (c_offsets[None, :] < c_size)
        )
        conv_offsets = (
            n_offsets[:, None] * stride_conv_n
            + c_offsets[None, :] * stride_conv_c
            + h_offsets[:, None] * stride_conv_h
            + w_offsets[:, None] * stride_conv_w
        )
        mm_offsets = n_offsets[:, None] * c_size + c_offsets[None, :]
        x = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0)
        pooled_grad = tl.load(mm_ptr + mm_offsets, mask=mask, other=0.0)
        sigmoid = 1.0 / (tl.exp(-x) + 1.0)
        vals = pooled_grad * sigmoid * (x * (1.0 - sigmoid) + 1.0) * (1.0 / 49.0)
        acc = tl.sum(vals, axis=0)
        tl.atomic_add(out_ptr + c_offsets, acc, sem="relaxed", mask=c_offsets < c_size)


def triton_direct_avgpool_backward_reduce(
    mm: torch.Tensor,
    convolution_80: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda" or convolution_80.device.type != "cuda":
        raise RuntimeError("triton-direct requires CUDA inputs")

    out = torch.empty((mm.shape[1],), device=mm.device, dtype=mm.dtype)
    out.zero_()
    block_c = 16
    block_n = 8
    block_rows = block_n * 64
    grid = (triton.cdiv(mm.shape[1], block_c), triton.cdiv(mm.shape[0], block_n))
    _avgpool_backward_reduce_kernel[grid](
        mm,
        convolution_80,
        out,
        convolution_80.stride(0),
        convolution_80.stride(1),
        convolution_80.stride(2),
        convolution_80.stride(3),
        mm.shape[0],
        mm.shape[1],
        block_c,
        block_n,
        block_rows,
        num_warps=8,
    )
    return out


def make_inputs(device: torch.device) -> tuple:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
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
        "notes": "Direct avgpool-backward expand + SiLU-gradient channel reduction; avoids zero-fill as_strided_scatter/expand materialization.",
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
        choices=("triton-direct", "torch-direct"),
        default="triton-direct" if triton is not None and torch.cuda.is_available() else "torch-direct",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check", action="store_true", help="Compare oracle output to captured repro.py.")
    parser.add_argument("--bench", action="store_true", help="Benchmark the selected oracle implementation.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--append", action="store_true", help="Append benchmark results to measured_oracle_floors.csv.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--hardware", default="unknown")
    args = parser.parse_args()
    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    inputs = make_inputs(device)
    if args.impl == "triton-direct":
        oracle_fn = triton_direct_avgpool_backward_reduce
    else:
        oracle_fn = torch_direct_avgpool_backward_reduce

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
            diff = (out.float() - ref.float()).abs().max().item()
            correct = str(torch.allclose(out.float(), ref.float(), rtol=args.rtol, atol=args.atol))
            print(f"correct={correct} max_abs_diff={diff:.6g}")

        if args.bench:
            oracle_us = benchmark(lambda: oracle_fn(*inputs), device, args.warmup, args.rep)
            print(
                f"oracle_us={oracle_us:.3f} impl={args.impl} "
                f"device={device} warmup={args.warmup} rep={args.rep}"
            )
            if args.append:
                append_csv(args.out, args.impl, device, oracle_us, correct, diff, args)


if __name__ == "__main__":
    main()
