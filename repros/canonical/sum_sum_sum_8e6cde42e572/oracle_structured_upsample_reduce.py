"""
Canonical-local oracle scaffold for repro sum_sum_sum_8e6cde42e572.

Repro pattern:
    Max-unpool-like scatter_add upsampling is added to a skip tensor, gated by
    a BN-affine/ReLU mask, then reduced per channel for batch-norm-backward
    style outputs in ``repro.py``.

This file is intentionally local to the canonical repro so candidate fused
scatter/unpool + reduction kernels can be developed and measured without
changing the captured repro.  The runnable implementation below is a direct
PyTorch formulation that keeps the structured scatter_add and BN/ReLU channel
reductions explicit.  The Triton entry point is a scaffold with detailed TODOs
because an exact performant kernel needs to fuse index decoding, scatter
accumulation, masking, and two dependent channel reductions.
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



REPRO_ID = "sum_sum_sum_8e6cde42e572"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
CHANNEL_REDUCTION_SCALE = 2.0366266944734097e-07
SHAPE_LABEL = "torchbench_pytorch_unet_train_001_90a2f096"



def structured_maxpool_scatter_add(
    getitem_48: torch.Tensor,
    arg53_1: torch.Tensor,
    shape_param_0,
    shape_param_1,
    shape_param_2,
) -> torch.Tensor:
    """Direct PyTorch version of low-memory max-pool indices + scatter_add."""
    flat_values = torch.ops.aten.view.default(getitem_48, shape_param_0)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg53_1,
        [2, 2],
        [640, 959],
        [2, 2],
        [0, 0],
        [1, 1],
    )
    flat_indices = torch.ops.aten.view.default(indices, shape_param_1)
    flat_out = torch.zeros(
        (flat_values.shape[0], 640 * 959),
        device=getitem_48.device,
        dtype=getitem_48.dtype,
    )
    scattered = torch.ops.aten.scatter_add.default(flat_out, 1, flat_indices, flat_values)
    return torch.ops.aten.view.default(scattered, shape_param_2)


def torch_direct_oracle(
    getitem_6: torch.Tensor,
    getitem_48: torch.Tensor,
    arg53_1: torch.Tensor,
    arg49_1: torch.Tensor,
    arg50_1: torch.Tensor,
    arg51_1: torch.Tensor,
    arg4_1: torch.Tensor,
    arg5_1: torch.Tensor,
    full: torch.Tensor,
    shape_param_0,
    shape_param_1,
    shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation of the repro's returned reductions.

    The graph has three logical phases:

    * reconstruct the max-unpool-like ``scatter_add`` from pooled values and
      low-memory max-pool offsets;
    * add the scattered tensor to the first 64 channels of the skip tensor and
      apply the BN-affine/ReLU gate from ``arg49_1``/BN parameters;
    * reduce the masked value and masked-centered value per channel, then form
      the two BN-backward-style returned vectors.
    """
    scattered = structured_maxpool_scatter_add(
        getitem_48,
        arg53_1,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    )
    upsample_plus_skip = getitem_6[:, :64, :, :] + scattered

    centered_for_gate = arg49_1 - arg50_1
    bn_affine = centered_for_gate * arg51_1 * arg4_1[None, :, None, None] + arg5_1[None, :, None, None]
    relu_gate = torch.relu(bn_affine)
    masked = torch.where(relu_gate <= 0, full.to(dtype=upsample_plus_skip.dtype, device=upsample_plus_skip.device), upsample_plus_skip)

    mean = arg50_1.squeeze((0, 2, 3))[None, :, None, None]
    centered = arg49_1 - mean
    masked_sum = masked.sum(dim=(0, 2, 3))
    masked_centered_sum = (masked * centered).sum(dim=(0, 2, 3))
    inv_std = arg51_1.squeeze((0, 2, 3))

    mean_term = masked_sum * CHANNEL_REDUCTION_SCALE
    variance_term = masked_centered_sum * CHANNEL_REDUCTION_SCALE * inv_std * inv_std
    normalized_grad = masked - centered * variance_term[None, :, None, None] - mean_term[None, :, None, None]
    affine_scale = inv_std * arg4_1

    out_running_mean_like = masked_centered_sum * inv_std
    out_input_like = (normalized_grad * affine_scale[None, :, None, None]).sum(dim=(0, 2, 3))
    return out_running_mean_like, out_input_like


def triton_fused_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Placeholder for the intended fused Triton implementation.

    TODO for a P0 structured max-unpool scatter + BN reduction kernel:
    1. Launch over channel tiles and destination H/W tiles for the fixed UNet
       geometry ``N=8, C=64, H=640, W=959`` while reading pooled source tiles
       with geometry ``320x479``.
    2. Reconstruct ``prims._low_memory_max_pool_offsets_to_indices`` for
       ``kernel=stride=2`` and the odd output width.  The int8 offsets encode
       the selected 2x2 location; the exact linear destination index must match
       PyTorch before any fusion measurement is trusted.
    3. Fuse the ``scatter_add``/unpool with the skip add from
       ``getitem_6[:, :64]``.  ``scatter_add`` semantics still matter if index
       decoding or future shapes create duplicate linear destinations.
    4. Compute the BN-affine/ReLU predicate from ``arg49_1``, ``arg50_1``,
       ``arg51_1``, ``arg4_1``, and ``arg5_1`` before contributing each element
       to per-channel ``sum(masked)`` and ``sum(masked * centered)`` partials.
    5. The second returned reduction depends on both channel scalars; an exact
       fused version likely needs a two-phase reduction or persistent
       channel programs that retain these scalars before producing the final
       channel sums.
    6. Compare against ``torch_direct_oracle`` with fp32 tolerances, then record
       measured floors through the CSV plumbing below.

    The direct Torch oracle remains runnable so this scaffold can be used for
    correctness checks, timing structure, and result logging today.
    """
    raise NotImplementedError(
        "Triton fused structured scatter/unpool + reduction kernel is a TODO; "
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
        "shape_label": SHAPE_LABEL,
        "family": "structured_scatter_reduce",
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
        "notes": "Torch direct structured scatter/unpool + BN/ReLU reduction scaffold; Triton exact kernel TODO.",
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


@oracle_impl(hardware="H100", shapes="(T([8, 128, 640, 959], f32), T([8, 64, 320, 479], f32), T([8, 64, 320, 479], i8, gen=Index(4)), T([8, 64, 640, 959], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), S([512, 153280]), S([512, 153280]), S([8, 64, 640, 959]))")
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
