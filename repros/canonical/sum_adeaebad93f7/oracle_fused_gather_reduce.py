"""
Canonical-local oracle scaffold for repro sum_adeaebad93f7.

Repro pattern:
    Demucs training fragment that applies random sign flips, random stereo swap,
    random time-window gather, random group permutation, and finally reduces the
    four shifted sources with ``sum(dim=1)``.

This file is intentionally local to the canonical repro so candidate fused
irregular gather-reduce kernels can be developed and measured without changing
``repro.py``.  The runnable oracle below is a direct PyTorch formulation that
keeps the index math explicit and avoids materializing the final
``[16, 4, 4, 2, 382788]`` grouped gather before the source reduction.  The Triton entry
point is a scaffold with TODOs for a one-pass implementation of the same index
composition.
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
from torch import device as torch_device

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)



REPRO_ID = "sum_adeaebad93f7"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
DEFAULT_TIME = 382_788
INPUT_TIME = 426_888
MAX_OFFSET = INPUT_TIME - DEFAULT_TIME


def _inductor_random_inputs(
    device: torch.device,
    time: int = DEFAULT_TIME,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the same internal random tensors used by ``repro.py``.

    The captured repro uses Inductor prims rather than ordinary ``torch.rand``
    calls.  Keeping those prims here makes ``--check`` compare the same random
    stream when the oracle and repro are run from a fresh process.
    """
    if device.type != "cuda":
        raise RuntimeError("captured repro random prims require CUDA; use --random-source torch for CPU smoke tests")

    seeds = torch.ops.prims.inductor_seeds.default(4, device=torch_device(type="cuda", index=device.index or 0))
    group_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 3)
    group_random = torch.ops.prims.inductor_random.default([16, 4, 4, 1, 1], group_seed, "rand")
    group_sorted, group_indices = torch.ops.aten.sort.default(group_random, 1)

    sign_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 0)
    sign_bits = torch.ops.prims.inductor_randint.default(0, 2, [64, 4, 1, 1], sign_seed)
    signs = torch.ops.aten.sub.Tensor(torch.ops.aten.mul.Tensor(sign_bits.to(torch.float32), 2), 1)

    stereo_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 1)
    stereo_indices = torch.ops.prims.inductor_randint.default(0, 2, [64, 4, 1, 1], stereo_seed)

    offset_seed = torch.ops.prims.inductor_lookup_seed.default(seeds, 2)
    offsets = torch.ops.prims.inductor_randint.default(0, MAX_OFFSET + 1, [64, 4, 1, 1], offset_seed)

    if time != DEFAULT_TIME:
        offsets = torch.clamp(offsets, max=INPUT_TIME - time)
    return group_sorted, group_indices, signs, stereo_indices, offsets


def _torch_random_inputs(
    device: torch.device,
    time: int = DEFAULT_TIME,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Portable random inputs for CPU/CUDA smoke tests of the oracle structure."""
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    group_random = torch.rand((16, 4, 4, 1, 1), device=device, generator=generator)
    group_sorted, group_indices = torch.sort(group_random, dim=1)
    signs = torch.randint(0, 2, (64, 4, 1, 1), device=device, generator=generator).float().mul(2).sub(1)
    stereo_indices = torch.randint(0, 2, (64, 4, 1, 1), device=device, generator=generator)
    offsets = torch.randint(0, INPUT_TIME - time + 1, (64, 4, 1, 1), device=device, generator=generator)
    return group_sorted, group_indices, signs, stereo_indices, offsets


def torch_fused_gather_reduce_oracle(
    arg0_1: torch.Tensor,
    _shape_param_0=None,
    _shape_param_1=None,
    _shape_param_2=None,
    _shape_param_3=None,
    _shape_param_4=None,
    *,
    random_source: str = "inductor",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct PyTorch oracle for the Demucs fused gather-reduce pattern.

    The captured graph materializes this chain before reducing:

    1. Drop channel 0, producing ``x[n, source, stereo, sample]``.
    2. Apply one random sign per ``(n, source)``.
    3. Randomly swap stereo by gathering ``stereo`` and ``1 - stereo``.
    4. Gather a length-382788 time window using one random offset per
       ``(n, source)``.
    5. Reshape ``n=64`` as ``(batch=16, stem=4)``, gather stem groups with the
       sorted random indices, then sum the four source lanes.

    This oracle composes steps 4 and 5 directly: it builds the sampled
    ``[64, 4, 2, time]`` tensor, views it as ``[16, 4, 4, 2, time]``, and uses
    ``torch.take_along_dim(...).sum(dim=2)`` so the reduction intent stays
    adjacent to the irregular gather.  A Triton kernel should fuse the time
    gather, group gather, and source sum without allocating the grouped tensor.
    """
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4

    if arg0_1.ndim != 4 or arg0_1.shape[:3] != (64, 5, 2):
        raise ValueError(f"expected arg0_1 shape [64, 5, 2, T], got {tuple(arg0_1.shape)}")
    if arg0_1.shape[3] < DEFAULT_TIME:
        raise ValueError(f"input time dimension must be at least {DEFAULT_TIME}, got {arg0_1.shape[3]}")

    time = DEFAULT_TIME
    if random_source == "inductor":
        group_sorted, group_indices, signs, stereo_indices, offsets = _inductor_random_inputs(arg0_1.device, time)
    elif random_source == "torch":
        group_sorted, group_indices, signs, stereo_indices, offsets = _torch_random_inputs(arg0_1.device, time, seed)
    else:
        raise ValueError(f"unknown random_source={random_source!r}")

    signed_sources = arg0_1[:, 1:, :, :] * signs.to(dtype=arg0_1.dtype)
    stereo_pair = torch.cat(
        [
            torch.gather(signed_sources, 2, stereo_indices.expand(64, 4, 1, arg0_1.shape[3])),
            torch.gather(signed_sources, 2, (1 - stereo_indices).expand(64, 4, 1, arg0_1.shape[3])),
        ],
        dim=2,
    )

    sample_positions = torch.arange(time, device=arg0_1.device, dtype=torch.int64).view(1, 1, 1, time)
    time_indices = sample_positions + offsets.expand(64, 4, 2, 1)
    windowed = torch.gather(stereo_pair, 3, time_indices)

    grouped = windowed.view(16, 4, 4, 2, time)
    group_index = group_indices.expand(16, 4, 4, 2, time)
    reduced = torch.take_along_dim(grouped, group_index, dim=1).sum(dim=2).view(64, 2, time)
    return group_sorted, reduced


def triton_fused_gather_reduce_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """Placeholder for the intended fused Triton implementation.

    TODO for P2 irregular_gather_reduce:
    1. Launch over ``(batch16, subgroup4, stereo2, time_block)`` or over the
       flattened output ``[64, 2, 382788]``.  Each program computes one tile of
       the final reduced tensor and never stores the intermediate
       ``[16, 4, 4, 2, 382788]`` group gather.
    2. For each output element, loop over the four source-reduction lanes.  Load
       ``source_group = group_indices[batch16, output_group, reduce_lane]`` and
       map it back to the source ``n = batch16 * 4 + source_group`` while
       preserving ``output_group`` in the final flattened output view.
    3. Compose all irregular addressing in registers: channel slice ``1:5``,
       per-source sign, stereo swap/complement, and per-source time offset.
       The source address is equivalent to
       ``arg0[n, 1 + subgroup, stereo_index, t + offset[n, subgroup]]`` for one
       lane and ``arg0[n, 1 + subgroup, 1 - stereo_index, ...]`` for the other.
    4. Accumulate four fp32 lanes, cast/store to the repro output dtype, and
       return the sorted random tensor unchanged or produce it via the same
       sort side path as the captured repro.
    5. Validate against ``torch_fused_gather_reduce_oracle`` and ``repro.py``
       with fp32 tolerances before recording measured floors.
    """
    raise NotImplementedError(
        "Triton fused irregular gather-reduce kernel is a TODO; "
        "use --impl torch-fused for the runnable scaffold."
    )


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
        "shape_label": "demucs_train_64x5x2x426888_default",
        "family": "irregular_gather_reduce",
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
        "notes": "Torch direct fused-index scaffold; see Triton TODO for one-pass irregular gather-reduce plan.",
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
    parser.add_argument("--impl", choices=("torch-fused", "triton-todo"), default="torch-fused")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--random-source",
        choices=("inductor", "torch"),
        default="inductor",
        help="Use captured Inductor random prims or portable torch RNG for oracle-only smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used only with --random-source torch.")
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


def oracle_forward(inputs):
    return triton_fused_gather_reduce_oracle(*inputs)


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
