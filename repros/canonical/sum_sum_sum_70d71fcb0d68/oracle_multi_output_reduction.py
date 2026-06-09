"""
Canonical-local oracle scaffold for repro sum_sum_sum_70d71fcb0d68.

Repro pattern:
    Three channel-wise sum outputs over N/H/W for the ConvNeXtV2 GRN/GELU
    backward-like fragment in ``repro.py``.

This file is intentionally local to the canonical repro so candidate one-pass
multi-output reduction kernels can be developed and measured without changing
the captured repro.  The runnable oracle implementation below is a direct
PyTorch formulation that keeps the three returned reductions explicit and shares
the common GELU/GRN terms.  The Triton entry point is a scaffold with detailed
TODOs because an exact performant kernel needs a tiled multi-output reduction
implementation over the non-contiguous channels-last NCHW layout.
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



REPRO_ID = "sum_sum_sum_70d71fcb0d68"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"


def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x * (torch.erf(x * 0.7071067811865476) + 1.0)


def gelu_exact_backward_factor(x: torch.Tensor) -> torch.Tensor:
    erf_term = torch.erf(x * 0.7071067811865476) + 1.0
    normal_pdf = torch.exp(x * x * -0.5) * 0.3989422804014327
    return 0.5 * erf_term + x * normal_pdf


def torch_direct_oracle(
    convolution_2: torch.Tensor,
    pow_2: torch.Tensor,
    add_5: torch.Tensor,
    getitem_164: torch.Tensor,
    primals_13: torch.Tensor,
    full_default: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct PyTorch formulation of the three repro outputs.

    This keeps the reduction intent visible for later Triton lowering:

    * output 0: sum(getitem_164 * gelu(convolution_2) * pow_2 / add_5) by C
    * output 1: sum(getitem_164) by C
    * output 2: sum(GRN/GELU backward expression) by C
    """
    del _shape_param_0, _shape_param_1, _shape_param_2

    gelu = gelu_exact(convolution_2)
    x_n = pow_2 / add_5
    weight = primals_13.reshape(1, -1, 1, 1)
    weighted_grad = getitem_164 * weight

    out_weight = (getitem_164 * gelu * x_n).sum(dim=(0, 2, 3))
    out_bias = getitem_164.sum(dim=(0, 2, 3))

    spatial_dot = (weighted_grad * gelu).sum(dim=(2, 3), keepdim=True)
    mean_backprop = (-spatial_dot * (x_n / add_5)).sum(dim=1, keepdim=True) / convolution_2.shape[1]
    norm_backprop = spatial_dot / add_5 + mean_backprop.expand_as(spatial_dot)

    safe_norm_recip_gelu = torch.where(pow_2 == 0, full_default, gelu / pow_2)
    grn_input_grad = getitem_164 + weighted_grad * x_n + norm_backprop * safe_norm_recip_gelu
    out_input = (grn_input_grad * gelu_exact_backward_factor(convolution_2)).sum(dim=(0, 2, 3))

    return out_weight, out_bias, out_input


def triton_one_pass_oracle(*args, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Placeholder for the intended one-pass Triton implementation.

    TODO for P0 multi_output_reduction_templates:
    1. Use a 2-D launch grid over channels and N/H/W tiles.  The repro tensors
       are shaped NCHW but normally channels-last contiguous with strides like
       ``(C*H*W, 1, W*C, C)``, so address arithmetic must use actual strides.
    2. In each program, load a block of C channels and a block of flattened
       N/H/W positions, compute shared GELU and GELU-gradient terms once, and
       accumulate all partials needed by the three final outputs.
    3. The third output depends on ``spatial_dot[n, c] = sum_h,w(weight[c] *
       getitem[n,c,h,w] * gelu[n,c,h,w])`` and on a per-N sum across C of a
       term derived from ``spatial_dot``.  A practical exact Triton version may
       need either a cooperative two-phase kernel or persistent programs that
       keep C tiles resident while reducing H/W, then reduce across C.
    4. Store three C-length outputs from the same reduction pipeline and compare
       against ``torch_direct_oracle`` with fp32 tolerances before recording
       measured floors.

    The direct Torch oracle remains runnable so this scaffold can be used for
    CLI correctness, benchmarking structure, and CSV append plumbing today.
    """
    raise NotImplementedError(
        "Triton one-pass multi-output reduction kernel is a TODO; "
        "use --impl torch-direct for the runnable scaffold."
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
        "shape_label": "128x320x56x56_default",
        "family": "multi_output_reduction",
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
        "notes": "Torch direct shared-expression scaffold; see Triton TODO for one-pass multi-output reduction plan.",
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


@oracle_impl(hardware="H100", shapes="(T([128, 320, 56, 56], f32, stride=(1003520, 1, 17920, 320)), T([128, 320, 1, 1], f32), T([128, 1, 1, 1], f32), T([128, 320, 56, 56], f32, stride=(1003520, 1, 17920, 320)), T([320], f32), T([], f32), S([320]), S([320]), S([128, 320, 1, 1]))")
def oracle_forward(inputs):
    return triton_one_pass_oracle(*inputs)


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
