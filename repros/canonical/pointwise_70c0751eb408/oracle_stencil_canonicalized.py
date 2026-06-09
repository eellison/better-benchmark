"""
Canonical-local oracle scaffold for repro pointwise_70c0751eb408.

P1 investigation: layout_indexing_stencil_fusion.

The captured repro is the PyHPC isoneutral-mixing pointwise/stencil fragment.
Its generated graph contains hundreds of slice/select views and more than one
hundred slice_scatter/select_scatter functional updates around f64 3-D stencils.
The optimization question is how much of the 67-kernel compiled time is real
stencil arithmetic versus layout/indexing reconstruction overhead.

This file is intentionally local to the canonical repro.  It provides:

* a runnable exact direct-PyTorch oracle entry point that delegates to the
  captured repro with cloned writeback buffers, preserving semantics while
  giving this investigation a stable CLI, correctness check, benchmark harness,
  and CSV append path;
* a documented canonical stencil/layout plan for replacing slice_scatter chains
  with explicit-offset loads over contiguous logical interiors;
* a Triton scaffold with the required explicit-offset-load TODOs.

The exact direct oracle is not a target floor by itself.  It is the runnable
control.  The intended measured floor is ``--impl triton-explicit-offsets`` once
that kernel is filled in and checked against ``torch-direct``/``repro.py``.
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
from typing import Any, Callable, Iterable

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



REPRO_ID = "pointwise_70c0751eb408"
FAMILY = "layout_indexing_stencil_fusion"
SHAPE_LABEL = "torchbench_pyhpc_isoneutral_mixing_infer_000_d6b8a8e0"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
DEFAULT_CSV = REPO_ROOT / "investigation_results" / "measured_oracle_floors.csv"
BASELINE_CSV = REPO_ROOT / "investigation_results" / "per_repro_realistic_floors.csv"

# Repro.forward mutates these tensor argument positions via aten.copy_.  Clone
# them before each oracle invocation so correctness and benchmark repetitions do
# not compare against already-updated buffers.
MUTATED_INPUT_INDICES = (10, 11, 13, 16, 17, 18, 19)


class StencilRegion:
    """Logical interior slice carried by the canonical stencil plan."""

    def __init__(self, i0: int, i1: int, j0: int, j1: int, k0: int, k1: int) -> None:
        self.i0 = i0
        self.i1 = i1
        self.j0 = j0
        self.j1 = j1
        self.k0 = k0
        self.k1 = k1

    def as_slices(self) -> tuple[slice, slice, slice]:
        return (slice(self.i0, self.i1), slice(self.j0, self.j1), slice(self.k0, self.k1))

    def label(self) -> str:
        return f"i={self.i0}:{self.i1},j={self.j0}:{self.j1},k={self.k0}:{self.k1}"


INTERIOR_REGIONS = (
    StencilRegion(1, -2, 2, -2, 1, 2**63 - 1),
    StencilRegion(1, -2, 2, -2, 0, -1),
    StencilRegion(2, -2, 2, -2, 1, 2**63 - 1),
    StencilRegion(2, -2, 2, -2, 0, -1),
)


CANONICALIZATION_NOTES = """
Canonicalized floor plan:
1. Identify repeated functional updates such as sumz[:, :, ki:] += expr and the
   adjacent K_11/K_22/K_33/K_13/K_23 writebacks that currently materialize as
   slice_scatter chains.
2. Convert each logical interior update to a scatter-free stencil assignment:
   program ids map to (i, j, k, component), all reads use base pointer +
   i*stride0 + j*stride1 + k*stride2 plus small fixed neighbor offsets.
3. Keep boundary regions separate.  The high-value floor is the large interiors:
   [1:-2, 2:-2, 1:], [1:-2, 2:-2, :-1], [2:-2, 2:-2, 1:], and [2:-2, 2:-2, :-1].
4. Fuse common subexpressions for drodxe/drodze/taper and the mask/dzw/dxt/dyt
   scale factors rather than reconstructing the same sliced tensors for every
   component update.
5. Store directly to the seven mutated output tensors and return the same view
   outputs as repro.py for an exact correctness comparison.
""".strip()


TRITON_EXPLICIT_OFFSET_TODO = """
TODO for --impl triton-explicit-offsets:
- Specialize the default shape (204, 204, 26) and f64 dtype first; generalize
  from actual tensor strides only after an exact floor exists.
- Use a 2-D/3-D launch grid over interior tiles.  Recommended logical lanes are
  flattened (i, j, k) points with masks for i/j/k boundary exclusions.
- Replace every slice/select chain with explicit pointer arithmetic.  Example:
  x[i+di, j+dj, k+dk] => base + (i+di)*s0 + (j+dj)*s1 + (k+dk)*s2.
- Treat 4-D tensors [204,204,26,3] and 5-D tensors [204,204,26,2,2] as extra
  fixed component offsets folded into the base pointer computation.
- Compute drodT/drdS horizontal and vertical stencil terms once per tile, then
  form drodxe/drodze, taper = 0.5 * (tanh((-abs(slope / iso_slopec) + eps) /
  eps) + 1), and max(0, diffloc * taper) without temporary scatter tensors.
- Store the final seven copy_ destinations directly.  For returned interior
  views, either return strided views of those destinations or materialize only
  the exact tensors needed by the repro tuple.
- Validate against torch_direct_oracle with f64 tolerances before appending a
  measured floor row.
""".strip()


def clone_mutated_inputs(inputs: Iterable[Any]) -> list[Any]:
    cloned = list(inputs)
    for index in MUTATED_INPUT_INDICES:
        if index < len(cloned) and isinstance(cloned[index], torch.Tensor):
            cloned[index] = cloned[index].clone()
    return cloned


def output_fingerprint(outputs: tuple[Any, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(out.shape) for out in outputs if isinstance(out, torch.Tensor))


def torch_direct_oracle(*inputs: Any) -> tuple[Any, ...]:
    """Exact runnable direct-PyTorch oracle/control for this scaffold.

    The captured repro already expresses the stencil in PyTorch ATen terms, but
    with layout/indexing noise exposed as many slices and functional scatters.
    This function preserves that exact behavior while isolating mutable outputs
    from repeated runs.  It is the correctness reference for future canonical
    explicit-offset Triton kernels, not the final optimized oracle floor.
    """
    module = _load_repro_module()
    safe_inputs = clone_mutated_inputs(inputs)
    with torch.no_grad():
        return module.Repro()(*safe_inputs)


def triton_explicit_offsets_oracle(*_inputs: Any) -> tuple[Any, ...]:
    """Scaffold for the canonical explicit-offset stencil/layout oracle.

    See ``TRITON_EXPLICIT_OFFSET_TODO`` for the concrete kernel plan.  Keeping
    this entry point wired into the CLI makes the expected implementation shape
    clear while preventing accidental CSV rows for an unimplemented floor.
    """
    raise NotImplementedError(
        "Triton explicit-offset stencil oracle is a TODO. Use --impl torch-direct "
        "for the runnable control and --print-plan for canonicalization notes."
    )


def load_inputs(device: str) -> list[Any]:
    if device != "cuda":
        raise ValueError("repro.py hard-codes CUDA allocation sites; use --device cuda")
    module = _load_repro_module()
    return module.make_inputs()


def compare_outputs(actual: tuple[Any, ...], expected: tuple[Any, ...], rtol: float, atol: float) -> tuple[bool, float]:
    if len(actual) != len(expected):
        return False, math.inf

    max_abs_diff = 0.0
    for actual_item, expected_item in zip(actual, expected):
        if isinstance(actual_item, torch.Tensor) and isinstance(expected_item, torch.Tensor):
            diff = (actual_item - expected_item).abs().max().item() if actual_item.numel() else 0.0
            max_abs_diff = max(max_abs_diff, float(diff))
            if not torch.allclose(actual_item, expected_item, rtol=rtol, atol=atol):
                return False, max_abs_diff
        elif actual_item != expected_item:
            return False, math.inf
    return True, max_abs_diff


def benchmark(fn: Callable[[], Any], warmup: int, rep: int) -> float:
    if torch.cuda.is_available():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) * 1000.0 / rep

    for _ in range(warmup):
        fn()
    start_time = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - start_time) * 1_000_000.0 / rep


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def load_baseline_row() -> dict[str, str]:
    if not BASELINE_CSV.exists():
        return {}
    with BASELINE_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("repro_id") == REPRO_ID:
                return row
    return {}


def append_csv(
    path: Path,
    impl: str,
    device_name: str,
    oracle_us: float,
    correct: str,
    max_abs_diff: float,
    args: argparse.Namespace,
) -> None:
    baseline = load_baseline_row()
    best_compile_us = float(baseline.get("best_compile_us", "nan") or "nan")
    memcopy_sol_us = float(baseline.get("memcopy_sol_us", "nan") or "nan")
    total_bytes = int(float(baseline.get("total_bytes", "0") or 0))
    n_kernels = int(float(baseline.get("n_kernels", "0") or 0))
    oracle_path = Path(__file__).resolve().relative_to(REPO_ROOT)

    row = {
        "repro_id": REPRO_ID,
        "repro_path": str(REPRO_PATH.relative_to(REPO_ROOT)),
        "shape_label": SHAPE_LABEL,
        "family": FAMILY,
        "oracle_impl": impl,
        "oracle_path": str(oracle_path),
        "hardware": args.hardware,
        "device_name": device_name,
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
        "max_abs_diff": max_abs_diff,
        "tolerance": f"rtol={args.rtol},atol={args.atol}" if args.check else "not_checked",
        "n_warmup": args.warmup,
        "n_rep": args.rep,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Runnable control for P1 layout stencil canonicalization; final floor "
            "requires Triton explicit-offset interior loads/stores."
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical-local stencil/layout oracle scaffold.")
    parser.add_argument("--impl", choices=("torch-direct", "triton-explicit-offsets"), default="torch-direct")
    parser.add_argument("--device", default="cuda", help="Only cuda is supported because repro.py hard-codes CUDA tensors.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true", help="Compare selected oracle against an independent repro.py run.")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark the selected oracle implementation.")
    parser.add_argument("--append", action="store_true", help="Append measured_oracle_floors.csv when benchmark completes.")
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--print-plan", action="store_true", help="Print canonicalization and Triton TODO notes.")
    return parser.parse_args()


@oracle_impl(hardware="H100", shapes="(T([26], f64), T([204, 204, 26], f64), T([204, 204, 26], f64), T([204, 204, 26, 3], f64), T([26], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204, 204, 26, 3], f64), T([204, 204, 26], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([26], f64), T([204, 204, 26], f64), T([204], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26, 2, 2], f64), T([204, 204, 26], f64), T([204], f64), T([204], f64), T([204], f64))")
def oracle_forward(inputs):
    return triton_explicit_offsets_oracle(*inputs)


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
