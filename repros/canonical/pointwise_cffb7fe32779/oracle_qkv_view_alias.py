"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete QKV split/view/permute scope by collapsing the decomposed metadata-only chain into three direct `as_strided` aliases of the projected input at the exact storage offsets for q, k, and v, whereas Inductor lowers the captured view/split/view/permute graph through its normal compiled path and the benchmark harness sees no capturable GPU work for this repro; Inductor cannot produce a meaningful CUDA-graph device floor for it today because the whole computation is host-side tensor metadata and alias construction rather than a kernel; the fix is ALGEBRAIC_ELIMINATION: canonicalize this attention QKV metadata pattern to direct alias-return metadata, and treat empty-graph repros as not proving a GPU performance floor."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    inferred = [index for index, dim in enumerate(dims) if dim == -1]
    if len(inferred) > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")

    if inferred:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[inferred[0]] = numel // known

    resolved = tuple(dims)
    product = 1
    for dim in resolved:
        product *= dim
    if product != numel:
        raise ValueError(f"shape {resolved} has {product} elements, expected {numel}")
    return resolved


def _validate_layout(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int, int], tuple[int, int, int, int], int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_20, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_20, torch.Tensor):
        raise TypeError("first input must be a tensor")
    if addmm_20.ndim != 2:
        raise ValueError(f"expected rank-2 projected input, got shape={tuple(addmm_20.shape)}")
    if not addmm_20.is_contiguous():
        raise ValueError(f"expected contiguous projected input, got stride={tuple(addmm_20.stride())}")

    rows = int(addmm_20.shape[0])
    hidden3 = int(addmm_20.shape[1])
    numel = int(addmm_20.numel())

    batch, seq, hidden3_from_view = _resolve_view_shape(shape0, numel)
    if (batch * seq, hidden3_from_view) != (rows, hidden3):
        raise ValueError(
            f"first view shape {(batch, seq, hidden3_from_view)} is incompatible "
            f"with input shape={tuple(addmm_20.shape)}"
        )

    head_shapes = (
        _resolve_view_shape(shape1, numel // 3),
        _resolve_view_shape(shape2, numel // 3),
        _resolve_view_shape(shape3, numel // 3),
    )
    if head_shapes[1] != head_shapes[0] or head_shapes[2] != head_shapes[0]:
        raise ValueError(f"q/k/v view shapes differ: {head_shapes}")

    batch1, seq1, heads, head_dim = head_shapes[0]
    chunk = heads * head_dim
    if (batch1, seq1) != (batch, seq) or hidden3 != 3 * chunk:
        raise ValueError(
            f"head view shape {head_shapes[0]} is incompatible with "
            f"first view {(batch, seq, hidden3)}"
        )

    output_shape = (batch, heads, seq, head_dim)
    output_stride = (seq * hidden3, head_dim, hidden3, 1)
    return addmm_20, output_shape, output_stride, chunk


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm_20, output_shape, output_stride, chunk = _validate_layout(inputs)
    base_offset = int(addmm_20.storage_offset())
    return (
        addmm_20.as_strided(output_shape, output_stride, base_offset),
        addmm_20.as_strided(output_shape, output_stride, base_offset + chunk),
        addmm_20.as_strided(output_shape, output_stride, base_offset + 2 * chunk),
    )


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
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
