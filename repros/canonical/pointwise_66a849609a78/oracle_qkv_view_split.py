"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BEiT QKV reshape/permute/unbind scope as exact metadata-only `as_strided` views, returning three f32 `[B,H,N,D]` tensors that alias the original contiguous `[B*N,3*H*D]` input with eager-compatible strides and Q/K/V storage offsets, whereas Inductor already has no required tensor data movement for this view-only graph under CUDAGraph timing; Inductor cannot materially improve this local repro because the full scope is alias-preserving metadata rather than fusible arithmetic, scatter, reduction, or layout materialization; the fix is BANDWIDTH_BOUND: record it as at floor unless broader graph-wrapper metadata overhead changes both paths together."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

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


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple, torch.Size)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int, name: str) -> tuple[int, ...]:
    dims = list(_shape_tuple(value, name))
    if dims.count(-1) > 1:
        raise ValueError(f"{name} has more than one inferred dimension: {dims}")
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known <= 0 or numel % known != 0:
            raise ValueError(f"{name}={dims} cannot be inferred for numel={numel}")
        dims[dims.index(-1)] = numel // known

    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(
            f"{name}={resolved} has {_numel(resolved)} elements, expected {numel}"
        )
    return resolved


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, int, int, int, int]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    addmm_44, shape0, shape1 = inputs
    if not isinstance(addmm_44, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input 0 must be a tensor, got {type(addmm_44)!r}")
    if addmm_44.dtype is not torch.float32:
        raise TypeError(f"{REPRO_ID} expects f32 input, got {addmm_44.dtype}")
    if not addmm_44.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if addmm_44.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects rank-2 input, got {tuple(addmm_44.shape)}")
    if not addmm_44.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={addmm_44.stride()}")

    rows = int(addmm_44.shape[0])
    hidden = int(addmm_44.shape[1])
    numel = int(addmm_44.numel())

    batch, seq, view_hidden = _resolve_view_shape(shape0, numel, "_shape_param_0")
    if (batch * seq, view_hidden) != (rows, hidden):
        raise ValueError(
            f"_shape_param_0={(batch, seq, view_hidden)} is incompatible with "
            f"input shape={tuple(addmm_44.shape)}"
        )

    batch1, seq1, qkv, heads, head_dim = _resolve_view_shape(
        shape1,
        numel,
        "_shape_param_1",
    )
    if (batch1, seq1) != (batch, seq):
        raise ValueError(
            f"_shape_param_1 batch/seq {(batch1, seq1)} does not match "
            f"_shape_param_0 {(batch, seq)}"
        )
    if qkv != 3:
        raise ValueError(f"{REPRO_ID} expects QKV dimension 3, got {qkv}")
    if heads <= 0 or head_dim <= 0 or 3 * heads * head_dim != hidden:
        raise ValueError(
            f"invalid QKV factorization hidden={hidden}, heads={heads}, head_dim={head_dim}"
        )

    return addmm_44, batch, seq, heads, head_dim


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    addmm_44, batch, seq, heads, head_dim = _validate_inputs(inputs)
    hidden = 3 * heads * head_dim
    output_shape = (batch, heads, seq, head_dim)
    output_stride = (seq * hidden, head_dim, hidden, 1)
    base_offset = int(addmm_44.storage_offset())
    qkv_stride = heads * head_dim

    return (
        torch.as_strided(
            addmm_44,
            output_shape,
            output_stride,
            storage_offset=base_offset,
        ),
        torch.as_strided(
            addmm_44,
            output_shape,
            output_stride,
            storage_offset=base_offset + qkv_stride,
        ),
        torch.as_strided(
            addmm_44,
            output_shape,
            output_stride,
            storage_offset=base_offset + 2 * qkv_stride,
        ),
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
