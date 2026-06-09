"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the full view/view/permute/unbind scope as three direct `as_strided` aliases into the projected QKV tensor, whereas Inductor currently routes the metadata-only chain through the compiled graph path and any benchmark capture has no real CUDA work to measure; Inductor cannot produce a meaningful device-side performance improvement for this repro today because the complete computation is alias metadata construction, not a kernel or memory transform; the fix is ALGEBRAIC_ELIMINATION: canonicalize this QKV view-unbind pattern to direct alias metadata and classify empty metadata-only captures as not proving a GPU floor."""
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
    oracle_impl,
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
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int, name: str) -> tuple[int, ...]:
    dims = list(_shape_tuple(value, name))
    inferred = [idx for idx, dim in enumerate(dims) if dim == -1]
    if len(inferred) > 1:
        raise ValueError(f"{name} may contain at most one inferred dim, got {dims}")

    if inferred:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"{name}={dims} cannot be resolved for numel={numel}")
        dims[inferred[0]] = numel // known

    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(
            f"{name} resolves to {resolved} with {_numel(resolved)} elements, "
            f"expected {numel}"
        )
    return resolved


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    running = 1
    for dim in reversed(shape):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    addmm_44, shape0_arg, shape1_arg = inputs
    if not isinstance(addmm_44, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if addmm_44.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects rank-2 input, got {tuple(addmm_44.shape)}")
    if not addmm_44.is_contiguous():
        raise ValueError(
            f"{REPRO_ID} expects the captured contiguous input layout, "
            f"got stride={tuple(addmm_44.stride())}"
        )

    numel = int(addmm_44.numel())
    shape0 = _resolve_view_shape(shape0_arg, numel, "_shape_param_0")
    shape1 = _resolve_view_shape(shape1_arg, numel, "_shape_param_1")
    if len(shape0) != 3:
        raise ValueError(f"_shape_param_0 must resolve to rank 3, got {shape0}")
    if len(shape1) != 5:
        raise ValueError(f"_shape_param_1 must resolve to rank 5, got {shape1}")

    batch, seq, three, heads, head_dim = shape1
    if shape0 != (batch, seq, three * heads * head_dim):
        raise ValueError(
            f"reshape chain is incompatible: _shape_param_0={shape0}, "
            f"_shape_param_1={shape1}"
        )
    if three != 3:
        raise ValueError(f"unbind dimension must have extent 3, got {three}")
    return addmm_44, shape0, shape1


@oracle_impl(hardware="H100", shapes="(T([25216, 2304], f32), S([128, 197, 2304]), S([128, 197, 3, 12, -1]))")
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
    addmm_44, _shape0, shape1 = _validate_inputs(inputs)
    base_strides = _contiguous_strides(shape1)
    batch, seq, _three, heads, head_dim = shape1

    output_shape = (batch, heads, seq, head_dim)
    output_stride = (base_strides[0], base_strides[3], base_strides[1], base_strides[4])
    storage_offset = int(addmm_44.storage_offset())
    qkv_stride = base_strides[2]

    return (
        addmm_44.as_strided(output_shape, output_stride, storage_offset=storage_offset),
        addmm_44.as_strided(
            output_shape,
            output_stride,
            storage_offset=storage_offset + qkv_stride,
        ),
        addmm_44.as_strided(
            output_shape,
            output_stride,
            storage_offset=storage_offset + 2 * qkv_stride,
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
