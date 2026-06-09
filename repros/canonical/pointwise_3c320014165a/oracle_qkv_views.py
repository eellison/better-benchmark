"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete GPT-style Q/K/V split-view-permute scope by returning the three final `[B, H, S, D]` aliases directly from the original contiguous projection storage with the exact eager storage offsets and strides, whereas Inductor keeps the decomposed view/split/getitem/view/permute graph behind a compiled callable; Inductor cannot eliminate this today because its lowering does not canonicalize multi-output split-plus-layout metadata into direct `as_strided` alias returns with no generated work, and the fix is ALGEBRAIC_ELIMINATION: collapse these metadata-only chains to storage-offset algebra and bypass kernel generation for such multi-output view regions."""
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
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

SPLIT_SIZE = 768


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle implementation ---
#
# This repro's full scope is metadata-only. The eager outputs are aliases into
# the input storage, so a Triton materialization kernel would be a scope bug.


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
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"{name} has more than one inferred dimension: {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer {name}={dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known

    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"{name}={resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, int, list[tuple[int, int, int, int]]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    addmm_20, shape0, shape1, shape2, shape3 = inputs
    if not isinstance(addmm_20, torch.Tensor):
        raise TypeError(f"expected first input to be a tensor, got {type(addmm_20)!r}")
    if addmm_20.ndim != 2:
        raise ValueError(f"expected rank-2 projection input, got shape={tuple(addmm_20.shape)}")
    if addmm_20.dtype != torch.float32:
        raise ValueError(f"expected torch.float32 input, got {addmm_20.dtype}")
    if not addmm_20.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(addmm_20.stride())}")

    rows = int(addmm_20.shape[0])
    hidden = int(addmm_20.shape[1])
    first_view = _resolve_view_shape(shape0, int(addmm_20.numel()), "_shape_param_0")
    if len(first_view) != 3:
        raise ValueError(f"_shape_param_0 must be rank 3, got {first_view}")
    batch, seq, view_hidden = first_view
    if (batch * seq, view_hidden) != (rows, hidden):
        raise ValueError(
            f"_shape_param_0={first_view} is incompatible with input shape={tuple(addmm_20.shape)}"
        )
    if hidden != 3 * SPLIT_SIZE:
        raise ValueError(f"repro.py splits by {SPLIT_SIZE}, but input hidden dim is {hidden}")

    split_numel = batch * seq * SPLIT_SIZE
    view_shapes = [
        _resolve_view_shape(shape1, split_numel, "_shape_param_1"),
        _resolve_view_shape(shape2, split_numel, "_shape_param_2"),
        _resolve_view_shape(shape3, split_numel, "_shape_param_3"),
    ]
    for name, shape in zip(("_shape_param_1", "_shape_param_2", "_shape_param_3"), view_shapes):
        if len(shape) != 4:
            raise ValueError(f"{name} must be rank 4 for permute(0, 2, 1, 3), got {shape}")
        if shape[0] != batch or shape[1] != seq or shape[2] * shape[3] != SPLIT_SIZE:
            raise ValueError(
                f"{name}={shape} is incompatible with split shape "
                f"({batch}, {seq}, {SPLIT_SIZE})"
            )

    return addmm_20, hidden, view_shapes


def _output_specs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, list[tuple[tuple[int, ...], tuple[int, ...], int]]]:
    addmm_20, hidden, view_shapes = _validate_inputs(inputs)
    base_offset = int(addmm_20.storage_offset())
    split_offsets = (SPLIT_SIZE, 2 * SPLIT_SIZE, 0)

    specs = []
    for view_shape, split_offset in zip(view_shapes, split_offsets):
        batch, seq, heads, head_dim = view_shape
        specs.append(
            (
                (batch, heads, seq, head_dim),
                (seq * hidden, head_dim, hidden, 1),
                base_offset + split_offset,
            )
        )
    return addmm_20, specs


@oracle_impl(hardware="H100", shapes="(T([16384, 2304], f32), S([32, 512, 2304]), S([32, 512, 12, 64]), S([32, 512, 12, 64]), S([32, 512, 12, 64]))")
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
    addmm_20, specs = _output_specs(inputs)
    return tuple(
        torch.as_strided(
            addmm_20,
            size=size,
            stride=stride,
            storage_offset=storage_offset,
        )
        for size, stride, storage_offset in specs
    )


def _check_layout_alias(inputs: tuple[Any, ...] | list[Any], outputs: Any) -> bool:
    addmm_20, specs = _output_specs(inputs)
    if not isinstance(outputs, tuple):
        print(f"  SCOPE_MISMATCH: oracle output structure is {type(outputs)!r}, expected tuple")
        return False
    if len(outputs) != len(specs):
        print(f"  SCOPE_MISMATCH: oracle produces {len(outputs)} outputs, expected {len(specs)}")
        return False

    all_pass = True
    input_storage = addmm_20.untyped_storage().data_ptr()
    for index, (output, (size, stride, storage_offset)) in enumerate(zip(outputs, specs)):
        storage_ok = output.untyped_storage().data_ptr() == input_storage
        layout_ok = (
            tuple(output.shape) == size
            and tuple(output.stride()) == stride
            and int(output.storage_offset()) == storage_offset
            and output.dtype == addmm_20.dtype
            and output.device == addmm_20.device
            and storage_ok
        )
        print(
            f"  output {index} layout/alias: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(output.shape)} stride={list(output.stride())} "
            f"offset={output.storage_offset()} aliases_input={storage_ok})"
        )
        all_pass = all_pass and layout_ok
    return all_pass


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
        with torch.no_grad():
            layout_outputs = oracle_forward(inputs)
        ok = ok and _check_layout_alias(inputs, layout_outputs)
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
