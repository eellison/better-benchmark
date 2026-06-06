"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer attention-head layout materialization by directly filling the final contiguous `[B*12*4, 256, 64]` output storage from the contiguous `[B*1024, 768]` input in one Triton layout-copy kernel, folding the intermediate clone/view/size-one-permute chain into the final store map, whereas Inductor lowers the same view/permute/view/permute/clone/view/permute/view chain through a more generic pointwise copy schedule; Inductor cannot do this today because layout-copy codegen does not specialize this Longformer head-layout family into direct final-view stores across the split sequence blocks; the fix is NEW_PATTERN: add a direct final-view layout-copy template for B/S/H/D-to-(B*H*blocks, tile, D) materializations with metadata-only tail views eliminated."""
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


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_VECTORS": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_VECTORS": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_VECTORS": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_VECTORS": 64}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_VECTORS", "S", "H", "D"],
    )
    @triton.jit
    def _longformer_head_layout_kernel(
        input_ptr,
        output_ptr,
        TOTAL_VECTORS: tl.constexpr,
        S: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        BLOCK_VECTORS: tl.constexpr,
    ):
        vector = tl.program_id(0) * BLOCK_VECTORS + tl.arange(0, BLOCK_VECTORS)
        dim = tl.arange(0, D)
        mask = vector < TOTAL_VECTORS

        seq = vector % S
        batch_head = vector // S
        head = batch_head % H
        batch = batch_head // H

        input_offsets = (
            (batch[:, None] * S + seq[:, None]) * H * D
            + head[:, None] * D
            + dim[None, :]
        )
        output_offsets = vector[:, None] * D + dim[None, :]
        values = tl.load(input_ptr + input_offsets, mask=mask[:, None], other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask[:, None])


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"expected shape list/tuple, got {type(value)!r}")
    return tuple(int(dim) for dim in value)


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _resolve_view_shape(value: Any, numel: int) -> tuple[int, ...]:
    dims = list(_shape_tuple(value))
    neg_one_count = dims.count(-1)
    if neg_one_count > 1:
        raise ValueError(f"only one inferred dimension is valid, got {dims}")
    if neg_one_count == 1:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        if known == 0 or numel % known != 0:
            raise ValueError(f"cannot infer shape {dims} for numel={numel}")
        dims[dims.index(-1)] = numel // known
    resolved = tuple(dims)
    if _numel(resolved) != numel:
        raise ValueError(f"shape {resolved} has {_numel(resolved)} elements, expected {numel}")
    return resolved


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], tuple[int, int, int], int, int, int, int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    mm_137, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(mm_137, torch.Tensor):
        raise TypeError(f"{REPRO_ID} first input must be a tensor")
    if not mm_137.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if mm_137.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {mm_137.dtype}")
    if mm_137.ndim != 2:
        raise ValueError(f"{REPRO_ID} expects a rank-2 input, got shape={tuple(mm_137.shape)}")
    if not mm_137.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={tuple(mm_137.stride())}")

    input_rows = int(mm_137.shape[0])
    hidden = int(mm_137.shape[1])
    total = int(mm_137.numel())

    batch, seq, view_hidden = _resolve_view_shape(shape0, total)
    if (batch * seq, view_hidden) != (input_rows, hidden):
        raise ValueError(
            f"first view shape {(batch, seq, view_hidden)} does not match input "
            f"shape={tuple(mm_137.shape)}"
        )

    seq1, batch1, heads, head_dim = _resolve_view_shape(shape1, total)
    if (seq1, batch1, heads * head_dim) != (seq, batch, hidden):
        raise ValueError(
            f"head view shape {(seq1, batch1, heads, head_dim)} is incompatible "
            f"with first view {(batch, seq, hidden)}"
        )

    clone_view_shape = _resolve_view_shape(shape2, total)
    if len(clone_view_shape) != 4:
        raise ValueError(f"clone view shape must be rank 4, got {clone_view_shape}")
    expected_first = batch * heads
    if clone_view_shape[0] != expected_first or clone_view_shape[3] != head_dim:
        raise ValueError(
            f"clone view shape {clone_view_shape} must start with {expected_first} "
            f"and end with {head_dim}"
        )
    seq_blocks = int(clone_view_shape[1])
    seq_tile = int(clone_view_shape[2])
    if seq_blocks * seq_tile != seq:
        raise ValueError(
            f"clone view sequence split {(seq_blocks, seq_tile)} does not cover seq={seq}"
        )

    size_one_view_shape = _resolve_view_shape(shape3, total)
    expected_size_one_shape = clone_view_shape + (1,)
    if size_one_view_shape != expected_size_one_shape:
        raise ValueError(
            f"size-one view shape {size_one_view_shape} must equal {expected_size_one_shape}"
        )

    output_shape = _resolve_view_shape(shape4, total)
    expected_output_shape = (expected_first * seq_blocks, seq_tile, head_dim)
    if output_shape != expected_output_shape:
        raise ValueError(f"output shape {output_shape} must equal {expected_output_shape}")

    output_stride = _contiguous_stride(output_shape)
    total_vectors = batch * heads * seq
    return mm_137, output_shape, output_stride, total_vectors, seq, heads, head_dim


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
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    mm_137, output_shape, output_stride, total_vectors, seq, heads, head_dim = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        output_stride,
        device=mm_137.device,
        dtype=mm_137.dtype,
    )
    grid = lambda meta: (triton.cdiv(total_vectors, meta["BLOCK_VECTORS"]),)
    _longformer_head_layout_kernel[grid](
        mm_137,
        output,
        TOTAL_VECTORS=total_vectors,
        S=seq,
        H=heads,
        D=head_dim,
    )
    return output


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
