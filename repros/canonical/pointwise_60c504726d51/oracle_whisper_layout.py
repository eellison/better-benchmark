"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full Whisper `permute.clone._unsafe_view.view` scope as a shape-specialized 64-wide Triton layout copy into the final contiguous `[12000, 384]` output, whereas Inductor currently lowers the same materialization as a generic flattened pointwise copy for the permuted clone; Inductor cannot do materially less work today because the full scope requires the f32 read/write materialization and its scheduler/codegen lacks a first-class transformer head-layout copy pattern that consistently beats the generic pointwise copy; the fix is NEW_PATTERN: add a specialized layout-copy lowering for head-major to token-major transformer activations only if it is reliably faster than the existing pointwise clone."""
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

    @triton.jit
    def _whisper_layout_kernel(
        input_ptr,
        output_ptr,
        TOTAL_VECTORS: tl.constexpr,
        SEQ: tl.constexpr,
        HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        vector_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        vector_mask = vector_offsets < TOTAL_VECTORS

        head = vector_offsets % HEADS
        tmp = vector_offsets // HEADS
        seq = tmp % SEQ
        batch = tmp // SEQ

        input_offsets = (
            batch[:, None] * HEADS * SEQ * HEAD_DIM
            + head[:, None] * SEQ * HEAD_DIM
            + seq[:, None] * HEAD_DIM
            + dim_offsets[None, :]
        )
        output_offsets = vector_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
        mask = vector_mask[:, None] & (dim_offsets[None, :] < HEAD_DIM)
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int], int, int, int, int]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    arg15_1, shape_param = inputs
    if not isinstance(arg15_1, torch.Tensor):
        raise TypeError(f"expected arg15_1 tensor, got {type(arg15_1)!r}")
    if arg15_1.ndim != 4:
        raise ValueError(f"expected rank-4 arg15_1, got shape={tuple(arg15_1.shape)}")
    if arg15_1.dtype != torch.float32:
        raise TypeError(f"expected float32 arg15_1, got {arg15_1.dtype}")
    if not arg15_1.is_contiguous():
        raise ValueError(f"arg15_1 must be contiguous, got stride={arg15_1.stride()}")

    batch, heads, seq, head_dim = (int(dim) for dim in arg15_1.shape)
    if head_dim != 64:
        raise ValueError(f"expected head dimension 64, got {head_dim}")

    out_shape = _shape_tuple(shape_param, "_shape_param_0")
    expected_shape = (batch * seq, heads * head_dim)
    if out_shape != expected_shape:
        raise ValueError(f"_shape_param_0 {out_shape} != expected {expected_shape}")

    return arg15_1, expected_shape, batch, heads, seq, head_dim


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    arg15_1, shape_param = inputs
    permute_default = torch.ops.aten.permute.default(arg15_1, [0, 2, 1, 3])
    clone_default = torch.ops.aten.clone.default(
        permute_default,
        memory_format=torch.contiguous_format,
    )
    unsafe_shape = [
        int(arg15_1.shape[0]),
        int(arg15_1.shape[2]),
        int(arg15_1.shape[1]) * int(arg15_1.shape[3]),
    ]
    unsafe_view_default = torch.ops.aten._unsafe_view.default(clone_default, unsafe_shape)
    return torch.ops.aten.view.default(unsafe_view_default, shape_param)


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
    arg15_1, out_shape, batch, heads, seq, head_dim = _validate_inputs(inputs)
    if triton is None or not arg15_1.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=arg15_1.device,
        dtype=arg15_1.dtype,
    )
    total_vectors = batch * seq * heads
    grid = lambda meta: (triton.cdiv(total_vectors, meta["BLOCK_M"]),)
    _whisper_layout_kernel[grid](
        arg15_1,
        output,
        TOTAL_VECTORS=total_vectors,
        SEQ=seq,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_D=head_dim,
        BLOCK_M=128,
        num_warps=8,
        num_stages=3,
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
