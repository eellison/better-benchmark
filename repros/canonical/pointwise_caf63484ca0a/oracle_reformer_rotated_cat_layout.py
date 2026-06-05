"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer rotated-chunk concat layout materialization in one Triton copy kernel, directly writing the final contiguous `[B*768, 128, 64]` output from the token-major `[B*4096, 768]` input while applying the cyclic slice/cat on the first 64 columns and the unrotated concat half on the second 64 columns, whereas Inductor lowers the decomposed view/permute/view/slice/cat/cat/view graph through generic materialization kernels with per-element address reconstruction; Inductor cannot do this today because its scheduler/codegen does not recognize this fixed Reformer chunk rotation plus concat as a block-structured layout-copy pattern over 64-wide head vectors; the fix is NEW_PATTERN: add a guarded lowering for this Reformer rotated chunk concat layout copy that writes the final flattened output directly."""
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

SEQ_LEN = 4096
HEADS = 12
CHUNKS = 64
HEAD_DIM = 64
MODEL_DIM = HEADS * HEAD_DIM
CAT_COLS = CHUNKS * 2

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
        ],
        key=["SOURCE_CHUNKS"],
    )
    @triton.jit
    def _rotated_cat_layout_kernel(
        input_ptr,
        output_ptr,
        SOURCE_CHUNKS: tl.constexpr,
        SEQ: tl.constexpr,
        N_HEADS: tl.constexpr,
        N_CHUNKS: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        HIDDEN: tl.constexpr,
        OUT_COLS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        tile_n = tl.program_id(0)
        source_block = tl.program_id(1)
        n_offsets = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        chunk = source_block % N_CHUNKS
        head = (source_block // N_CHUNKS) % N_HEADS
        batch = source_block // (N_HEADS * N_CHUNKS)

        source_seq = chunk * N_CHUNKS + n_offsets

        input_offsets = (
            batch * SEQ * HIDDEN
            + source_seq[:, None] * HIDDEN
            + head * HEAD_SIZE
            + dim_offsets[None, :]
        )
        mask = (
            (source_block < SOURCE_CHUNKS)
            & (n_offsets[:, None] < N_CHUNKS)
            & (dim_offsets[None, :] < HEAD_SIZE)
        )
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)

        self_row = source_block
        next_chunk = tl.where(chunk == N_CHUNKS - 1, 0, chunk + 1)
        next_row = batch * (N_HEADS * N_CHUNKS) + head * N_CHUNKS + next_chunk

        self_offsets = (
            (self_row * OUT_COLS + (N_CHUNKS + n_offsets[:, None])) * HEAD_SIZE
            + dim_offsets[None, :]
        )
        next_offsets = (
            (next_row * OUT_COLS + n_offsets[:, None]) * HEAD_SIZE
            + dim_offsets[None, :]
        )
        tl.store(output_ptr + self_offsets, values, mask=mask)
        tl.store(output_ptr + next_offsets, values, mask=mask)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, tuple[int, int, int], int]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_16, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(mm_16, torch.Tensor):
        raise TypeError(f"expected mm_16 tensor, got {type(mm_16)!r}")
    if mm_16.ndim != 2 or int(mm_16.shape[1]) != MODEL_DIM:
        raise ValueError(f"unexpected mm_16 shape: {tuple(mm_16.shape)}")
    if int(mm_16.shape[0]) % SEQ_LEN != 0:
        raise ValueError(f"mm_16 rows must be a multiple of {SEQ_LEN}, got {mm_16.shape[0]}")
    if tuple(mm_16.stride()) != (MODEL_DIM, 1):
        raise ValueError(f"mm_16 must be contiguous with stride {(MODEL_DIM, 1)}, got {mm_16.stride()}")
    if mm_16.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected fp16/fp32 mm_16, got {mm_16.dtype}")

    batch = int(mm_16.shape[0]) // SEQ_LEN
    shape0_tuple = _shape_tuple(shape0, "_shape_param_0")
    shape1_tuple = _shape_tuple(shape1, "_shape_param_1")
    shape2_tuple = _shape_tuple(shape2, "_shape_param_2")
    shape3_tuple = _shape_tuple(shape3, "_shape_param_3")
    shape4_tuple = _shape_tuple(shape4, "_shape_param_4")

    expected_shape0 = (batch, SEQ_LEN, MODEL_DIM)
    expected_shape1 = (batch, SEQ_LEN, HEADS, HEAD_DIM)
    expected_shape2 = (batch, HEADS, CHUNKS, CHUNKS, HEAD_DIM)
    expected_shape3 = (batch, HEADS, CHUNKS, CAT_COLS, HEAD_DIM)
    expected_shape4 = (batch * HEADS * CHUNKS, CAT_COLS, HEAD_DIM)
    if shape0_tuple != expected_shape0:
        raise ValueError(f"shape0 {shape0_tuple} != {expected_shape0}")
    if shape1_tuple != expected_shape1:
        raise ValueError(f"shape1 {shape1_tuple} != {expected_shape1}")
    if shape2_tuple != expected_shape2:
        raise ValueError(f"shape2 {shape2_tuple} != {expected_shape2}")
    if shape3_tuple != expected_shape3:
        raise ValueError(f"shape3 {shape3_tuple} != {expected_shape3}")
    if shape4_tuple != expected_shape4:
        raise ValueError(f"shape4 {shape4_tuple} != {expected_shape4}")

    return mm_16, shape4_tuple, batch


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    mm_16, shape0, shape1, shape2, shape3, shape4 = inputs
    view_default = torch.ops.aten.view.default(mm_16, shape0)
    view_default_1 = torch.ops.aten.view.default(view_default, shape1)
    permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3])
    view_default_2 = torch.ops.aten.view.default(permute_default, shape2)
    slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 2, -1, 9223372036854775807)
    slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 2, 0, -1)
    cat_default = torch.ops.aten.cat.default([slice_tensor, slice_tensor_1], 2)
    cat_default_1 = torch.ops.aten.cat.default([cat_default, view_default_2], 3)
    expand_default = torch.ops.aten.expand.default(cat_default_1, shape3)
    return torch.ops.aten.view.default(expand_default, shape4)


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
    mm_16, output_shape, batch = _validate_inputs(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if not mm_16.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        output_shape,
        (output_shape[1] * output_shape[2], output_shape[2], 1),
        device=mm_16.device,
        dtype=mm_16.dtype,
    )
    source_chunks = batch * HEADS * CHUNKS
    grid = lambda meta: (triton.cdiv(CHUNKS, meta["BLOCK_N"]), source_chunks)
    _rotated_cat_layout_kernel[grid](
        mm_16,
        output,
        SOURCE_CHUNKS=source_chunks,
        SEQ=SEQ_LEN,
        N_HEADS=HEADS,
        N_CHUNKS=CHUNKS,
        HEAD_SIZE=HEAD_DIM,
        HIDDEN=MODEL_DIM,
        OUT_COLS=CAT_COLS,
        BLOCK_D=HEAD_DIM,
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
