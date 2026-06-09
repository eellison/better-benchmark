"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Reformer head-layout materialization as a structured 64-wide block transpose from `[B*4096, 768]` token-major input into the final contiguous `[B*768, 64, 64]` output, whereas Inductor lowers the same view/permute/view/expand/clone/view scope as a generic flat pointwise clone with per-element div/mod address reconstruction; Inductor cannot do this today because its scheduler/codegen does not recognize this fixed-head Reformer layout copy as a block-structured transpose with one address calculation per 64-value vector; the fix is NEW_PATTERN: add a specialized layout-copy pattern or scheduler lowering for this Reformer token/head block transpose."""
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
    get_shape_key,
    has_stochastic_ops,
)


SEQ_LEN = 4096
HEADS = 12
HEAD_DIM = 64
HIDDEN = HEADS * HEAD_DIM


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable shape, got {type(value)!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int],
    int,
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_14, shape0, shape1, shape2, shape3, shape4 = inputs
    if not isinstance(mm_14, torch.Tensor):
        raise TypeError(f"expected mm_14 tensor, got {type(mm_14)!r}")
    if mm_14.ndim != 2 or int(mm_14.shape[1]) != HIDDEN:
        raise ValueError(f"unexpected mm_14 shape: {tuple(mm_14.shape)}")
    if int(mm_14.shape[0]) % SEQ_LEN != 0:
        raise ValueError(f"mm_14 rows must be a multiple of {SEQ_LEN}, got {mm_14.shape[0]}")
    if mm_14.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"expected fp16/fp32 mm_14, got {mm_14.dtype}")
    if mm_14.stride() != (HIDDEN, 1):
        raise ValueError(f"mm_14 must be contiguous with stride {(HIDDEN, 1)}, got {mm_14.stride()}")

    batch = int(mm_14.shape[0]) // SEQ_LEN
    shape0_tuple = _shape_tuple(shape0, "_shape_param_0")
    shape1_tuple = _shape_tuple(shape1, "_shape_param_1")
    shape2_tuple = _shape_tuple(shape2, "_shape_param_2")
    shape3_tuple = _shape_tuple(shape3, "_shape_param_3")
    shape4_tuple = _shape_tuple(shape4, "_shape_param_4")

    expected_shape0 = (batch, SEQ_LEN, HIDDEN)
    expected_shape1 = (batch, SEQ_LEN, HEADS, HEAD_DIM)
    expected_shape2 = (batch, HEADS, 64, 64, HEAD_DIM)
    expected_shape4 = (batch * HEADS * HEAD_DIM, 64, 64)
    if shape0_tuple != expected_shape0:
        raise ValueError(f"shape0 {shape0_tuple} != {expected_shape0}")
    if shape1_tuple != expected_shape1:
        raise ValueError(f"shape1 {shape1_tuple} != {expected_shape1}")
    if shape2_tuple != expected_shape2:
        raise ValueError(f"shape2 {shape2_tuple} != {expected_shape2}")
    if shape3_tuple != expected_shape2:
        raise ValueError(f"shape3 {shape3_tuple} != {expected_shape2}")
    if shape4_tuple != expected_shape4:
        raise ValueError(f"shape4 {shape4_tuple} != {expected_shape4}")

    return (
        mm_14,
        shape0_tuple,
        shape1_tuple,
        shape2_tuple,
        shape3_tuple,
        shape4_tuple,
        batch,
    )


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_M": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_M": 64}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL_VECTORS"],
    )
    @triton.jit
    def _reformer_head_layout_kernel(
        input_ptr,
        output_ptr,
        TOTAL_VECTORS: tl.constexpr,
        SEQ: tl.constexpr,
        N_HEADS: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        MODEL_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        vector_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        vector_mask = vector_offsets < TOTAL_VECTORS

        seq = vector_offsets % SEQ
        tmp = vector_offsets // SEQ
        head = tmp % N_HEADS
        batch = tmp // N_HEADS

        input_offsets = (
            batch[:, None] * SEQ * MODEL_DIM
            + seq[:, None] * MODEL_DIM
            + head[:, None] * HEAD_SIZE
            + dim_offsets[None, :]
        )
        output_offsets = vector_offsets[:, None] * HEAD_SIZE + dim_offsets[None, :]
        mask = vector_mask[:, None] & (dim_offsets[None, :] < HEAD_SIZE)
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        tl.store(output_ptr + output_offsets, values, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    mm_14, shape0, shape1, shape2, shape3, shape4, _batch = _validate_inputs(inputs)
    view_default = torch.ops.aten.view.default(mm_14, shape0)
    view_default_1 = torch.ops.aten.view.default(view_default, shape1)
    permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3])
    view_default_2 = torch.ops.aten.view.default(permute_default, shape2)
    expand_default = torch.ops.aten.expand.default(view_default_2, shape3)
    clone_default = torch.ops.aten.clone.default(expand_default, memory_format=torch.contiguous_format)
    return torch.ops.aten.view.default(clone_default, shape4)


@oracle_impl(hardware="H100", shapes="(T([4096, 768], f16), S([1, 4096, 768]), S([1, 4096, 12, 64]), S([1, 12, 64, 64, 64]), S([1, 12, 64, 64, 64]), S([768, 64, 64]))")
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
    mm_14, _shape0, _shape1, _shape2, _shape3, shape4, batch = _validate_inputs(inputs)
    if triton is None or not mm_14.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape4,
        (shape4[1] * shape4[2], shape4[2], 1),
        device=mm_14.device,
        dtype=mm_14.dtype,
    )
    total_vectors = batch * HEADS * SEQ_LEN
    grid = lambda meta: (triton.cdiv(total_vectors, meta["BLOCK_M"]),)
    _reformer_head_layout_kernel[grid](
        mm_14,
        output,
        TOTAL_VECTORS=total_vectors,
        SEQ=SEQ_LEN,
        N_HEADS=HEADS,
        HEAD_SIZE=HEAD_DIM,
        MODEL_DIM=HIDDEN,
        BLOCK_D=HEAD_DIM,
    )
    return output


def _check_layout(output: torch.Tensor, shape4: tuple[int, int, int]) -> bool:
    return (
        tuple(output.shape) == shape4
        and output.stride() == (shape4[1] * shape4[2], shape4[2], 1)
        and output.storage_offset() == 0
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
        _mm_14, _shape0, _shape1, _shape2, _shape3, shape4, _batch = _validate_inputs(inputs)
        with torch.no_grad():
            layout_output = oracle_forward(inputs)
            if layout_output.is_cuda:
                torch.cuda.synchronize()
        layout_ok = _check_layout(layout_output, shape4)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_output.shape)} stride={layout_output.stride()})"
        )
        ok = ok and layout_ok
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
