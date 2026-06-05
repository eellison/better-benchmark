"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the BERT attention-output head reorder clone with the sibling hidden-dimension sum by writing the final contiguous [16384,768] backing storage and accumulating the [768] sum from the same tile stream, whereas Inductor currently schedules the view/permute/clone/view layout materialization and dim-0 sum as separate generic layout and reduction work over the materialized clone; Inductor cannot do this today because its scheduler does not preserve a layout-changing clone producer as a multi-output reduction source that can both materialize a returned transposed view and emit compatible column-reduction partials; the fix is SCHEDULER_FUSION: add a layout-materializing multi-output reduction schedule for fixed attention head reorders with sibling hidden-dimension reductions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


BATCH = 32
HEADS = 12
SEQ = 512
HEAD_DIM = 64
INPUT_BH = BATCH * HEADS
ROWS = BATCH * SEQ
FEATURES = HEADS * HEAD_DIM

INPUT_SHAPE = (INPUT_BH, SEQ, HEAD_DIM)
VIEW0_SHAPE = (BATCH, HEADS, SEQ, HEAD_DIM)
VIEW1_SHAPE = (BATCH, SEQ, FEATURES)
CLONE_SHAPE = (ROWS, FEATURES)
TRANSPOSE_SHAPE = (FEATURES, ROWS)
TRANSPOSE_STRIDE = (1, FEATURES)
SUM_SHAPE = (FEATURES,)
SUM_STRIDE = (1,)

BLOCK_ROWS = 64
BLOCK_FEATURES = 32


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _layout_copy_reduce_partials_kernel(
        x_ptr,
        clone_ptr,
        partials_ptr,
        x_stride_bh: tl.constexpr,
        x_stride_s: tl.constexpr,
        x_stride_d: tl.constexpr,
        HEADS_: tl.constexpr,
        HEAD_DIM_: tl.constexpr,
        SEQ_: tl.constexpr,
        ROWS_: tl.constexpr,
        FEATURES_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_FEATURES_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        feature_block = tl.program_id(1)

        rows = row_block * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        features = feature_block * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)

        batch = rows // SEQ_
        seq = rows - batch * SEQ_
        head = features // HEAD_DIM_
        dim = features - head * HEAD_DIM_
        bh = batch[:, None] * HEADS_ + head[None, :]

        mask = (rows[:, None] < ROWS_) & (features[None, :] < FEATURES_)
        input_offsets = (
            bh * x_stride_bh
            + seq[:, None] * x_stride_s
            + dim[None, :] * x_stride_d
        )
        clone_offsets = rows[:, None] * FEATURES_ + features[None, :]

        values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(clone_ptr + clone_offsets, values, mask=mask)

        partials = tl.sum(tl.where(mask, values, 0.0), axis=0)
        tl.store(
            partials_ptr + row_block * FEATURES_ + features,
            partials,
            mask=features < FEATURES_,
        )

    @triton.jit
    def _finish_partials_kernel(
        partials_ptr,
        sum_ptr,
        NUM_ROW_BLOCKS: tl.constexpr,
        FEATURES_: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        feature = tl.program_id(0)
        row_blocks = tl.arange(0, BLOCK_R)
        mask = row_blocks < NUM_ROW_BLOCKS
        values = tl.load(
            partials_ptr + row_blocks * FEATURES_ + feature,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        tl.store(sum_ptr + feature, tl.sum(values, axis=0))


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...]) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    bmm_44, shape_param_0, shape_param_1, shape_param_2, shape_param_3 = inputs
    if not isinstance(bmm_44, torch.Tensor):
        raise TypeError(f"expected input 0 to be a tensor, got {type(bmm_44)!r}")
    if bmm_44.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if bmm_44.dtype != torch.float32:
        raise TypeError(f"expected float32 input, got {bmm_44.dtype}")
    if tuple(bmm_44.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected bmm_44 shape: {tuple(bmm_44.shape)}")
    if tuple(bmm_44.stride()) != (SEQ * HEAD_DIM, HEAD_DIM, 1) or not bmm_44.is_contiguous():
        raise ValueError(f"unexpected bmm_44 stride: {tuple(bmm_44.stride())}")
    if _shape_tuple(shape_param_0) != VIEW0_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != VIEW1_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != CLONE_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    if _shape_tuple(shape_param_3) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_3: {shape_param_3!r}")
    return bmm_44


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    bmm_44 = _validate_inputs(tuple(inputs))
    clone_storage = torch.empty(CLONE_SHAPE, device=bmm_44.device, dtype=bmm_44.dtype)
    sums = torch.empty_strided(SUM_SHAPE, SUM_STRIDE, device=bmm_44.device, dtype=bmm_44.dtype)

    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    partials = torch.empty(
        (num_row_blocks, FEATURES),
        device=bmm_44.device,
        dtype=torch.float32,
    )

    grid = (num_row_blocks, triton.cdiv(FEATURES, BLOCK_FEATURES))
    _layout_copy_reduce_partials_kernel[grid](
        bmm_44,
        clone_storage,
        partials,
        x_stride_bh=bmm_44.stride(0),
        x_stride_s=bmm_44.stride(1),
        x_stride_d=bmm_44.stride(2),
        HEADS_=HEADS,
        HEAD_DIM_=HEAD_DIM,
        SEQ_=SEQ,
        ROWS_=ROWS,
        FEATURES_=FEATURES,
        BLOCK_ROWS_=BLOCK_ROWS,
        BLOCK_FEATURES_=BLOCK_FEATURES,
        num_warps=8,
        num_stages=4,
    )
    _finish_partials_kernel[(FEATURES,)](
        partials,
        sums,
        NUM_ROW_BLOCKS=num_row_blocks,
        FEATURES_=FEATURES,
        BLOCK_R=triton.next_power_of_2(num_row_blocks),
        num_warps=8,
        num_stages=1,
    )

    transposed = torch.as_strided(clone_storage, TRANSPOSE_SHAPE, TRANSPOSE_STRIDE)
    if tuple(transposed.shape) != TRANSPOSE_SHAPE or transposed.stride() != TRANSPOSE_STRIDE:
        raise RuntimeError("oracle produced an unexpected transpose layout")
    return transposed, sums


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
