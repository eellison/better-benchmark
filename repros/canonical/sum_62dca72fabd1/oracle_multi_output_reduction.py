"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTa attention-output divide, head reorder clone, returned transpose view, and sibling hidden-dimension sum by writing the final contiguous [4096, 1536] backing storage and accumulating [1536] column-sum partials from the same tile stream, whereas Inductor currently schedules the div/view/permute/clone/view layout materialization and dim-0 sum as separate generic layout and reduction work over the materialized clone; Inductor cannot do this today because its scheduler does not preserve a layout-changing clone producer as a multi-output reduction source that can both materialize a returned transposed view and emit compatible column-reduction partials; the fix is SCHEDULER_FUSION: add a layout-materializing multi-output reduction schedule for fixed attention head reorders with sibling hidden-dimension reductions."""
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

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

BATCH = 8
HEADS = 24
QUERY = 64
KEY = 512
INPUT_BH = BATCH * HEADS
ROWS = BATCH * KEY
FEATURES = HEADS * QUERY

INPUT_SHAPE = (INPUT_BH, QUERY, KEY)
INPUT_STRIDE = (QUERY * KEY, KEY, 1)
VIEW0_SHAPE = (BATCH, HEADS, KEY, QUERY)
VIEW1_SHAPE = (BATCH, KEY, FEATURES)
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


if triton is not None:

    @triton.jit
    def _layout_div_reduce_partials_kernel(
        x_ptr,
        scale_ptr,
        clone_ptr,
        partials_ptr,
        x_stride_bh: tl.constexpr,
        x_stride_q: tl.constexpr,
        x_stride_k: tl.constexpr,
        HEADS_: tl.constexpr,
        QUERY_: tl.constexpr,
        KEY_: tl.constexpr,
        ROWS_: tl.constexpr,
        FEATURES_: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_FEATURES_: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        feature_block = tl.program_id(1)

        rows = row_block * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        features = feature_block * BLOCK_FEATURES_ + tl.arange(0, BLOCK_FEATURES_)

        batch = rows // KEY_
        key = rows - batch * KEY_
        head = features // QUERY_
        query = features - head * QUERY_
        batch_head = batch[:, None] * HEADS_ + head[None, :]

        mask = (rows[:, None] < ROWS_) & (features[None, :] < FEATURES_)
        input_offsets = (
            batch_head * x_stride_bh
            + query[None, :] * x_stride_q
            + key[:, None] * x_stride_k
        )
        clone_offsets = rows[:, None] * FEATURES_ + features[None, :]

        scale = tl.load(scale_ptr).to(tl.float32)
        values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        values = values / scale

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


def _validate_inputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    bmm_94, full_3, shape_param_0, shape_param_1, shape_param_2, shape_param_3 = inputs
    if not isinstance(bmm_94, torch.Tensor):
        raise TypeError(f"expected input 0 to be a tensor, got {type(bmm_94)!r}")
    if not isinstance(full_3, torch.Tensor):
        raise TypeError(f"expected input 1 to be a tensor, got {type(full_3)!r}")
    if bmm_94.device.type != "cuda" or full_3.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA tensor inputs")
    if bmm_94.dtype != torch.float32 or full_3.dtype != torch.float32:
        raise TypeError(f"expected float32 inputs, got {bmm_94.dtype} and {full_3.dtype}")
    if tuple(bmm_94.shape) != INPUT_SHAPE:
        raise ValueError(f"unexpected bmm_94 shape: {tuple(bmm_94.shape)}")
    if tuple(bmm_94.stride()) != INPUT_STRIDE or not bmm_94.is_contiguous():
        raise ValueError(f"unexpected bmm_94 stride: {tuple(bmm_94.stride())}")
    if tuple(full_3.shape) != ():
        raise ValueError(f"unexpected full_3 shape: {tuple(full_3.shape)}")
    if _shape_tuple(shape_param_0) != VIEW0_SHAPE:
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != VIEW1_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    if _shape_tuple(shape_param_2) != CLONE_SHAPE:
        raise ValueError(f"unexpected _shape_param_2: {shape_param_2!r}")
    if _shape_tuple(shape_param_3) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_3: {shape_param_3!r}")
    return bmm_94, full_3


def oracle_forward(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward computation with matching output layouts."""
    bmm_94, full_3 = _validate_inputs(tuple(inputs))

    clone_storage = torch.empty(CLONE_SHAPE, device=bmm_94.device, dtype=torch.float32)
    sums = torch.empty_strided(SUM_SHAPE, SUM_STRIDE, device=bmm_94.device, dtype=torch.float32)

    num_row_blocks = triton.cdiv(ROWS, BLOCK_ROWS)
    partials = torch.empty((num_row_blocks, FEATURES), device=bmm_94.device, dtype=torch.float32)

    grid = (num_row_blocks, triton.cdiv(FEATURES, BLOCK_FEATURES))
    _layout_div_reduce_partials_kernel[grid](
        bmm_94,
        full_3,
        clone_storage,
        partials,
        x_stride_bh=bmm_94.stride(0),
        x_stride_q=bmm_94.stride(1),
        x_stride_k=bmm_94.stride(2),
        HEADS_=HEADS,
        QUERY_=QUERY,
        KEY_=KEY,
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
    if tuple(transposed.shape) != TRANSPOSE_SHAPE or tuple(transposed.stride()) != TRANSPOSE_STRIDE:
        raise RuntimeError("oracle produced an unexpected transpose layout")
    return transposed, sums


def main() -> None:
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

    inputs = get_inputs()
    instance = get_repro_instance()

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
