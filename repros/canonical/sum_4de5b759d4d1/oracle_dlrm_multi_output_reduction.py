"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full DLRM multi-output graph by materializing the shared add/permute buffer for the returned slice views and fusing the block-0 add, mm slice, mask where, and first row-reduction stage into one Triton producer, whereas Inductor currently schedules the full graph as separate add, where, partial-reduction, and finalize kernels around the materialized view boundary; Inductor is already within the harness floor for this small multi-output graph once CUDAGraph timing removes dispatch overhead, so the oracle does not expose a meaningful remaining optimization gap; the fix is BANDWIDTH_BOUND: keep this as an at-floor full-scope artifact rather than a scheduler work item unless broader launch or multi-output view overhead work moves the baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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

ROWS = 2048
MM_COLS = 100
COLS = 64
BLOCKS = 9
ADD_COLS = BLOCKS * COLS
TAIL_BLOCKS = BLOCKS - 1
TAIL_ELEMENTS = ROWS * TAIL_BLOCKS * COLS
TAIL_BLOCK_N = 1024
WHERE_ROWS_PER_BLOCK = 32
WHERE_BLOCK_C = 16
NUM_ROW_BLOCKS = (ROWS + WHERE_ROWS_PER_BLOCK - 1) // WHERE_ROWS_PER_BLOCK
FINALIZE_BLOCK_M = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _materialize_tail_add_kernel(
        bmm_ptr,
        bmm_1_ptr,
        add_flat_ptr,
        bmm_s0: tl.constexpr,
        bmm_s1: tl.constexpr,
        bmm_s2: tl.constexpr,
        bmm_1_s0: tl.constexpr,
        bmm_1_s1: tl.constexpr,
        bmm_1_s2: tl.constexpr,
        COLS_: tl.constexpr,
        ADD_COLS_: tl.constexpr,
        TAIL_BLOCKS_: tl.constexpr,
        TAIL_ELEMENTS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TAIL_ELEMENTS_
        tail_cols = TAIL_BLOCKS_ * COLS_
        row = offsets // tail_cols
        tail_col = offsets - row * tail_cols
        block = tail_col // COLS_ + 1
        col = tail_col - (block - 1) * COLS_

        values = (
            tl.load(
                bmm_1_ptr + row * bmm_1_s0 + block * bmm_1_s1 + col * bmm_1_s2,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                bmm_ptr + row * bmm_s0 + col * bmm_s1 + block * bmm_s2,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
        )
        tl.store(add_flat_ptr + row * ADD_COLS_ + block * COLS_ + col, values, mask=mask)

    @triton.jit
    def _block0_where_partial_sum_kernel(
        mm_ptr,
        bmm_ptr,
        bmm_1_ptr,
        mask_ptr,
        full_ptr,
        add_flat_ptr,
        where_ptr,
        partial_ptr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        bmm_s0: tl.constexpr,
        bmm_s1: tl.constexpr,
        bmm_s2: tl.constexpr,
        bmm_1_s0: tl.constexpr,
        bmm_1_s1: tl.constexpr,
        bmm_1_s2: tl.constexpr,
        mask_s0: tl.constexpr,
        mask_s1: tl.constexpr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        ADD_COLS_: tl.constexpr,
        WHERE_ROWS_PER_BLOCK_: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row_block = tl.program_id(0)
        col_block = tl.program_id(1)
        row_offsets = row_block * WHERE_ROWS_PER_BLOCK_ + tl.arange(0, WHERE_ROWS_PER_BLOCK_)
        col_offsets = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
        rows = row_offsets[:, None]
        cols = col_offsets[None, :]
        active = (rows < ROWS_) & (cols < COLS_)
        full_value = tl.load(full_ptr).to(tl.float32)

        values = (
            tl.load(bmm_1_ptr + rows * bmm_1_s0 + cols * bmm_1_s2, mask=active, other=0.0).to(tl.float32)
            + tl.load(bmm_ptr + rows * bmm_s0 + cols * bmm_s1, mask=active, other=0.0).to(tl.float32)
        )
        tl.store(add_flat_ptr + rows * ADD_COLS_ + cols, values, mask=active)

        mm_values = tl.load(
            mm_ptr + rows * mm_s0 + cols * mm_s1,
            mask=active,
            other=0.0,
        ).to(tl.float32)
        pred = tl.load(
            mask_ptr + rows * mask_s0 + cols * mask_s1,
            mask=active,
            other=0,
        )
        where_values = tl.where(pred, full_value, mm_values + values)
        tl.store(where_ptr + rows * COLS_ + cols, where_values, mask=active)
        partial = tl.sum(tl.where(active, where_values, 0.0), axis=0)
        tl.store(partial_ptr + row_block * COLS_ + col_offsets, partial, mask=col_offsets < COLS_)

    @triton.jit
    def _finalize_sum_kernel(
        partial_ptr,
        sum_base_ptr,
        NUM_ROW_BLOCKS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        block_offsets = tl.arange(0, BLOCK_M)[:, None]
        col_offsets = tl.arange(0, BLOCK_C)
        values = tl.load(
            partial_ptr + block_offsets * COLS_ + col_offsets[None, :],
            mask=block_offsets < NUM_ROW_BLOCKS_,
            other=0.0,
        ).to(tl.float32)
        sums = tl.sum(values, axis=0)
        tl.store(sum_base_ptr + col_offsets, sums)


def _validate_inputs(inputs: tuple[object, ...]) -> None:
    if len(inputs) != 7:
        raise ValueError(f"expected 7 inputs, got {len(inputs)}")
    mm_5, bmm, bmm_1, arg53_1, full, shape_param_0, shape_param_1 = inputs
    expected_tensors = {
        "mm_5": (mm_5, (ROWS, MM_COLS), torch.float32),
        "bmm": (bmm, (ROWS, COLS, BLOCKS), torch.float32),
        "bmm_1": (bmm_1, (ROWS, BLOCKS, COLS), torch.float32),
        "arg53_1": (arg53_1, (ROWS, COLS), torch.bool),
        "full": (full, (), torch.float32),
    }
    for name, (tensor, shape, dtype) in expected_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    if list(shape_param_0) != [ROWS, ADD_COLS]:
        raise ValueError(f"unexpected view shape parameter 0: {shape_param_0}")
    if list(shape_param_1) != [COLS]:
        raise ValueError(f"unexpected view shape parameter 1: {shape_param_1}")


def oracle_forward(inputs):
    """Run the full-scope oracle and return the same view structure as Repro.forward."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(tuple(inputs))
    mm_5, bmm, bmm_1, arg53_1, full, _shape_param_0, _shape_param_1 = inputs

    add_base = torch.empty_strided(
        (ROWS, BLOCKS, COLS),
        (ADD_COLS, COLS, 1),
        device=mm_5.device,
        dtype=torch.float32,
    )
    add_flat = add_base.view(ROWS, ADD_COLS)
    where_base = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=mm_5.device,
        dtype=torch.float32,
    )
    partial = torch.empty(
        (NUM_ROW_BLOCKS, COLS),
        device=mm_5.device,
        dtype=torch.float32,
    )
    sum_base = torch.empty_strided(
        (1, COLS),
        (COLS, 1),
        device=mm_5.device,
        dtype=torch.float32,
    )

    _materialize_tail_add_kernel[(triton.cdiv(TAIL_ELEMENTS, TAIL_BLOCK_N),)](
        bmm,
        bmm_1,
        add_flat,
        bmm_s0=bmm.stride(0),
        bmm_s1=bmm.stride(1),
        bmm_s2=bmm.stride(2),
        bmm_1_s0=bmm_1.stride(0),
        bmm_1_s1=bmm_1.stride(1),
        bmm_1_s2=bmm_1.stride(2),
        COLS_=COLS,
        ADD_COLS_=ADD_COLS,
        TAIL_BLOCKS_=TAIL_BLOCKS,
        TAIL_ELEMENTS_=TAIL_ELEMENTS,
        BLOCK_N=TAIL_BLOCK_N,
        num_warps=4,
    )
    _block0_where_partial_sum_kernel[
        (NUM_ROW_BLOCKS, triton.cdiv(COLS, WHERE_BLOCK_C))
    ](
        mm_5,
        bmm,
        bmm_1,
        arg53_1,
        full,
        add_flat,
        where_base,
        partial,
        mm_s0=mm_5.stride(0),
        mm_s1=mm_5.stride(1),
        bmm_s0=bmm.stride(0),
        bmm_s1=bmm.stride(1),
        bmm_s2=bmm.stride(2),
        bmm_1_s0=bmm_1.stride(0),
        bmm_1_s1=bmm_1.stride(1),
        bmm_1_s2=bmm_1.stride(2),
        mask_s0=arg53_1.stride(0),
        mask_s1=arg53_1.stride(1),
        ROWS_=ROWS,
        COLS_=COLS,
        ADD_COLS_=ADD_COLS,
        WHERE_ROWS_PER_BLOCK_=WHERE_ROWS_PER_BLOCK,
        BLOCK_C=WHERE_BLOCK_C,
        num_warps=4,
    )
    _finalize_sum_kernel[(1,)](
        partial,
        sum_base,
        NUM_ROW_BLOCKS_=NUM_ROW_BLOCKS,
        COLS_=COLS,
        BLOCK_M=FINALIZE_BLOCK_M,
        BLOCK_C=COLS,
        num_warps=4,
    )

    return (
        add_flat[:, 64:128],
        add_flat[:, 128:192],
        add_flat[:, 192:256],
        add_flat[:, 256:320],
        add_flat[:, 320:384],
        add_flat[:, 384:448],
        add_flat[:, 448:512],
        add_flat[:, 512:576],
        where_base.permute(1, 0),
        sum_base.view(COLS),
    )


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
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
