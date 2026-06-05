"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DeBERTa reshape/permute/clone/divide scope by materializing the required contiguous `[4096,1536]` base buffer and accumulating the sibling `[1536]` column sum from the same scaled layout pass, whereas Inductor schedules the required layout-producing clone and the reduction as generic work around the materialized buffer; Inductor cannot do this today because its scheduler does not fuse a required materialized layout side output with a downstream reduction consumer that reads the same logical tensor through view/permute structure; the fix is SCHEDULER_FUSION: teach reduction scheduling to emit a full-scope layout-copy plus partial-reduction epilogue when the cloned layout is returned and also reduced inside the captured graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

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

P_BLOCK = 24
H = 64
N = 512
B = 8
ROWS = B * N
C = P_BLOCK * H
SCALE = 0.125
ROW_BLOCK = 64
COL_BLOCK = 64
NUM_ROW_BLOCKS = triton.cdiv(ROWS, ROW_BLOCK)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


@triton.jit
def _layout_and_partial_sum_kernel(
    x_ptr,
    out_base_ptr,
    partial_ptr,
    ROWS_: tl.constexpr,
    C_: tl.constexpr,
    ROW_BLOCK_: tl.constexpr,
    COL_BLOCK_: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * ROW_BLOCK_ + tl.arange(0, ROW_BLOCK_)
    cols = col_block * COL_BLOCK_ + tl.arange(0, COL_BLOCK_)

    row_mask = rows < ROWS_
    col_mask = cols < C_
    mask = row_mask[:, None] & col_mask[None, :]

    batch = rows // 512
    n = rows - batch * 512
    p = cols // 64
    h = cols - p * 64

    input_offsets = (
        (batch[:, None] * 24 + p[None, :]) * 32768
        + h[None, :] * 512
        + n[:, None]
    )
    out_offsets = rows[:, None] * C_ + cols[None, :]

    values = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32) * 0.125
    tl.store(out_base_ptr + out_offsets, values, mask=mask)

    partial = tl.sum(tl.where(mask, values, 0.0), axis=0)
    tl.store(partial_ptr + row_block * C_ + cols, partial, mask=col_mask)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    out_sum_ptr,
    NUM_ROW_BLOCKS_: tl.constexpr,
    C_: tl.constexpr,
    ROW_BLOCKS_TILE_: tl.constexpr,
    COL_BLOCK_: tl.constexpr,
):
    cols = tl.program_id(0) * COL_BLOCK_ + tl.arange(0, COL_BLOCK_)
    row_blocks = tl.arange(0, ROW_BLOCKS_TILE_)
    mask = (row_blocks[:, None] < NUM_ROW_BLOCKS_) & (cols[None, :] < C_)
    offsets = row_blocks[:, None] * C_ + cols[None, :]
    partials = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_sum_ptr + cols, tl.sum(partials, axis=0), mask=cols < C_)


def _expect_shape_param(actual: object, expected: list[int]) -> None:
    if list(actual) != expected:
        raise ValueError(f"unexpected shape parameter {actual}, expected {expected}")


def _check_output_layout(outputs: tuple[torch.Tensor, torch.Tensor]) -> bool:
    out_t, out_sum = outputs
    return (
        tuple(out_t.shape) == (C, ROWS)
        and tuple(out_t.stride()) == (1, C)
        and out_t.storage_offset() == 0
        and tuple(out_sum.shape) == (C,)
        and tuple(out_sum.stride()) == (1,)
        and out_sum.storage_offset() == 0
    )


def oracle_forward(inputs):
    """Run the full-scope oracle computation."""
    bmm_2, shape0, shape1, shape2, shape3 = inputs

    if bmm_2.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if bmm_2.dtype != torch.float32:
        raise ValueError(f"expected float32 input, got {bmm_2.dtype}")
    if tuple(bmm_2.shape) != (B * P_BLOCK, H, N):
        raise ValueError(f"unexpected input shape {tuple(bmm_2.shape)}")
    if not bmm_2.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={tuple(bmm_2.stride())}")

    _expect_shape_param(shape0, [B, P_BLOCK, N, H])
    _expect_shape_param(shape1, [B, N, C])
    _expect_shape_param(shape2, [ROWS, C])
    _expect_shape_param(shape3, [C])

    out_base = torch.empty((ROWS, C), device=bmm_2.device, dtype=torch.float32)
    partial = torch.empty((NUM_ROW_BLOCKS, C), device=bmm_2.device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=bmm_2.device, dtype=torch.float32)

    grid = (NUM_ROW_BLOCKS, triton.cdiv(C, COL_BLOCK))
    _layout_and_partial_sum_kernel[grid](
        bmm_2,
        out_base,
        partial,
        ROWS_=ROWS,
        C_=C,
        ROW_BLOCK_=ROW_BLOCK,
        COL_BLOCK_=COL_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _finalize_sum_kernel[(triton.cdiv(C, COL_BLOCK),)](
        partial,
        out_sum,
        NUM_ROW_BLOCKS_=NUM_ROW_BLOCKS,
        C_=C,
        ROW_BLOCKS_TILE_=NUM_ROW_BLOCKS,
        COL_BLOCK_=COL_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return out_base.t(), out_sum


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
        with torch.no_grad():
            outputs = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_output_layout(outputs)
        print(f"  output layout: {'PASS' if layout_ok else 'FAIL'}")
        ok = ok and layout_ok
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
