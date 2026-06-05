"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete DeiT select-scatter layer-norm-backward return tuple by treating the zero-filled `select_scatter` as a structured single-token gather/mask producer, deriving the token-0 row reductions directly from `mm`, writing the required transposed side output, and accumulating all three returned channel reductions from the same producer, whereas Inductor currently materializes the dense `[128,197,192]` zero/select-scatter tensor and schedules the row reductions, sibling channel reductions, and permute side output as generic dense work; Inductor cannot do this today because scheduler/codegen does not model zero-fill `select_scatter` as a structured scatter-reduce producer feeding both row-wise epilogues and sibling reductions; the fix is SCATTER_REDUCE: add a structured select-scatter lowering that maps sparse token sources directly into row-reduction epilogues, emits required materialized scatter stores, and accumulates compatible channel reductions without materializing the dense scatter input."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 128
TOKENS = 197
CHANNELS = 192
ROWS = BATCH * TOKENS
SIDE_NUMEL = ROWS * CHANNELS
OUTPUT_NUMEL = 3 * CHANNELS
BLOCK_ZERO = 1024
BLOCK_C = 256

EXPECTED_SHAPE_PARAMS = (
    [ROWS, CHANNELS],
    [CHANNELS],
)


def _check_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"expected 6 repro inputs, got {len(inputs)}")
    mm, primals_150, mul_84, div, shape0, shape1 = inputs
    if mm.shape != (BATCH, CHANNELS) or mm.dtype != torch.float32:
        raise ValueError(f"unexpected mm: shape={tuple(mm.shape)} dtype={mm.dtype}")
    if primals_150.shape != (CHANNELS,) or primals_150.dtype != torch.float32:
        raise ValueError(
            f"unexpected primals_150: shape={tuple(primals_150.shape)} "
            f"dtype={primals_150.dtype}"
        )
    if mul_84.shape != (BATCH, TOKENS, CHANNELS) or mul_84.dtype != torch.float32:
        raise ValueError(
            f"unexpected mul_84: shape={tuple(mul_84.shape)} dtype={mul_84.dtype}"
        )
    if div.shape != (BATCH, TOKENS, 1) or div.dtype != torch.float32:
        raise ValueError(f"unexpected div: shape={tuple(div.shape)} dtype={div.dtype}")
    for idx, (actual, expected) in enumerate(
        zip((shape0, shape1), EXPECTED_SHAPE_PARAMS),
        start=4,
    ):
        if list(actual) != expected:
            raise ValueError(f"unexpected shape param {idx}: {actual} != {expected}")


if triton is not None:

    @triton.jit
    def _zero_buffers_kernel(
        side_ptr,
        sum_with_mul84_ptr,
        sum_scatter_ptr,
        side_reduction_ptr,
        total: tl.constexpr,
        side_numel: tl.constexpr,
        channels: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        zero = tl.zeros((BLOCK,), tl.float32)

        side_mask = offsets < side_numel
        tl.store(side_ptr + offsets, zero, mask=side_mask)

        tail = offsets - side_numel
        out0_mask = (tail >= 0) & (tail < channels)
        out1_mask = (tail >= channels) & (tail < 2 * channels)
        out3_mask = (tail >= 2 * channels) & (tail < 3 * channels)
        out0_offsets = tl.where(out0_mask, tail, 0)
        out1_offsets = tl.where(out1_mask, tail - channels, 0)
        out3_offsets = tl.where(out3_mask, tail - 2 * channels, 0)
        tl.store(sum_with_mul84_ptr + out0_offsets, zero, mask=out0_mask)
        tl.store(sum_scatter_ptr + out1_offsets, zero, mask=out1_mask)
        tl.store(side_reduction_ptr + out3_offsets, zero, mask=out3_mask)

    @triton.jit
    def _token0_scatter_reduce_kernel(
        mm_ptr,
        primals_150_ptr,
        mul_84_ptr,
        div_ptr,
        side_ptr,
        sum_with_mul84_ptr,
        sum_scatter_ptr,
        side_reduction_ptr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        batch = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < channels

        mm = tl.load(mm_ptr + batch * channels + cols, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(primals_150_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mul_84 = tl.load(
            mul_84_ptr + batch * tokens * channels + cols,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        div = tl.load(div_ptr + batch * tokens).to(tl.float32)

        weighted = mm * weight
        row_sum = tl.sum(weighted, axis=0)
        row_dot = tl.sum(weighted * mul_84, axis=0)
        side_token = div * (weighted * channels - row_sum - mul_84 * row_dot)

        token0_offsets = batch * tokens * channels + cols
        tl.store(side_ptr + token0_offsets, side_token, mask=mask)

        tl.atomic_add(sum_with_mul84_ptr + cols, mm * mul_84, sem="relaxed", mask=mask)
        tl.atomic_add(sum_scatter_ptr + cols, mm, sem="relaxed", mask=mask)
        tl.atomic_add(side_reduction_ptr + cols, side_token, sem="relaxed", mask=mask)


def oracle_forward(inputs):
    """Run the full Repro.forward computation for the standard oracle harness."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    _check_inputs(inputs)
    mm, primals_150, mul_84, div, _shape_param_0, _shape_param_1 = inputs
    if mm.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")

    sum_with_mul84 = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)
    sum_scatter = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)
    side_base = torch.empty((ROWS, CHANNELS), device=mm.device, dtype=torch.float32)
    side_reduction = torch.empty((CHANNELS,), device=mm.device, dtype=torch.float32)

    total = SIDE_NUMEL + OUTPUT_NUMEL
    _zero_buffers_kernel[(triton.cdiv(total, BLOCK_ZERO),)](
        side_base,
        sum_with_mul84,
        sum_scatter,
        side_reduction,
        total,
        SIDE_NUMEL,
        CHANNELS,
        BLOCK=BLOCK_ZERO,
        num_warps=4,
    )
    _token0_scatter_reduce_kernel[(BATCH,)](
        mm,
        primals_150,
        mul_84,
        div,
        side_base,
        sum_with_mul84,
        sum_scatter,
        side_reduction,
        CHANNELS,
        TOKENS,
        BLOCK=BLOCK_C,
        num_warps=8,
    )

    return (sum_with_mul84, sum_scatter, side_base.t(), side_reduction)


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
