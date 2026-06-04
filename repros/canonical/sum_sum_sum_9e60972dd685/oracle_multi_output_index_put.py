"""
Oracle for sum_sum_sum_9e60972dd685

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full GPT-J layer-norm-backward-shaped multi-output graph, including the four `[128,4096]` addends, both row reductions, both returned `[4096]` column reductions, and the dense `[50400,4096]` `index_put(accumulate=True)` output, whereas Inductor lowers the arithmetic and indexed scatter materialization through generic scheduled kernels; Inductor cannot materially beat this full scope today because the returned `index_put` tensor is a real dense contiguous output whose zero-fill/write dominates the runtime, so even a scatter-reduce lowering for the row-local producer cannot remove the required 826 MB materialization; the classification is BANDWIDTH_BOUND: treat the apparent SOL gap as missing dense-output byte accounting rather than an actionable scheduler-fusion, scatter-reduce, split-K, algebraic-elimination, recompute-fusion, or new-pattern gap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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

ROWS = 128
HIDDEN = 4096
VOCAB = 50400
BLOCK_H = 1024
H_CHUNKS = HIDDEN // BLOCK_H
BLOCK_ZERO = 2048


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_and_row_partials_kernel(
        mm328_ptr,
        mm333_ptr,
        mm335_ptr,
        mm337_ptr,
        arg1_ptr,
        arg199_ptr,
        arg201_ptr,
        arg202_ptr,
        out_index_put_ptr,
        out_sum2_ptr,
        out_sum3_ptr,
        row_sum_partials_ptr,
        weighted_sum_partials_ptr,
        TOTAL_INDEX_PUT: tl.constexpr,
        ROWS_: tl.constexpr,
        BLOCK_H_: tl.constexpr,
        BLOCK_ZERO_: tl.constexpr,
        H_CHUNKS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
    ):
        pid = tl.program_id(0)

        zero_offsets = pid * BLOCK_ZERO_ + tl.arange(0, BLOCK_ZERO_)
        tl.store(
            out_index_put_ptr + zero_offsets,
            tl.zeros((BLOCK_ZERO_,), tl.float32),
            mask=zero_offsets < TOTAL_INDEX_PUT,
        )

        row_partial_programs: tl.constexpr = ROWS_ * H_CHUNKS_
        if pid < row_partial_programs:
            row = pid // H_CHUNKS_
            chunk = pid - row * H_CHUNKS_
            cols = chunk * BLOCK_H_ + tl.arange(0, BLOCK_H_)
            mask = cols < HIDDEN_
            base = row * HIDDEN_ + cols

            mm_sum = (
                tl.load(mm328_ptr + base, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm333_ptr + base, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm335_ptr + base, mask=mask, other=0.0).to(tl.float32)
                + tl.load(mm337_ptr + base, mask=mask, other=0.0).to(tl.float32)
            )
            weight = tl.load(arg1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            centered = (
                tl.load(arg199_ptr + base, mask=mask, other=0.0).to(tl.float32)
                - tl.load(arg201_ptr + row).to(tl.float32)
            )
            scale = tl.load(arg202_ptr + row).to(tl.float32)
            mul2 = centered * scale
            mul = mm_sum * weight

            partial = tl.sum(mul, axis=0)
            weighted_partial = tl.sum(mul * mul2, axis=0)
            partial_offset = row * H_CHUNKS_ + chunk
            tl.store(row_sum_partials_ptr + partial_offset, partial)
            tl.store(weighted_sum_partials_ptr + partial_offset, weighted_partial)

            if row == 0:
                zeros = tl.zeros((BLOCK_H_,), tl.float32)
                tl.store(out_sum2_ptr + cols, zeros, mask=mask)
                tl.store(out_sum3_ptr + cols, zeros, mask=mask)

    @triton.jit
    def _scatter_and_column_reduce_kernel(
        mm328_ptr,
        mm333_ptr,
        mm335_ptr,
        mm337_ptr,
        arg1_ptr,
        arg199_ptr,
        arg201_ptr,
        arg202_ptr,
        add378_ptr,
        index_ptr,
        full_scalar_ptr,
        out_sum2_ptr,
        out_sum3_ptr,
        out_index_put_ptr,
        row_sum_partials_ptr,
        weighted_sum_partials_ptr,
        BLOCK_H_: tl.constexpr,
        H_CHUNKS_: tl.constexpr,
        HIDDEN_: tl.constexpr,
        VOCAB_: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk = tl.program_id(1)
        cols = chunk * BLOCK_H_ + tl.arange(0, BLOCK_H_)
        mask = cols < HIDDEN_
        base = row * HIDDEN_ + cols

        mm_sum = (
            tl.load(mm328_ptr + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm333_ptr + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm335_ptr + base, mask=mask, other=0.0).to(tl.float32)
            + tl.load(mm337_ptr + base, mask=mask, other=0.0).to(tl.float32)
        )
        weight = tl.load(arg1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        centered = (
            tl.load(arg199_ptr + base, mask=mask, other=0.0).to(tl.float32)
            - tl.load(arg201_ptr + row).to(tl.float32)
        )
        scale = tl.load(arg202_ptr + row).to(tl.float32)
        mul2 = centered * scale

        partial_cols = tl.arange(0, H_CHUNKS_)
        partial_base = row * H_CHUNKS_ + partial_cols
        row_sum = tl.sum(tl.load(row_sum_partials_ptr + partial_base), axis=0)
        weighted_sum = tl.sum(
            tl.load(weighted_sum_partials_ptr + partial_base),
            axis=0,
        )

        tl.atomic_add(out_sum2_ptr + cols, mm_sum * mul2, sem="relaxed", mask=mask)
        tl.atomic_add(out_sum3_ptr + cols, mm_sum, sem="relaxed", mask=mask)

        ln_backward = (scale / HIDDEN_) * (
            (mm_sum * weight * HIDDEN_) - row_sum - mul2 * weighted_sum
        )
        update = tl.load(add378_ptr + base, mask=mask, other=0.0).to(tl.float32) + ln_backward

        raw_index = tl.load(index_ptr + row).to(tl.int64)
        normalized_index = tl.where(raw_index < 0, raw_index + VOCAB_, raw_index)
        full_scalar = tl.load(full_scalar_ptr).to(tl.float32)
        scatter_value = tl.where(raw_index == -1, full_scalar, update)
        in_bounds = (normalized_index >= 0) & (normalized_index < VOCAB_)
        tl.atomic_add(
            out_index_put_ptr + normalized_index * HIDDEN_ + cols,
            scatter_value,
            sem="relaxed",
            mask=mask & in_bounds,
        )


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 15:
        raise ValueError(f"expected 15 inputs, got {len(inputs)}")

    tensor_specs = {
        "mm_328": (inputs[0], (ROWS, HIDDEN), torch.float32),
        "mm_333": (inputs[1], (ROWS, HIDDEN), torch.float32),
        "mm_335": (inputs[2], (ROWS, HIDDEN), torch.float32),
        "mm_337": (inputs[3], (ROWS, HIDDEN), torch.float32),
        "arg1_1": (inputs[4], (HIDDEN,), torch.float32),
        "arg199_1": (inputs[5], (1, ROWS, HIDDEN), torch.float32),
        "arg201_1": (inputs[6], (1, ROWS, 1), torch.float32),
        "arg202_1": (inputs[7], (1, ROWS, 1), torch.float32),
        "add_378": (inputs[8], (1, ROWS, HIDDEN), torch.float32),
        "arg0_1": (inputs[9], (1, ROWS), torch.int64),
        "full_1": (inputs[10], (), torch.float32),
    }
    for name, (tensor, shape, dtype) in tensor_specs.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for this oracle")

    for i in range(11, 15):
        if list(inputs[i]) != [1, ROWS, HIDDEN]:
            raise ValueError(f"unexpected view shape parameter {i}: {inputs[i]}")


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    (
        mm_328,
        mm_333,
        mm_335,
        mm_337,
        arg1_1,
        arg199_1,
        arg201_1,
        arg202_1,
        add_378,
        arg0_1,
        full_1,
        *_shape_params,
    ) = inputs

    out_sum2 = torch.empty_strided((HIDDEN,), (1,), device=mm_328.device, dtype=torch.float32)
    out_sum3 = torch.empty_strided((HIDDEN,), (1,), device=mm_328.device, dtype=torch.float32)
    out_index_put = torch.empty_strided(
        (VOCAB, HIDDEN),
        (HIDDEN, 1),
        device=mm_328.device,
        dtype=torch.float32,
    )
    row_sum_partials = torch.empty_strided(
        (ROWS, H_CHUNKS),
        (H_CHUNKS, 1),
        device=mm_328.device,
        dtype=torch.float32,
    )
    weighted_sum_partials = torch.empty_strided(
        (ROWS, H_CHUNKS),
        (H_CHUNKS, 1),
        device=mm_328.device,
        dtype=torch.float32,
    )

    total_index_put = VOCAB * HIDDEN
    zero_blocks = triton.cdiv(total_index_put, BLOCK_ZERO)
    partial_blocks = ROWS * H_CHUNKS
    _zero_and_row_partials_kernel[(max(zero_blocks, partial_blocks),)](
        mm_328,
        mm_333,
        mm_335,
        mm_337,
        arg1_1,
        arg199_1,
        arg201_1,
        arg202_1,
        out_index_put,
        out_sum2,
        out_sum3,
        row_sum_partials,
        weighted_sum_partials,
        TOTAL_INDEX_PUT=total_index_put,
        ROWS_=ROWS,
        BLOCK_H_=BLOCK_H,
        BLOCK_ZERO_=BLOCK_ZERO,
        H_CHUNKS_=H_CHUNKS,
        HIDDEN_=HIDDEN,
        num_warps=8,
    )
    _scatter_and_column_reduce_kernel[(ROWS, H_CHUNKS)](
        mm_328,
        mm_333,
        mm_335,
        mm_337,
        arg1_1,
        arg199_1,
        arg201_1,
        arg202_1,
        add_378,
        arg0_1,
        full_1,
        out_sum2,
        out_sum3,
        out_index_put,
        row_sum_partials,
        weighted_sum_partials,
        BLOCK_H_=BLOCK_H,
        H_CHUNKS_=H_CHUNKS,
        HIDDEN_=HIDDEN,
        VOCAB_=VOCAB,
        num_warps=8,
    )
    return out_sum2, out_sum3, out_index_put


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
