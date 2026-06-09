"""
Oracle for amax_sum_sum_1458251679be

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full
GPT-2 sequence-classification softmax-backward scatter return value directly in
the final padded [4, 8192] layout, including the scalar div/where producer, the
row softmax gradient over [8, 2], duplicate-index index_put(accumulate=True)
into the logical [8, 1024, 2] buffer, view, permute, and constant_pad_nd
epilogue, whereas Inductor currently lowers the decomposed softmax-gradient
producer, generic scatter-add, transpose, and pad as separate scheduled work
around a materialized scatter buffer; Inductor cannot do this today because its
scheduler/codegen does not recognize a row-local softmax-backward producer
feeding a structured duplicate-index scatter-reduce whose only live consumer is
a padded transposed layout; the fix is SCATTER_REDUCE: add an indexed
scatter-reduce lowering that keeps the row-local producer in registers and emits
the final padded transpose layout directly.
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
    oracle_impl,
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

ROWS = 8
SCATTER_COLS = 1024
CHANNELS = 2
OUT_ROWS = 4
OUT_COLS = ROWS * SCATTER_COLS
OUT_SHAPE = (OUT_ROWS, OUT_COLS)
OUT_STRIDE = (OUT_COLS, 1)
BLOCK_N = 1024


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_scope_scatter_pad_kernel(
        arg293_ptr,
        arg230_ptr,
        arg232_ptr,
        arg231_ptr,
        arg229_ptr,
        arg294_ptr,
        arg78_ptr,
        arg228_ptr,
        out_ptr,
        arg232_s0: tl.constexpr,
        arg232_s1: tl.constexpr,
        arg231_s0: tl.constexpr,
        arg231_s1: tl.constexpr,
        arg229_s0: tl.constexpr,
        arg229_s1: tl.constexpr,
        arg294_s0: tl.constexpr,
        arg294_s1: tl.constexpr,
        out_s0: tl.constexpr,
        out_s1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        out_row = tl.program_id(0)
        col_block = tl.program_id(1)
        cols = col_block * BLOCK + tl.arange(0, BLOCK)
        active = cols < 8192
        acc = tl.full((BLOCK,), 0.0, tl.float32)

        if out_row < 2:
            channel = out_row
            div_value = tl.load(arg293_ptr).to(tl.float32) / tl.load(arg230_ptr).to(
                tl.float32
            )

            for source_row in tl.static_range(0, 8):
                scatter_row = tl.load(arg78_ptr + source_row).to(tl.int64)
                scatter_col = tl.load(arg228_ptr + source_row).to(tl.int64)
                target_col = scatter_row * 1024 + scatter_col

                row_mask = tl.load(
                    arg231_ptr + source_row * arg231_s0,
                )
                where_scale = tl.where(row_mask, div_value, 0.0)

                pred0 = tl.load(arg232_ptr + source_row * arg232_s0)
                pred1 = tl.load(arg232_ptr + source_row * arg232_s0 + arg232_s1)
                mask_grad0 = tl.where(pred0, -1.0, 0.0) * where_scale
                mask_grad1 = tl.where(pred1, -1.0, 0.0) * where_scale
                mask_grad_sum = mask_grad0 + mask_grad1

                x0 = tl.load(arg229_ptr + source_row * arg229_s0).to(tl.float32)
                x1 = tl.load(arg229_ptr + source_row * arg229_s0 + arg229_s1).to(
                    tl.float32
                )
                row_max = tl.maximum(x0, x1)
                shifted0 = x0 - row_max
                shifted1 = x1 - row_max
                exp0 = tl.exp(shifted0)
                exp1 = tl.exp(shifted1)
                log_sum = tl.log(exp0 + exp1)

                if channel == 0:
                    prob = tl.exp(shifted0 - log_sum)
                    mask_grad = mask_grad0
                    residual = tl.load(arg294_ptr + source_row * arg294_s0).to(
                        tl.float32
                    )
                else:
                    prob = tl.exp(shifted1 - log_sum)
                    mask_grad = mask_grad1
                    residual = tl.load(
                        arg294_ptr + source_row * arg294_s0 + arg294_s1
                    ).to(tl.float32)

                value = residual + (mask_grad - prob * mask_grad_sum)
                acc += tl.where(cols == target_col, value, 0.0)

        tl.store(out_ptr + out_row * out_s0 + cols * out_s1, acc, mask=active)


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 9:
        raise ValueError(f"expected 9 inputs, got {len(inputs)}")
    (
        arg293_1,
        arg230_1,
        arg232_1,
        arg231_1,
        arg229_1,
        arg294_1,
        arg78_1,
        arg228_1,
        shape_param_0,
    ) = inputs
    del arg293_1, arg230_1

    expected = {
        "arg232_1": (arg232_1, (ROWS, CHANNELS), torch.bool),
        "arg231_1": (arg231_1, (ROWS, 1), torch.bool),
        "arg229_1": (arg229_1, (ROWS, CHANNELS), torch.float32),
        "arg294_1": (arg294_1, (ROWS, CHANNELS), torch.float32),
        "arg78_1": (arg78_1, (ROWS,), torch.int64),
        "arg228_1": (arg228_1, (ROWS,), torch.int64),
    }
    for name, (tensor, shape, dtype) in expected.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device.type != "cuda":
            raise RuntimeError("Triton oracle requires CUDA inputs")
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape} dtype={dtype}, "
                f"got shape={tuple(tensor.shape)} dtype={tensor.dtype}"
            )

    if list(shape_param_0) != [OUT_COLS, CHANNELS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param_0}")


@oracle_impl(hardware="H100", shapes="(T([], f32), T([], f32), T([8, 2], b8), T([8, 1], b8), T([8, 2], f32), T([8, 2], f32), T([8], i64, gen=Index(8)), T([8], i64, gen=Index(8)), S([8192, 2]))")
def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    (
        arg293_1,
        arg230_1,
        arg232_1,
        arg231_1,
        arg229_1,
        arg294_1,
        arg78_1,
        arg228_1,
        _shape_param_0,
    ) = inputs
    del _shape_param_0

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=arg229_1.device,
        dtype=torch.float32,
    )
    _full_scope_scatter_pad_kernel[
        (OUT_ROWS, triton.cdiv(OUT_COLS, BLOCK_N))
    ](
        arg293_1,
        arg230_1,
        arg232_1,
        arg231_1,
        arg229_1,
        arg294_1,
        arg78_1,
        arg228_1,
        out,
        arg232_s0=arg232_1.stride(0),
        arg232_s1=arg232_1.stride(1),
        arg231_s0=arg231_1.stride(0),
        arg231_s1=arg231_1.stride(1),
        arg229_s0=arg229_1.stride(0),
        arg229_s1=arg229_1.stride(1),
        arg294_s0=arg294_1.stride(0),
        arg294_s1=arg294_1.stride(1),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        BLOCK=BLOCK_N,
        num_warps=4,
    )
    return out


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
