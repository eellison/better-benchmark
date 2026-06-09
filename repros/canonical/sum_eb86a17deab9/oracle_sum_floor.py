"""Gap diagnosis (classification: BANDWIDTH_BOUND): this diagnosis-only oracle computes the complete `sum(arg303_1, dim=[0, 1])` Repro.forward scope for contiguous f32[B, S, H] inputs, including the default f32[8, 1024, 768] case, as a shape-specialized row-tiled Triton reduction that reads the full tensor once, atomically accumulates column partials, and writes the final contiguous f32[H] output, whereas tuned Inductor already lowers this isolated one-op reduction to the same mandatory input-read, column-accumulation, and tiny output-store envelope; Inductor cannot be assigned a material local gap today because the captured graph has no producer, consumer, epilogue, scatter, layout transform, or materialized intermediate to remove, and the remaining work is dominated by the required read plus reduction/launch overhead; the fix is BANDWIDTH_BOUND: record this as an at-floor simple reduction unless broader reduction-template or launch-overhead changes move the whole family."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps syntax checks usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "BANDWIDTH_BOUND"

TILE_ROWS = 256
TILE_COLS = 32


from oracle_harness import (
    oracle_impl,  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _zero_output_kernel(
        out_ptr,
        COLS_: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        tl.store(out_ptr + cols, tl.zeros((BLOCK_COLS,), dtype=tl.float32), mask=cols < COLS_)

    @triton.jit
    def _atomic_sum_rows_kernel(
        x_ptr,
        out_ptr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_COLS: tl.constexpr,
    ):
        row_tile = tl.program_id(0)
        col_tile = tl.program_id(1)
        rows = row_tile * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = col_tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
        mask = (rows[:, None] < ROWS_) & (cols[None, :] < COLS_)
        values = tl.load(
            x_ptr + rows[:, None] * COLS_ + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        partial = tl.sum(values, axis=0)
        tl.atomic_add(out_ptr + cols, partial, sem="relaxed", mask=cols < COLS_)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (x,) = inputs
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(x)!r}")
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input, got shape={tuple(x.shape)}")
    if x.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {x.dtype}")
    if not x.is_cuda:
        raise ValueError("oracle_sum_floor.py expects CUDA inputs")
    if not x.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={x.stride()}")
    return x


@oracle_impl(hardware="H100", shapes="(T([8, 1024, 768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the exact full-scope `[B, S, H] -> [H]` reduction."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum_floor.py")

    x = _validate_inputs(inputs)
    rows = x.shape[0] * x.shape[1]
    cols = x.shape[2]
    out = torch.empty_strided((cols,), (1,), device=x.device, dtype=x.dtype)

    _zero_output_kernel[(triton.cdiv(cols, TILE_COLS),)](
        out,
        COLS_=cols,
        BLOCK_COLS=TILE_COLS,
        num_warps=1,
        num_stages=3,
    )
    _atomic_sum_rows_kernel[(triton.cdiv(rows, TILE_ROWS), triton.cdiv(cols, TILE_COLS))](
        x,
        out,
        ROWS_=rows,
        COLS_=cols,
        BLOCK_ROWS=TILE_ROWS,
        BLOCK_COLS=TILE_COLS,
        num_warps=4,
        num_stages=3,
    )
    return out


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
                    print(f"diagnosis_only: required comparison shows not_true_floor "
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
