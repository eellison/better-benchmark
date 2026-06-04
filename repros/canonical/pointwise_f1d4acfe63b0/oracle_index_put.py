"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete `eq(arg0, -1)` mask, unsqueezed row mask, zero-valued `where`, full zero tensor, duplicate-preserving `index_put(..., accumulate=True)`, and final dense `[128, 2560]` output by zero-filling the returned tensor and scattering masked row updates directly with Triton, whereas Inductor currently lowers the graph as generic pointwise mask/where materialization plus zero-fill and index_put accumulation work; Inductor cannot do this today because scheduler/codegen does not canonicalize a row-wise masked full-output index_put accumulate into a scatter-reduce schedule that folds source zeroing into the indexed update; the fix is SCATTER_REDUCE: add an index_put-accumulate lowering that recognizes sentinel-zeroed row scatters and emits a direct dense-output scatter-reduce kernel."""
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

CLASSIFICATION = "SCATTER_REDUCE"
ROWS = 128
COLS = 2560
BLOCK_COLS = 1024
NUM_WARPS = 8


def get_inputs() -> tuple[Any, ...]:
    """Load the exact default inputs from repro.py."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _masked_row_index_put_scatter_kernel(
        index_ptr,
        values_ptr,
        out_ptr,
        index_stride0: tl.constexpr,
        values_stride0: tl.constexpr,
        values_stride1: tl.constexpr,
        out_stride0: tl.constexpr,
        out_stride1: tl.constexpr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_COLS_: tl.constexpr,
    ):
        src_row = tl.program_id(0)
        col_offsets = tl.program_id(1) * BLOCK_COLS_ + tl.arange(0, BLOCK_COLS_)
        col_mask = col_offsets < COLS_

        raw_index = tl.load(index_ptr + src_row * index_stride0).to(tl.int64)
        wrapped_index = tl.where(raw_index < 0, raw_index + ROWS_, raw_index)
        active = (raw_index != -1) & (wrapped_index >= 0) & (wrapped_index < ROWS_)

        values = tl.load(
            values_ptr + src_row * values_stride0 + col_offsets * values_stride1,
            mask=col_mask & active,
            other=0.0,
        ).to(tl.float32)
        tl.atomic_add(
            out_ptr + wrapped_index * out_stride0 + col_offsets * out_stride1,
            values,
            sem="relaxed",
            mask=col_mask & active,
        )


def _validate_inputs(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if len(inputs) != 2:
        raise ValueError(f"expected 2 inputs, got {len(inputs)}")

    arg0_1, arg1_1 = inputs
    if not isinstance(arg0_1, torch.Tensor) or not isinstance(arg1_1, torch.Tensor):
        raise TypeError("expected tensor inputs")
    if arg0_1.device.type != "cuda" or arg1_1.device.type != "cuda":
        raise RuntimeError("the Triton oracle requires CUDA inputs")
    if arg0_1.dtype is not torch.int64:
        raise TypeError(f"arg0_1 must be int64, got {arg0_1.dtype}")
    if arg1_1.dtype is not torch.float32:
        raise TypeError(f"arg1_1 must be float32, got {arg1_1.dtype}")
    if tuple(arg0_1.shape) != (ROWS,):
        raise ValueError(f"arg0_1 expected shape {(ROWS,)}, got {tuple(arg0_1.shape)}")
    if tuple(arg1_1.shape) != (ROWS, COLS):
        raise ValueError(f"arg1_1 expected shape {(ROWS, COLS)}, got {tuple(arg1_1.shape)}")
    return arg0_1, arg1_1


def oracle_forward(inputs: tuple[Any, ...]) -> torch.Tensor:
    """Run the full eq/unsqueeze/where/full/index_put graph on the exact inputs."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    arg0_1, arg1_1 = _validate_inputs(inputs)
    out = torch.empty_strided(
        (ROWS, COLS),
        (COLS, 1),
        device=arg1_1.device,
        dtype=torch.float32,
    )
    out.zero_()
    _masked_row_index_put_scatter_kernel[(ROWS, triton.cdiv(COLS, BLOCK_COLS))](
        arg0_1,
        arg1_1,
        out,
        index_stride0=int(arg0_1.stride(0)),
        values_stride0=int(arg1_1.stride(0)),
        values_stride1=int(arg1_1.stride(1)),
        out_stride0=int(out.stride(0)),
        out_stride1=int(out.stride(1)),
        ROWS_=ROWS,
        COLS_=COLS,
        BLOCK_COLS_=BLOCK_COLS,
        num_warps=NUM_WARPS,
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
                true_floor = result["status"] == "GOOD"
                print(f"true floor: {'yes' if true_floor else 'no'} ({result['status']})")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            true_floor = result["status"] == "GOOD"
            print(f"true floor: {'yes' if true_floor else 'no'} ({result['status']})")


if __name__ == "__main__":
    main()
