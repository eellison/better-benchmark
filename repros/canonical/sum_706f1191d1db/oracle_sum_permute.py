"""
Oracle for sum_706f1191d1db.

Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full
Repro.forward scope by recognizing that full([128,1,768], 0) followed by
select_scatter(..., arg280_1, dim=1, index=0) and view([128,768]) is the
original contiguous float32[128,768] value, then one Triton kernel reads each
input element once, coalesces the backing-store writes for the required
permuted float32[768,128] side output with stride (1,768), and writes the dim-0
sum viewed as float32[768], whereas Inductor emits one generic outer-reduction
kernel for the full/select_scatter/view producer, materialized side output, and
sibling sum; Inductor cannot generate this schedule today because the reduction
scheduler does not model a complete singleton select_scatter identity feeding a
coalesced materialized-view store plus compatible column accumulator as a
shape-specialized multi-output template; the fix is SCHEDULER_FUSION: add a
layout-materializing multi-output reduction template for these identity
producer plus side-reduction cases.
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

ROWS = 128
COLS = 768
PERMUTE_SHAPE = (COLS, ROWS)
PERMUTE_STRIDE = (1, COLS)
SUM_SHAPE = (COLS,)
SUM_STRIDE = (1,)
BLOCK_N = 16


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _sum_permute_kernel(
        input_ptr,
        permute_out_ptr,
        sum_out_ptr,
        in_s0: tl.constexpr,
        in_s1: tl.constexpr,
        permute_s0: tl.constexpr,
        permute_s1: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        N_COLS: tl.constexpr,
    ):
        col_block = tl.program_id(0)
        rows = tl.arange(0, BLOCK_M)
        cols = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N_COLS

        input_offsets = rows[:, None] * in_s0 + cols[None, :] * in_s1
        values = tl.load(input_ptr + input_offsets, mask=mask[None, :], other=0.0)

        permute_offsets = rows[:, None] * permute_s1 + cols[None, :] * permute_s0
        tl.store(
            permute_out_ptr + permute_offsets,
            values,
            mask=mask[None, :],
        )

        sums = tl.sum(values, axis=0)
        tl.store(sum_out_ptr + cols, sums, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter to be iterable, got {value!r}") from exc


def _validate_inputs(inputs: tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    arg280_1, shape_param_0, shape_param_1 = inputs
    if not isinstance(arg280_1, torch.Tensor):
        raise TypeError("expected arg280_1 to be a tensor")
    if arg280_1.device.type != "cuda":
        raise RuntimeError("Triton oracle requires CUDA inputs")
    if tuple(arg280_1.shape) != (ROWS, COLS):
        raise ValueError(f"unexpected arg280_1 shape: {tuple(arg280_1.shape)}")
    if arg280_1.dtype is not torch.float32:
        raise TypeError(f"expected float32 arg280_1, got {arg280_1.dtype}")
    if not arg280_1.is_contiguous():
        raise ValueError("oracle expects the captured contiguous input layout")
    if _shape_tuple(shape_param_0) != (ROWS, COLS):
        raise ValueError(f"unexpected _shape_param_0: {shape_param_0!r}")
    if _shape_tuple(shape_param_1) != SUM_SHAPE:
        raise ValueError(f"unexpected _shape_param_1: {shape_param_1!r}")
    return arg280_1


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full full/select_scatter/view -> permute + dim-0 sum scope."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")

    arg280_1 = _validate_inputs(inputs)
    permute_out = torch.empty_strided(
        PERMUTE_SHAPE,
        PERMUTE_STRIDE,
        device=arg280_1.device,
        dtype=arg280_1.dtype,
    )
    sum_out = torch.empty_strided(
        SUM_SHAPE,
        SUM_STRIDE,
        device=arg280_1.device,
        dtype=arg280_1.dtype,
    )

    grid = (triton.cdiv(COLS, BLOCK_N),)
    _sum_permute_kernel[grid](
        arg280_1,
        permute_out,
        sum_out,
        in_s0=arg280_1.stride(0),
        in_s1=arg280_1.stride(1),
        permute_s0=permute_out.stride(0),
        permute_s1=permute_out.stride(1),
        BLOCK_M=ROWS,
        BLOCK_N=BLOCK_N,
        N_COLS=COLS,
        num_warps=4,
        num_stages=1,
    )
    return permute_out, sum_out


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
