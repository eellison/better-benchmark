"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete masked matrix return and sibling column-sum return from Repro.forward in one Triton reduction kernel, loading each `arg21_1`/`mm` tile once, applying the `arg21_1 <= 0` zero mask, writing the required `[4096, 1024]` transposed-view layout, and accumulating the `[4096]` sum side output from the same values, whereas Inductor currently lowers the decomposed where, permute metadata, and dim-0 sum through its generic reduction schedule with extra multi-output reduction overhead; Inductor cannot do this today because its scheduler/codegen does not form a full-scope plan that shares the masked pointwise producer between a materialized layout-changing output and a sibling column reduction; the fix is SCHEDULER_FUSION: teach reduction scheduling to keep such same-source pointwise producers virtual and emit one multi-output reduction/materialization kernel for compatible static shapes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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

ROWS = 1024
COLS = 4096


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=8, num_stages=3),
        ],
        key=[],
    )
    @triton.jit
    def _masked_transpose_sum_kernel(
        mask_input_ptr,
        source_ptr,
        out_transposed_ptr,
        out_sum_ptr,
        ROWS_: tl.constexpr,
        COLS_: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, ROWS_)
        offsets = rows[:, None] * COLS_ + cols[None, :]
        active = cols[None, :] < COLS_

        mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source = tl.load(source_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        values = tl.where(mask_input <= 0.0, 0.0, source)

        tl.store(out_transposed_ptr + offsets, values, mask=active)
        sums = tl.sum(values, axis=0)
        tl.store(out_sum_ptr + cols, sums, mask=cols < COLS_)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    arg21_1, mm, shape_param = inputs
    if not isinstance(arg21_1, torch.Tensor) or not isinstance(mm, torch.Tensor):
        raise TypeError("expected tensor inputs for arg21_1 and mm")
    if tuple(arg21_1.shape) != (ROWS, COLS) or tuple(mm.shape) != (ROWS, COLS):
        raise ValueError(f"expected two [{ROWS}, {COLS}] inputs, got {arg21_1.shape} and {mm.shape}")
    if arg21_1.dtype is not torch.float32 or mm.dtype is not torch.float32:
        raise ValueError(f"expected float32 inputs, got {arg21_1.dtype} and {mm.dtype}")
    if not arg21_1.is_cuda or not mm.is_cuda:
        raise ValueError("CUDA tensors are required")
    if not arg21_1.is_contiguous() or not mm.is_contiguous():
        raise ValueError(f"expected contiguous inputs, got strides {arg21_1.stride()} and {mm.stride()}")
    if tuple(int(x) for x in shape_param) != (COLS,):
        raise ValueError(f"expected shape parameter [{COLS}], got {shape_param}")
    return arg21_1, mm, shape_param


@oracle_impl(hardware="H100", shapes="(T([1024, 4096], f32), T([1024, 4096], f32), S([4096]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope with the same output count/layouts."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    arg21_1, mm, shape_param = _validate_inputs(inputs)
    out_transposed = torch.empty_strided(
        (COLS, ROWS),
        (1, COLS),
        device=mm.device,
        dtype=mm.dtype,
    )
    out_sum = torch.empty_strided(
        tuple(int(x) for x in shape_param),
        (1,),
        device=mm.device,
        dtype=mm.dtype,
    )

    grid = lambda meta: (triton.cdiv(COLS, meta["BLOCK_N"]),)
    _masked_transpose_sum_kernel[grid](
        arg21_1,
        mm,
        out_transposed,
        out_sum,
        ROWS_=ROWS,
        COLS_=COLS,
    )
    return out_transposed, out_sum


def main():
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
