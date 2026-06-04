"""
Oracle for sum_sum_sum_72f8a586759c

Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full
nanogpt layer-norm-backward-shaped graph from the sparse index_put producer,
including the logical [1,64,768] scatter, all pointwise products/subtractions,
three live reductions, the materialized [768,64] permuted side output, and the
final [768] view, whereas Inductor currently schedules the zero/index_put
producer, row reductions, pointwise epilogue, permute, and sibling reductions as
generic dense work; Inductor cannot do this today because its scheduler/codegen
does not recognize a zero-fill plus single-row scatter feeding dependent
reductions and a sparse materialized transpose; the fix is SCATTER_REDUCE: add
a structured scatter-reduce lowering that keeps the one live row in registers
and writes the required output layouts directly.
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

BATCH = 1
ROWS = 64
HIDDEN = 768
BLOCK_H = 1024


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _full_scope_index_put_kernel(
        mm_ptr,
        index_ptr,
        arg74_ptr,
        arg232_ptr,
        arg235_ptr,
        out0_ptr,
        out1_ptr,
        out2_ptr,
        out3_ptr,
        mm_s0: tl.constexpr,
        mm_s1: tl.constexpr,
        arg74_s0: tl.constexpr,
        arg232_s0: tl.constexpr,
        arg232_s1: tl.constexpr,
        arg232_s2: tl.constexpr,
        arg235_s0: tl.constexpr,
        arg235_s1: tl.constexpr,
        arg235_s2: tl.constexpr,
        out2_s0: tl.constexpr,
        out2_s1: tl.constexpr,
        BLOCK: tl.constexpr,
        N_ROWS: tl.constexpr,
        N_HIDDEN: tl.constexpr,
    ):
        out_col = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < N_HIDDEN

        raw_index = tl.load(index_ptr).to(tl.int64)
        scatter_row = tl.where(raw_index < 0, raw_index + N_ROWS, raw_index)
        is_live_col = out_col == scatter_row

        mm = tl.load(mm_ptr + 0 * mm_s0 + offsets * mm_s1, mask=mask, other=0.0).to(
            tl.float32
        )
        weight = tl.load(arg74_ptr + offsets * arg74_s0, mask=mask, other=0.0).to(
            tl.float32
        )
        scatter_arg232 = tl.load(
            arg232_ptr + 0 * arg232_s0 + scatter_row * arg232_s1 + offsets * arg232_s2,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        index_put_mul = mm * weight
        row_sum = tl.sum(index_put_mul, axis=0)
        weighted_row_sum = tl.sum(index_put_mul * scatter_arg232, axis=0)
        scale = tl.load(
            arg235_ptr + 0 * arg235_s0 + scatter_row * arg235_s1 + 0 * arg235_s2
        ).to(tl.float32)

        side_value = scale * (
            index_put_mul * 768.0 - row_sum - scatter_arg232 * weighted_row_sum
        )
        permuted_value = tl.where(is_live_col, side_value, 0.0)

        tl.store(
            out2_ptr + offsets * out2_s0 + out_col * out2_s1,
            permuted_value,
            mask=mask,
        )
        tl.store(out0_ptr + offsets, mm * scatter_arg232, mask=mask & is_live_col)
        tl.store(out1_ptr + offsets, mm, mask=mask & is_live_col)
        tl.store(out3_ptr + offsets, side_value, mask=mask & is_live_col)


def _validate_inputs(inputs: tuple[Any, ...]) -> None:
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    (
        mm,
        arg233_1,
        arg74_1,
        arg232_1,
        arg235_1,
        shape_param_0,
        shape_param_1,
        shape_param_2,
    ) = inputs

    expected = {
        "mm": (mm, (BATCH, HIDDEN), torch.float32),
        "arg233_1": (arg233_1, (1,), torch.int64),
        "arg74_1": (arg74_1, (HIDDEN,), torch.float32),
        "arg232_1": (arg232_1, (BATCH, ROWS, HIDDEN), torch.float32),
        "arg235_1": (arg235_1, (BATCH, ROWS, 1), torch.float32),
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

    if list(shape_param_0) != [BATCH, 1, HIDDEN]:
        raise ValueError(f"unexpected first view shape parameter: {shape_param_0}")
    if list(shape_param_1) != [ROWS, HIDDEN]:
        raise ValueError(f"unexpected second view shape parameter: {shape_param_1}")
    if list(shape_param_2) != [HIDDEN]:
        raise ValueError(f"unexpected final view shape parameter: {shape_param_2}")


def oracle_forward(inputs: tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    if triton is None:
        raise RuntimeError("triton is not available")

    _validate_inputs(inputs)
    (
        mm,
        arg233_1,
        arg74_1,
        arg232_1,
        arg235_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs

    out0 = torch.empty_strided((HIDDEN,), (1,), device=mm.device, dtype=torch.float32)
    out1 = torch.empty_strided((HIDDEN,), (1,), device=mm.device, dtype=torch.float32)
    out2 = torch.empty_strided(
        (HIDDEN, ROWS),
        (1, HIDDEN),
        device=mm.device,
        dtype=torch.float32,
    )
    out3 = torch.empty_strided((HIDDEN,), (1,), device=mm.device, dtype=torch.float32)

    _full_scope_index_put_kernel[(ROWS,)](
        mm,
        arg233_1,
        arg74_1,
        arg232_1,
        arg235_1,
        out0,
        out1,
        out2,
        out3,
        mm_s0=mm.stride(0),
        mm_s1=mm.stride(1),
        arg74_s0=arg74_1.stride(0),
        arg232_s0=arg232_1.stride(0),
        arg232_s1=arg232_1.stride(1),
        arg232_s2=arg232_1.stride(2),
        arg235_s0=arg235_1.stride(0),
        arg235_s1=arg235_1.stride(1),
        arg235_s2=arg235_1.stride(2),
        out2_s0=out2.stride(0),
        out2_s1=out2.stride(1),
        BLOCK=BLOCK_H,
        N_ROWS=ROWS,
        N_HIDDEN=HIDDEN,
        num_warps=8,
    )
    return out0, out1, out2, out3


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
