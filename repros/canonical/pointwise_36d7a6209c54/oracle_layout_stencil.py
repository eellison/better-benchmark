"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full scalar `arg2 >= arg3` predicate, two int64 table gathers from `arg4[arg2]` and `arg4[arg3]`, equality test, bitwise-and, and scalar bool return in one minimal Triton program, whereas Inductor currently lowers the same zero-dimensional indexed pointwise graph through its generic single-kernel pointwise/index codegen and is already near launch floor; Inductor cannot do this today because its pattern library has no dedicated zero-dimensional indexed-predicate template that strips the generic pointwise indexing scaffold for scalar inputs and outputs; the fix is NEW_PATTERN: add a guarded scalar indexed-predicate lowering for fixed-shape zero-dimensional lookup/compare expressions."""
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


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "NEW_PATTERN"
TABLE_N = 6144


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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _scalar_index_compare_kernel(arg2_ptr, arg3_ptr, table_ptr, output_ptr):
        idx2 = tl.load(arg2_ptr)
        idx3 = tl.load(arg3_ptr)
        val2 = tl.load(table_ptr + idx2.to(tl.int64))
        val3 = tl.load(table_ptr + idx3.to(tl.int64))
        tl.store(output_ptr, (idx2 >= idx3) & (val2 == val3))


def _validate_inputs(inputs: tuple[Any, ...] | list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects three inputs, got {len(inputs)}")

    arg2, arg3, table = inputs
    if not isinstance(arg2, torch.Tensor):
        raise TypeError(f"{REPRO_ID} arg2 must be a tensor")
    if not isinstance(arg3, torch.Tensor):
        raise TypeError(f"{REPRO_ID} arg3 must be a tensor")
    if not isinstance(table, torch.Tensor):
        raise TypeError(f"{REPRO_ID} arg4 must be a tensor")
    if arg2.shape != torch.Size([]) or arg3.shape != torch.Size([]):
        raise ValueError(f"expected scalar index tensors, got {tuple(arg2.shape)} and {tuple(arg3.shape)}")
    if arg2.dtype != torch.int32 or arg3.dtype != torch.int32:
        raise TypeError(f"expected int32 scalar indices, got {arg2.dtype} and {arg3.dtype}")
    if tuple(table.shape) != (TABLE_N,) or table.dtype != torch.int64:
        raise TypeError(f"expected int64[{TABLE_N}] table, got {table.dtype}{tuple(table.shape)}")
    if not (arg2.is_cuda and arg3.is_cuda and table.is_cuda):
        raise ValueError("oracle_layout_stencil.py expects CUDA inputs")
    if arg2.device != arg3.device or arg2.device != table.device:
        raise ValueError("all inputs must be on the same CUDA device")
    if not table.is_contiguous():
        raise ValueError("expected contiguous int64 lookup table")
    return arg2, arg3, table


@oracle_impl(hardware="H100", shapes="(T([], i32, gen=Index(6144)), T([], i32, gen=Index(6144)), T([6144], i64, gen=Index(6144)))")
def oracle_forward(inputs):
    """Run the exact full Repro.forward scope with one scalar Triton kernel."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_layout_stencil.py")

    arg2, arg3, table = _validate_inputs(inputs)
    output = torch.empty((), device=table.device, dtype=torch.bool)
    _scalar_index_compare_kernel[(1,)](arg2, arg3, table, output, num_warps=1)
    return output


def _check_layout(output: torch.Tensor) -> bool:
    return (
        output.shape == torch.Size([])
        and output.dtype == torch.bool
        and output.stride() == ()
    )


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
        with torch.no_grad():
            layout_out = oracle_forward(inputs)
            torch.cuda.synchronize()
        layout_ok = _check_layout(layout_out)
        print(
            f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
            f"(shape={list(layout_out.shape)} stride={layout_out.stride()} "
            f"dtype={layout_out.dtype})"
        )
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
