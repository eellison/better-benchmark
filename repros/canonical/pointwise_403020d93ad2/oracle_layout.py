"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full `b8[8, 1024]` output by applying the exact ordered `f32 < 0` predicate in one static Triton row kernel with the returned contiguous bool layout, whereas Inductor currently lowers `aten.lt.Scalar(arg0, 0)` through generic pointwise float-comparison codegen for this tiny output; Inductor cannot do this today because its pointwise scheduler/codegen has no specialized static zero-comparison lowering that bypasses generic indexing and launch-shape heuristics for a contiguous bool store; the fix is NEW_PATTERN: add an Inductor `x < 0.0` f32-to-bool pointwise pattern that preserves `-0.0` and NaN semantics while emitting the minimal static contiguous-output compare kernel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # pragma: no cover - keeps py_compile useful.
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

INPUT_SHAPE = (8, 1024)
OUTPUT_SHAPE = INPUT_SHAPE
OUTPUT_STRIDE = (1024, 1)
NUMEL = 8 * 1024
BLOCK_N = 2048


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _lt_zero_f32_kernel(
        x_ptr,
        out_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_SIZE)
        offsets = block_id * BLOCK_SIZE + cols
        values = tl.load(x_ptr + offsets) < 0.0
        tl.store(out_ptr + offsets, values)


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope for the fixed f32[8, 1024] input."""
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (arg0,) = inputs
    if not isinstance(arg0, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor")
    if tuple(arg0.shape) != INPUT_SHAPE:
        raise ValueError(f"{REPRO_ID} expects shape {INPUT_SHAPE}, got {tuple(arg0.shape)}")
    if arg0.dtype is not torch.float32:
        raise ValueError(f"{REPRO_ID} expects torch.float32 input, got {arg0.dtype}")
    if not arg0.is_cuda:
        raise ValueError(f"{REPRO_ID} expects a CUDA input")
    if not arg0.is_contiguous():
        raise ValueError(f"{REPRO_ID} expects contiguous input, got stride={arg0.stride()}")

    out = torch.empty_like(arg0, dtype=torch.bool)
    _lt_zero_f32_kernel[(NUMEL // BLOCK_N,)](
        arg0,
        out,
        BLOCK_SIZE=BLOCK_N,
        num_warps=8,
    )
    return out


def _check_layout(instance: torch.nn.Module, inputs: list[object] | tuple[object, ...]) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    ok = (
        tuple(oracle_out.shape) == tuple(eager_out.shape) == OUTPUT_SHAPE
        and oracle_out.dtype == eager_out.dtype == torch.bool
        and oracle_out.stride() == eager_out.stride() == OUTPUT_STRIDE
        and oracle_out.storage_offset() == eager_out.storage_offset() == 0
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype} storage_offset={oracle_out.storage_offset()})"
    )
    return ok


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
        ok = ok and _check_layout(instance, inputs)
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
