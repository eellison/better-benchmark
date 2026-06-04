"""
Oracle for pointwise_b869961e6b54

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: The oracle covers the full no-input repro by
    allocating the exact eager `float32[8, 1, 1, 1024]` layout and using one
    Triton kernel to materialize the algebraically folded result of
    `1.0 - full(..., 1.0)`, namely zeros.
  Why Inductor cannot/does not do this today: Inductor already performs the
    same constant simplification and emits one store-only pointwise kernel; it
    cannot remove the fresh output allocation or the 32 KiB store without
    changing the user-visible tensor result or relying on a larger surrounding
    graph to consume the value differently.
  Required Inductor change: BANDWIDTH_BOUND; no scheduler, scatter/reduce,
    split-K, algebraic, or recomputation change is expected to materially beat
    the one-launch fresh-output floor for this isolated graph.
"""
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
CLASSIFICATION = "BANDWIDTH_BOUND"

OUTPUT_SHAPE = (8, 1, 1, 1024)
OUTPUT_STRIDE = (1024, 1024, 1024, 1)
OUTPUT_NUMEL = 8 * 1 * 1 * 1024
BLOCK_SIZE = 1024


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _zero_fill_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        values = tl.full((block_size,), 0.0, tl.float32)
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with one Triton zero-fill kernel."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    grid = (triton.cdiv(OUTPUT_NUMEL, BLOCK_SIZE),)
    _zero_fill_kernel[grid](
        output,
        n_elements=OUTPUT_NUMEL,
        block_size=BLOCK_SIZE,
        num_warps=4,
    )
    return output


def _check_layout(inputs: list[object] | tuple[object, ...]) -> bool:
    with torch.no_grad():
        output = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = (
        tuple(output.shape) == OUTPUT_SHAPE
        and output.stride() == OUTPUT_STRIDE
        and output.dtype is torch.float32
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(shape={list(output.shape)} stride={output.stride()} dtype={output.dtype})"
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify correctness against eager Repro",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark oracle vs torch.compile",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for correctness check",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for correctness check",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=25,
        help="Warmup iterations for benchmark",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=200,
        help="Repetitions for benchmark",
    )
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument(
        "--all-shapes",
        action="store_true",
        help="Benchmark across all shapes from shapes.txt",
    )
    parser.add_argument(
        "--show-hw",
        action="store_true",
        help="Print GPU hardware info and exit",
    )
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
        ok = _check_layout(inputs) and ok
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
