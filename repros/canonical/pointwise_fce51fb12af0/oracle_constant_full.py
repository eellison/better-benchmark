"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete no-input `aten.full([32, 3, 224, 224], 1.0)` repro by allocating the exact contiguous `float32` output and filling it with one Triton store kernel, whereas Inductor already lowers the same graph to equivalent no-load stores through the generic pointwise scheduler; Inductor cannot materially improve this inside the captured region because the user-visible result is a fresh 19 MB CUDA tensor that must be written, so the fix is BANDWIDTH_BOUND: treat this as the constant materialization memory-store floor rather than a missing fusion or algebraic optimization."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


OUTPUT_SHAPE = (32, 3, 224, 224)
OUTPUT_STRIDE = (150528, 50176, 224, 1)
OUTPUT_NUMEL = 32 * 3 * 224 * 224
OUTPUT_DTYPE = torch.float32
BLOCK_SIZE = 1024


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _fill_one_f32_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        values = tl.full((block_size,), 1.0, tl.float32)
        tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs: list[object] | tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the exact eager output."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=torch.device("cuda", 0),
        dtype=OUTPUT_DTYPE,
    )
    grid = (triton.cdiv(OUTPUT_NUMEL, BLOCK_SIZE),)
    _fill_one_f32_kernel[grid](
        output,
        n_elements=OUTPUT_NUMEL,
        block_size=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return output


def _check_layout_and_bits(
    instance: torch.nn.Module,
    inputs: list[object] | tuple[object, ...],
) -> bool:
    with torch.no_grad():
        eager_out = instance(*inputs)
        oracle_out = oracle_forward(inputs)
        torch.cuda.synchronize()

    layout_ok = (
        tuple(oracle_out.shape) == OUTPUT_SHAPE
        and oracle_out.stride() == OUTPUT_STRIDE
        and oracle_out.dtype is OUTPUT_DTYPE
        and oracle_out.storage_offset() == 0
    )
    bits_ok = torch.equal(eager_out.view(torch.uint32), oracle_out.view(torch.uint32))
    print(
        f"  output 0 layout: {'PASS' if layout_ok else 'FAIL'} "
        f"(shape={list(oracle_out.shape)} stride={oracle_out.stride()} "
        f"dtype={oracle_out.dtype})"
    )
    print(f"  output 0 bit_pattern: {'PASS' if bits_ok else 'FAIL'}")
    return layout_ok and bits_ok


# --- CLI entry point ---
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

    # Handle --show-hw early
    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    if has_stochastic_ops(REPRO_PATH):
        print(
            f"NOTE: {REPRO_ID} contains stochastic ops; "
            "affected outputs will be auto-skipped"
        )

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
        ok = _check_layout_and_bits(instance, inputs) and ok
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
                        "WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            # The shared harness owns timing so graph capture, GPU locking, and
            # interleaved oracle/compile measurement stay intact.
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(
                    "WARNING: oracle is slower than compile "
                    f"(ratio={result['ratio']:.3f}x)"
                )


if __name__ == "__main__":
    main()
