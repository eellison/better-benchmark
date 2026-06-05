"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle executes the complete no-input Repro.forward scope by allocating the fresh contiguous cuda float64[200, 200, 26] tensor and filling all elements with zero using one Triton store kernel, whereas Inductor already lowers this standalone aten.full to the same essential allocation plus constant-store work; Inductor cannot materially improve this local region through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the mandatory fresh allocation, one launch, and 8.32 MB output write dominate the runtime; the fix is BANDWIDTH_BOUND: treat this as an at-floor constant-fill case unless a broader backend change can use a lower-overhead memset/fill primitive or elide the allocation in a larger graph."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful.
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

OUT_SHAPE = (200, 200, 26)
OUT_STRIDE = (5200, 26, 1)
OUT_NUMEL = 200 * 200 * 26
BLOCK_SIZE = 1024
CUDA_DEVICE = torch.device("cuda", 0)


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fill_zero_f64_kernel(
        out_ptr,
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.full((BLOCK,), 0.0, tl.float64)
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs: list[object]) -> torch.Tensor:
    """Run the full Repro.forward scope with a Triton constant-fill kernel."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("Triton is required for the timed oracle")

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=CUDA_DEVICE,
        dtype=torch.float64,
    )
    grid = (triton.cdiv(OUT_NUMEL, BLOCK_SIZE),)
    _fill_zero_f64_kernel[grid](
        out,
        n_elements=OUT_NUMEL,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return out


def _check_layout(out: torch.Tensor) -> bool:
    return (
        tuple(out.shape) == OUT_SHAPE
        and out.stride() == OUT_STRIDE
        and out.dtype == torch.float64
        and out.device.type == "cuda"
        and out.device.index == 0
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
            f"dtype={layout_out.dtype} device={layout_out.device})"
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
