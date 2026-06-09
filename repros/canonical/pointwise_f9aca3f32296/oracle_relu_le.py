"""
Oracle for pointwise_f9aca3f32296.

Gap diagnosis: Classification: BANDWIDTH_BOUND. The oracle computes the full
graph with one Triton timed kernel and replaces relu(addmm_5) <= 0 with
addmm_5 <= 0, which is equivalent for this f32 input including normal
signed-zero and NaN comparison behavior, but the measured runtime is still at
the same launch/allocation-scale floor as the compiled Inductor pointwise
kernel. Inductor already emits a fused tiny pointwise kernel, and the remaining
runtime is dominated by fresh bool output allocation plus one CUDA launch rather
than the eliminated relu arithmetic.
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

N_ELEMS = 2048
BLOCK = 1024


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _le_zero_kernel(x_ptr, out_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=1.0)
        out = x <= 0.0
        tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([2048, 1], f32))")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with a simplified Triton pointwise kernel."""
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (addmm_5,) = inputs
    if not isinstance(addmm_5, torch.Tensor):
        raise TypeError(f"{REPRO_ID} input must be a tensor")
    if addmm_5.shape != (2048, 1) or addmm_5.dtype is not torch.float32:
        raise ValueError(
            f"{REPRO_ID} expects f32[2048, 1], got {addmm_5.dtype} {tuple(addmm_5.shape)}"
        )

    out = torch.empty_like(addmm_5, dtype=torch.bool)
    grid = (triton.cdiv(N_ELEMS, BLOCK),)
    _le_zero_kernel[grid](addmm_5, out, n_elements=N_ELEMS, BLOCK_SIZE=BLOCK, num_warps=4)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--rep", type=int, default=200, help="Benchmark repetitions")
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark all shapes.txt shapes")
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
