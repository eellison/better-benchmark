"""
Oracle for pointwise_41aa929b1bf2

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: The oracle runs the full computation as a
    single minimal Triton fill kernel that allocates the required int64[32]
    output and stores zeros directly.
  Why Inductor cannot do it today: Inductor already lowers this full-scope
    constant tensor creation to the same essential shape of work, and the
    remaining cost is dominated by output allocation plus one CUDA launch.
  Required Inductor change: BANDWIDTH_BOUND; no scheduler/codegen optimization
    can materially beat this without changing user-visible fresh-tensor
    semantics or avoiding the launch outside this compiled region.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
BLOCK = 32


def get_inputs() -> tuple[object, ...]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create Repro() for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _fill_zero_i64_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        zeros = tl.full((BLOCK_SIZE,), 0, tl.int64)
        tl.store(out_ptr + offsets, zeros)


@oracle_impl(hardware="H100", shapes="()")
def oracle_forward(inputs: tuple[object, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope with a minimal Triton fill kernel."""
    if inputs:
        raise ValueError(f"{REPRO_ID} expects no inputs, got {len(inputs)}")
    if triton is None:
        raise RuntimeError("triton is required for this oracle")
    out = torch.empty((N,), device="cuda", dtype=torch.int64)
    _fill_zero_i64_kernel[(1,)](out, BLOCK_SIZE=BLOCK, num_warps=1)
    return out


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
