"""
Oracle for sum_d7cc0afd5743

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: uses one specialized Triton reduction kernel for the full f32[1000, 1] input and stores the final f32[1] viewed output directly.
  What Inductor change would fix: no lowering change is indicated if the required tuned Inductor variants match or beat this one-launch memory-bound reduction floor.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"


from oracle_harness import (  # noqa: E402
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
    def _sum_1000_to_1_kernel(
        x_ptr,
        out_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        total = tl.sum(values, axis=0)
        tl.store(out_ptr, total)


def oracle_forward(inputs):
    """Run the full-scope oracle computation.

    The repro computes aten.sum(arg, [0], keepdim=True) over f32[1000, 1] and
    then views the f32[1, 1] result as f32[1]. The oracle returns that final
    viewed output shape directly.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum_reduce.py")

    (arg9_1,) = inputs
    if not isinstance(arg9_1, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(arg9_1)!r}")
    if tuple(arg9_1.shape) != (1000, 1):
        raise ValueError(f"unexpected input shape: {tuple(arg9_1.shape)}")
    if arg9_1.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {arg9_1.dtype}")
    if not arg9_1.is_cuda:
        raise ValueError("oracle_sum_reduce.py expects CUDA inputs")
    if not arg9_1.is_contiguous():
        arg9_1 = arg9_1.contiguous()

    out = torch.empty_strided((1,), (1,), device=arg9_1.device, dtype=arg9_1.dtype)
    _sum_1000_to_1_kernel[(1,)](
        arg9_1,
        out,
        N=arg9_1.numel(),
        BLOCK_N=triton.next_power_of_2(arg9_1.numel()),
    )
    return out


def main():
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
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward, REPRO_DIR, REPRO_ID,
                warmup=args.warmup, rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"diagnosis_only: required comparison shows not_true_floor "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
