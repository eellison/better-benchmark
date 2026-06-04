"""
Oracle for pointwise_4e4da02d582f

Gap diagnosis:
  Classification: SCHEDULER_FUSION
  What oracle does differently: The oracle preserves the first split result as a view of the input and computes both dependent pointwise outputs from the second column in one Triton kernel.
  What Inductor change would fix: Inductor needs scheduler support for treating tuple returns that mix aliasing views and fused materialized pointwise outputs as one fused scheduling unit.
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
REPO_ROOT = REPRO_DIR.parents[2]
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


ROWS = 256
COLS = 2


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _pointwise_tuple_kernel(
        addmm_2,
        exp_out,
        grad_out,
        N: tl.constexpr,
        INPUT_ROW_STRIDE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < N
        second = tl.load(addmm_2 + offsets * INPUT_ROW_STRIDE + 1, mask=mask, other=0.0)
        tanh_second = 2.0 * tl.sigmoid(2.0 * second) - 1.0
        exp_val = tl.exp(6.0 * (tanh_second + 1.0) - 10.0)
        grad_val = 1.0 - tanh_second * tanh_second
        tl.store(exp_out + offsets, exp_val, mask=mask)
        tl.store(grad_out + offsets, grad_val, mask=mask)


def _validate_input(addmm_2: torch.Tensor) -> None:
    if addmm_2.ndim != 2 or tuple(addmm_2.shape) != (ROWS, COLS):
        raise ValueError(f"expected addmm_2 shape {(ROWS, COLS)}, got {tuple(addmm_2.shape)}")
    if addmm_2.dtype is not torch.float32:
        raise ValueError(f"expected addmm_2 dtype torch.float32, got {addmm_2.dtype}")
    if tuple(addmm_2.stride()) != (COLS, 1):
        raise ValueError(f"expected contiguous addmm_2 stride {(COLS, 1)}, got {tuple(addmm_2.stride())}")


def oracle_forward(inputs):
    """Run the full-scope oracle on the exact Repro.forward input tuple."""
    (addmm_2,) = inputs
    _validate_input(addmm_2)

    first_col = addmm_2.as_strided((ROWS, 1), (COLS, 1), 0)
    exp_out = torch.empty_strided((ROWS, 1), (1, 1), dtype=addmm_2.dtype, device=addmm_2.device)
    grad_out = torch.empty_strided((ROWS, 1), (1, 1), dtype=addmm_2.dtype, device=addmm_2.device)

    if addmm_2.is_cuda:
        if triton is None:
            raise RuntimeError("Triton is required for the CUDA oracle")
        _pointwise_tuple_kernel[(1,)](
            addmm_2,
            exp_out,
            grad_out,
            N=ROWS,
            INPUT_ROW_STRIDE=COLS,
            BLOCK_M=triton.next_power_of_2(ROWS),
        )
    else:
        second = addmm_2[:, 1:2]
        tanh_second = torch.tanh(second)
        exp_out.copy_(torch.exp(6.0 * (tanh_second + 1.0) - 10.0))
        grad_out.copy_(1.0 - tanh_second * tanh_second)

    return first_col, exp_out, grad_out


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
