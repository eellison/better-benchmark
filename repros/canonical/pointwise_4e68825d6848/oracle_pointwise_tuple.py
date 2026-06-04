"""
Oracle for pointwise_4e68825d6848

Gap diagnosis:
  Classification: BANDWIDTH_BOUND
  What oracle does differently: The oracle returns the first split column as the
    same non-contiguous input view and computes only the second-column
    tanh/add/mul/add/exp chain in one tiny Triton pointwise kernel.
  Why Inductor cannot do it today: Inductor already lowers this full scope to
    launch-floor pointwise work, and the remaining cost is dominated by one
    small allocation plus one CUDA launch for 256 float elements.
  Required Inductor change: BANDWIDTH_BOUND; no scheduler-fusion or pattern
    change should be claimed unless a full-scope oracle beats the required
    compiled configurations by more than launch-noise.
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
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"

N_ROWS = 256
N_COLS = 2
BLOCK_N = 256
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[object]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR).eval()


if triton is not None:

    @triton.jit
    def _second_column_exp_kernel(
        in_ptr,
        out_ptr,
        n_rows: tl.constexpr,
        stride_row: tl.constexpr,
        stride_col: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_rows
        second_col = tl.load(in_ptr + offsets * stride_row + stride_col, mask=mask)
        tanh_approx = (2.0 / (1.0 + tl.exp(-2.0 * second_col))) - 1.0
        values = tl.exp(6.0 * (tanh_approx + 1.0) - 10.0)
        tl.store(out_ptr + offsets, values, mask=mask)


def oracle_forward(inputs: list[object] | tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the full tuple-returning pointwise repro scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_pointwise_tuple.py")
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (addmm_2,) = inputs
    if not isinstance(addmm_2, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(addmm_2)!r}")
    if tuple(addmm_2.shape) != (N_ROWS, N_COLS):
        raise ValueError(f"unexpected input shape: {tuple(addmm_2.shape)}")
    if addmm_2.dtype is not torch.float32:
        raise ValueError(f"unexpected input dtype: {addmm_2.dtype}")
    if not addmm_2.is_cuda:
        raise ValueError("oracle_pointwise_tuple.py expects CUDA inputs")

    first_col = addmm_2[:, :1]
    out = torch.empty_strided((N_ROWS, 1), (1, 1), device=addmm_2.device, dtype=addmm_2.dtype)
    _second_column_exp_kernel[(1,)](
        addmm_2,
        out,
        n_rows=N_ROWS,
        stride_row=addmm_2.stride(0),
        stride_col=addmm_2.stride(1),
        BLOCK_SIZE=BLOCK_N,
        num_warps=1,
    )
    return first_col, out


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
