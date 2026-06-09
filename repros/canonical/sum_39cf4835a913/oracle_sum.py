"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete `torch.sum(mm, dim=0)` Repro.forward scope as a shape-specialized Triton column reduction that reads the contiguous f32[M, N] input once and writes the final contiguous f32[N] output directly, whereas Inductor currently lowers the same small dim-0 reduction through its generic reduction codegen path; Inductor cannot do this today because scheduler/codegen has no dedicated small-static-column-reduction template for the fixed short row extent and contiguous output layout; the fix is NEW_PATTERN: add a fixed-extent dim-0 column-reduction template for this layout and reduction extent."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"
CLASSIFICATION = "NEW_PATTERN"


# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
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


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 32}, num_warps=8, num_stages=3),
        ],
        key=["M_", "N_"],
    )
    @triton.jit
    def _sum_dim0_kernel(
        x_ptr,
        out_ptr,
        M_: tl.constexpr,
        N_: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        cols = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rows = tl.arange(0, BLOCK_M)
        mask = (rows[:, None] < M_) & (cols[None, :] < N_)
        values = tl.load(
            x_ptr + rows[:, None] * N_ + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        totals = tl.sum(values, axis=0)
        tl.store(out_ptr + cols, totals, mask=cols < N_)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    if len(inputs) != 1:
        raise ValueError(f"{REPRO_ID} expects one input, got {len(inputs)}")

    (mm,) = inputs
    if not isinstance(mm, torch.Tensor):
        raise TypeError(f"expected tensor input, got {type(mm)!r}")
    if mm.ndim != 2:
        raise ValueError(f"expected rank-2 input, got shape={tuple(mm.shape)}")
    if mm.dtype is not torch.float32:
        raise TypeError(f"expected float32 input, got {mm.dtype}")
    if not mm.is_cuda:
        raise ValueError("oracle_sum.py expects CUDA inputs")
    if not mm.is_contiguous():
        raise ValueError(f"expected contiguous input, got stride={mm.stride()}")
    return mm


@oracle_impl(hardware="H100", shapes="(T([128, 768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward column reduction."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_sum.py")

    mm = _validate_inputs(inputs)
    m, n = mm.shape
    out = torch.empty_strided((n,), (1,), device=mm.device, dtype=mm.dtype)
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_N"]),)
    _sum_dim0_kernel[grid](
        mm,
        out,
        M_=m,
        N_=n,
        BLOCK_M=triton.next_power_of_2(m),
    )
    return out


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
                    print(f"diagnosis_only: required comparison shows not_true_floor "
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
