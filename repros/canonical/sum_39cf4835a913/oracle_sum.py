"""
Oracle for sum_39cf4835a913

Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full
`torch.sum(mm, dim=0)` scope with one shape-specialized Triton column-reduction
kernel that reads the contiguous `f32[M, N]` input once and writes the final
contiguous `f32[N]` output directly, whereas Inductor already lowers this tiny
single-reduction graph to one compiled reduction kernel in the same launch and
memory-traffic regime; Inductor cannot materially do less work today through
scheduler fusion, scatter-reduce, cooperative split-K, algebraic elimination,
or recompute fusion because there is no producer/consumer graph around the
reduction and the required input read, f32 accumulation, output store, and one
GPU launch dominate; the fix is BANDWIDTH_BOUND: treat this as a diagnosis
artifact unless the full-scope Triton kernel beats the required tuned compile
configs on the same shape.
"""
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
CLASSIFICATION = "BANDWIDTH_BOUND"


from oracle_harness import (  # noqa: E402
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info as _harness_get_hardware_info,
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


def _get_hardware_info() -> dict[str, float | int | str | None]:
    try:
        return _harness_get_hardware_info()
    except AttributeError:
        props = torch.cuda.get_device_properties(0)
        return {
            "name": props.name,
            "sm_major": props.major,
            "sm_minor": props.minor,
            "num_sms": props.multi_processor_count,
            "shared_mem_per_sm": getattr(
                props,
                "max_shared_memory_per_multiprocessor",
                getattr(props, "shared_memory_per_multiprocessor", None),
            ),
            "total_mem_gb": props.total_memory / 1e9,
        }


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

        print(json.dumps(_get_hardware_info(), indent=2))
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
