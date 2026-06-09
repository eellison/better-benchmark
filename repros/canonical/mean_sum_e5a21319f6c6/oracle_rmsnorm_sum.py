"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RMSNorm-affine-then-global-sum repro scope by keeping each row's mean-square, rsqrt, affine multiply, bf16 cast, and scalar-sum contribution inside a small set of Triton reductions, whereas Inductor currently lowers the decomposed mean/rsqrt/mul/cast/sum graph through generic norm/reduction scheduling and cannot directly emit a scalar-producing RMSNorm epilogue template; Inductor cannot do this today because the normalization scheduler does not fuse a row-wise RMSNorm template with a dependent whole-tensor reduction over the bf16-rounded affine output; the fix is SCHEDULER_FUSION: teach Inductor's norm-template lowering to sink scalar sum consumers into the RMSNorm epilogue while preserving the required bf16 rounding semantics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
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
    get_shape_key,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

EPS = 1.0e-6
REDUCE_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _rmsnorm_sum_partials_kernel(
        x_ptr,
        weight_ptr,
        partials_ptr,
        rows_total: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK_H)
        row_mask = rows < rows_total
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]
        offsets = rows[:, None] * hidden + cols[None, :]

        x = tl.load(
            x_ptr + offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        square_sum = tl.sum(tl.where(mask, x * x, 0.0), axis=1)
        inv_rms = tl.rsqrt(square_sum * (1.0 / hidden) + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=col_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        affine = x * inv_rms[:, None] * weight[None, :]
        rounded = affine.to(tl.bfloat16).to(tl.float32)
        row_sums = tl.sum(tl.where(mask, rounded, 0.0), axis=1)
        partial = tl.sum(tl.where(row_mask, row_sums, 0.0), axis=0)
        tl.store(partials_ptr + tl.program_id(0), partial)

    @triton.jit
    def _sum_partials_kernel(
        partials_ptr,
        partials2_ptr,
        n_partials: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < n_partials
        values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(partials2_ptr + tile, total)

    @triton.jit
    def _final_sum_kernel(
        partials_ptr,
        out_ptr,
        n_partials: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < n_partials
        values = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(values, axis=0)
        tl.store(out_ptr, total)


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={value.stride()}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if len(inputs) != 2:
        raise ValueError(f"{REPRO_ID} expects 2 inputs, got {len(inputs)}")

    x = _require_tensor("arg0_1", inputs[0], None, torch.bfloat16)
    if x.ndim != 2:
        raise ValueError(f"arg0_1 must be rank-2, got shape={tuple(x.shape)}")
    rows, hidden = (int(x.shape[0]), int(x.shape[1]))
    weight = _require_tensor("arg1_1", inputs[1], (hidden,), torch.float32)
    if weight.device != x.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")
    if hidden <= 0:
        raise ValueError("hidden dimension must be positive")
    if hidden > 16384:
        raise ValueError(f"hidden dimension {hidden} exceeds this oracle's BLOCK_H limit")
    return x, weight, rows, hidden


def _block_m_for_hidden(hidden: int) -> int:
    if hidden <= 1024:
        return 8
    return 1


@oracle_impl(hardware="H100", shapes="(T([1152000, 512], bf16), T([512], f32))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same scalar bf16 sum as the captured RMSNorm-affine repro.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_rmsnorm_sum.py")

    x, weight, rows, hidden = _validate_inputs(inputs)
    block_m = _block_m_for_hidden(hidden)
    block_h = triton.next_power_of_2(hidden)
    n_partials = triton.cdiv(rows, block_m)
    n_partials2 = triton.cdiv(n_partials, REDUCE_BLOCK)

    partials = torch.empty((n_partials,), device=x.device, dtype=torch.float32)
    partials2 = torch.empty((n_partials2,), device=x.device, dtype=torch.float32)
    out = torch.empty((), device=x.device, dtype=torch.bfloat16)

    _rmsnorm_sum_partials_kernel[(n_partials,)](
        x,
        weight,
        partials,
        rows_total=rows,
        hidden=hidden,
        eps=EPS,
        BLOCK_M=block_m,
        BLOCK_H=block_h,
        num_warps=4 if hidden <= 1024 else 8,
    )
    _sum_partials_kernel[(n_partials2,)](
        partials,
        partials2,
        n_partials=n_partials,
        BLOCK_N=REDUCE_BLOCK,
        num_warps=4,
    )
    _final_sum_kernel[(1,)](
        partials2,
        out,
        n_partials=n_partials2,
        BLOCK_N=triton.next_power_of_2(n_partials2),
        num_warps=4,
    )
    return out


# --- CLI entry point ---
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
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
