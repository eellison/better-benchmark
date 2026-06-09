"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete T5 residual-add RMSNorm scope, including the global any(isinf(add)) bound switch, fp32 clamp, fp16 round-trip, fp32 mean-square reduction, rsqrt, fp16 weight multiply, and final [2048,H] view, whereas tuned Inductor must perform the same global predicate plus fixed-width row normalization work; Inductor cannot materially improve this today through local fusion because the clamp bound is a true whole-tensor dependency and the remaining row kernel is dominated by required memory traffic, launch sequencing, and small-K reduction latency; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template variant unless broader launch-overhead or normalization-template changes move the family."""
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

ROWS = 2048
EPS = 1.0e-6
ANY_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _any_inf_tiles_kernel(
        mm_ptr,
        residual_ptr,
        partial_flags_ptr,
        total_elements: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < total_elements

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        add_f16 = (mm + residual).to(tl.float16)
        is_inf = (add_f16 == float("inf")) | (add_f16 == -float("inf"))
        any_inf = tl.sum(tl.where(mask & is_inf, 1, 0), axis=0) != 0
        tl.store(partial_flags_ptr + tile, any_inf.to(tl.int8))

    @triton.jit
    def _finalize_any_inf_kernel(
        partial_flags_ptr,
        any_inf_ptr,
        num_tiles: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        tiles = tl.arange(0, BLOCK_TILES)
        mask = tiles < num_tiles
        flags = tl.load(partial_flags_ptr + tiles, mask=mask, other=0).to(tl.int32)
        any_inf = tl.sum(flags, axis=0) != 0
        tl.store(any_inf_ptr, any_inf.to(tl.int8))

    @triton.jit
    def _rmsnorm_row_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        any_inf_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_H)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        add_f16 = (mm + residual).to(tl.float16)
        add_f32 = add_f16.to(tl.float32)

        has_inf = tl.load(any_inf_ptr).to(tl.int32) != 0
        bound = tl.where(has_inf, 64504.0, 65504.0)
        clamped = tl.minimum(tl.maximum(add_f32, -bound), bound)
        clamped_f16 = clamped.to(tl.float16)
        norm_input = clamped_f16.to(tl.float32)

        square_sum = tl.sum(
            tl.where(mask, norm_input * norm_input, 0.0),
            axis=0,
        )
        inv_rms = tl.rsqrt(square_sum / hidden + eps)
        normalized_f16 = (norm_input * inv_rms).to(tl.float16)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = (weight * normalized_f16).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mm, residual, weight, shape0, shape1 = inputs
    if not isinstance(mm, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise TypeError("first two repro inputs must be tensors")
    if not isinstance(weight, torch.Tensor):
        raise TypeError("third repro input must be a tensor")

    if mm.ndim != 2:
        raise ValueError(f"mm input must be rank-2, got shape={tuple(mm.shape)}")
    rows, hidden = (int(mm.shape[0]), int(mm.shape[1]))
    if rows != ROWS:
        raise ValueError(f"unexpected row count: got={rows} expected={ROWS}")

    expected_residual_shape = (1, rows, hidden)
    expected_weight_shape = (hidden,)
    if tuple(residual.shape) != expected_residual_shape:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} != {expected_residual_shape}"
        )
    if tuple(weight.shape) != expected_weight_shape:
        raise ValueError(f"weight shape {tuple(weight.shape)} != {expected_weight_shape}")

    for index, value in enumerate((mm, residual, weight)):
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if tuple(int(dim) for dim in shape0) != expected_residual_shape:
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if tuple(int(dim) for dim in shape1) != (rows, hidden):
        raise ValueError(f"unexpected output view shape parameter: {shape1!r}")

    return mm, residual, weight, rows, hidden


@oracle_impl(hardware="H100", shapes="(T([2048, 512], f16), T([1, 2048, 512], f16), T([512], f16), S([1, 2048, 512]), S([2048, 512]))")
def oracle_forward(inputs):
    """Run the full residual-add RMSNorm repro computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_rmsnorm.py")

    mm, residual, weight, rows, hidden = _validate_inputs(inputs)
    total_elements = rows * hidden
    num_tiles = triton.cdiv(total_elements, ANY_BLOCK)
    partial_flags = torch.empty((num_tiles,), device=mm.device, dtype=torch.int8)
    any_inf = torch.empty((), device=mm.device, dtype=torch.int8)
    out = torch.empty_strided((rows, hidden), (hidden, 1), device=mm.device, dtype=torch.float16)

    _any_inf_tiles_kernel[(num_tiles,)](
        mm,
        residual,
        partial_flags,
        total_elements=total_elements,
        BLOCK_N=ANY_BLOCK,
        num_warps=4,
    )
    _finalize_any_inf_kernel[(1,)](
        partial_flags,
        any_inf,
        num_tiles=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=8,
    )
    _rmsnorm_row_kernel[(rows,)](
        mm,
        residual,
        weight,
        any_inf,
        out,
        hidden=hidden,
        eps=EPS,
        BLOCK_H=triton.next_power_of_2(hidden),
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
