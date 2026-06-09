"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full captured T5 inference RMSNorm path, including fp16 residual add, global any(isinf) clamp-bound selection, clamp, fp16 cast, fp32 mean(square), rsqrt eps=1e-6, fp16 weight multiply, and twelve aliasing [2048,512] view outputs over one final buffer, whereas Inductor still schedules this as generic scalar-predicate reduction plus row-normalization work without a semantic multi-alias T5 RMSNorm lowering; Inductor cannot do this today because normalization templates do not recognize the clamp-bound RMSNorm pattern with many metadata-only sibling view returns; the fix is NEW_PATTERN: add a T5 RMSNorm template that preserves the global finite-bound clamp choice, row RMSNorm math, fp16 weight epilogue, and arbitrary-count alias-view output contract."""
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

ROWS = 2048
HIDDEN = 512
INPUT_SHAPE = (ROWS, HIDDEN)
RESIDUAL_SHAPE = (1, ROWS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (ROWS * HIDDEN, HIDDEN, 1)
OUTPUT_COUNT = 12
TOTAL_ELEMENTS = ROWS * HIDDEN
BLOCK_ELEMS = 1024
BLOCK_H = 512
EPS = 1.0e-6
FINITE_BOUND = 65504.0
INF_BOUND = 64504.0

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

if triton is not None:

    @triton.jit
    def _clear_flag_kernel(flag_ptr):
        tl.store(flag_ptr, 0)

    @triton.jit
    def _any_inf_kernel(
        mm_ptr,
        residual_ptr,
        flag_ptr,
        n_elements: tl.constexpr,
        block_elems: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_elems + tl.arange(0, block_elems)
        mask = offsets < n_elements

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        add_f16 = (mm + residual).to(tl.float16)
        block_has_inf = tl.sum(
            tl.where((tl.abs(add_f16) == float("inf")) & mask, 1, 0),
            axis=0,
        )
        tl.atomic_or(flag_ptr, tl.where(block_has_inf > 0, 1, 0), sem="relaxed")

    @triton.jit
    def _t5_rmsnorm_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        flag_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        finite_bound: tl.constexpr,
        inf_bound: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        # aten.add on two fp16 tensors rounds before the explicit fp32 convert.
        add_f16 = (mm + residual).to(tl.float16)
        add_f32 = add_f16.to(tl.float32)

        has_inf = tl.load(flag_ptr) != 0
        bound = tl.where(has_inf, inf_bound, finite_bound)
        clamped_min = tl.where(add_f32 < -bound, -bound, add_f32)
        clamped = tl.where(clamped_min > bound, bound, clamped_min)
        clamped = tl.where(add_f32 != add_f32, add_f32, clamped)
        x = clamped.to(tl.float16).to(tl.float32)

        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / hidden + eps)

        norm_f16 = (x * inv_rms).to(tl.float16)
        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        out = (weight * norm_f16).to(tl.float16)
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]:
    if len(inputs) != 16:
        raise ValueError(f"{REPRO_ID} expects 16 inputs, got {len(inputs)}")

    mm_40, residual, weight, shape0, *output_shapes = inputs
    tensor_inputs = (mm_40, residual, weight)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    expected_shapes = (INPUT_SHAPE, RESIDUAL_SHAPE, WEIGHT_SHAPE)
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != torch.float16:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float16")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if residual.device != mm_40.device or weight.device != mm_40.device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(shape0) != RESIDUAL_SHAPE:
        raise ValueError(f"unexpected residual view shape parameter: {shape0!r}")
    if len(output_shapes) != OUTPUT_COUNT:
        raise ValueError(f"expected {OUTPUT_COUNT} output shape parameters, got {len(output_shapes)}")

    view_shapes = tuple(_shape_tuple(shape) for shape in output_shapes)
    for index, shape in enumerate(view_shapes, start=1):
        if shape != OUTPUT_SHAPE:
            raise ValueError(f"unexpected output shape parameter {index}: {shape!r}")

    return mm_40, residual, weight, view_shapes


@oracle_impl(hardware="H100", shapes="(T([2048, 512], f16), T([1, 2048, 512], f16), T([512], f16), S([1, 2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]), S([2048, 512]))")
def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward T5 RMSNorm computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same twelve fp16 [2048,512] view outputs. The returned views alias one
    [1,2048,512] base buffer, matching eager view alias/layout behavior.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_t5_rmsnorm_aliases.py")

    mm_40, residual, weight, view_shapes = _validate_inputs(inputs)
    has_inf = torch.empty((1,), device=mm_40.device, dtype=torch.int32)
    base = torch.empty_strided(
        RESIDUAL_SHAPE,
        BASE_STRIDE,
        device=mm_40.device,
        dtype=torch.float16,
    )

    _clear_flag_kernel[(1,)](has_inf)
    _any_inf_kernel[(triton.cdiv(TOTAL_ELEMENTS, BLOCK_ELEMS),)](
        mm_40,
        residual,
        has_inf,
        n_elements=TOTAL_ELEMENTS,
        block_elems=BLOCK_ELEMS,
    )
    _t5_rmsnorm_kernel[(ROWS,)](
        mm_40,
        residual,
        weight,
        has_inf,
        base,
        hidden=HIDDEN,
        eps=EPS,
        finite_bound=FINITE_BOUND,
        inf_bound=INF_BOUND,
        block_h=BLOCK_H,
        num_warps=4,
        num_stages=3,
    )
    return tuple(base.view(shape) for shape in view_shapes)


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
