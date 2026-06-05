"""Gap diagnosis (classification: RECOMPUTE_FUSION): this oracle computes the complete residual-add RMSNorm alias scope in one Triton row kernel, including the metadata-only input view, dtype-rounded residual add, fp32 mean(square) over the hidden dimension, eps=1e-6 rsqrt, fp16-rounded normalization, dtype-promoted weight multiply, and both aliasing output views over one final buffer, whereas Inductor lowers the same fused reduction as a generic two-pass row schedule that reloads the activation/residual inputs and recomputes the add for the epilogue after accumulating the mean-square; Inductor cannot do this today because its reduction scheduler does not keep the full hidden tile's producer values live across the row reduction for the normalization epilogue; the fix is RECOMPUTE_FUSION: add a guarded RMSNorm row schedule/template that retains the residual-add tile through the reduction and emits the affine epilogue without the second input pass."""
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

EPS = 1.0e-6
OUTPUT_COUNT = 2

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
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
    def _residual_rmsnorm_aliases_kernel(
        mm_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        is_bf16: tl.constexpr,
        output_is_f32: tl.constexpr,
        block_h: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, block_h)
        mask = cols < hidden
        offsets = row * hidden + cols

        mm = tl.load(mm_ptr + offsets, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
        added = mm + residual
        if is_bf16:
            added_rounded = added.to(tl.bfloat16)
        else:
            added_rounded = added.to(tl.float16)
        x = added_rounded.to(tl.float32)

        sum_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0)
        inv_rms = tl.rsqrt(sum_sq / hidden + eps)
        normalized_f16 = (x * inv_rms).to(tl.float16)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0)
        if output_is_f32:
            out = weight.to(tl.float32) * normalized_f16.to(tl.float32)
        else:
            out = weight * normalized_f16
        tl.store(out_ptr + offsets, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in value)


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mm_220, add_219, arg287_1, view_shape_arg, *output_shape_args = inputs
    tensor_inputs = (mm_220, add_219, arg287_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first three repro inputs must be tensors")

    if mm_220.ndim != 2:
        raise ValueError(f"mm input must be rank-2, got shape={tuple(mm_220.shape)}")
    rows, hidden = (int(mm_220.shape[0]), int(mm_220.shape[1]))

    view_shape = _shape_tuple(view_shape_arg)
    output_shapes = tuple(_shape_tuple(shape) for shape in output_shape_args)
    if len(output_shapes) != OUTPUT_COUNT:
        raise ValueError(f"expected {OUTPUT_COUNT} output shape parameters, got {len(output_shapes)}")

    if tuple(add_219.shape) != view_shape:
        raise ValueError(f"residual shape {tuple(add_219.shape)} != view shape {view_shape}")
    if tuple(arg287_1.shape) != (hidden,):
        raise ValueError(f"weight shape {tuple(arg287_1.shape)} != {(hidden,)}")
    if view_shape[-1:] != (hidden,):
        raise ValueError(f"view shape {view_shape} must keep hidden dimension {hidden}")
    if rows * hidden != add_219.numel():
        raise ValueError("mm view shape must have the same number of elements as residual")

    for index, shape in enumerate(output_shapes):
        if shape != (rows, hidden):
            raise ValueError(f"unexpected output shape parameter {index}: {shape!r}")

    expected_dtype = mm_220.dtype
    if expected_dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"mm input dtype {expected_dtype} is not supported")
    for index, value in enumerate(tensor_inputs):
        if value.dtype != expected_dtype:
            raise TypeError(f"input {index} dtype {value.dtype} != {expected_dtype}")
        if value.device != mm_220.device:
            raise ValueError("all tensor inputs must be on the same device")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for this Triton oracle")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    return mm_220, add_219, arg287_1, view_shape, output_shapes


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same two output views. The returned views alias one contiguous
    [B, M, H]-style base buffer, matching eager view alias/layout behavior.

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_residual_rmsnorm_aliases.py")

    mm_220, add_219, arg287_1, view_shape, output_shapes = _validate_inputs(inputs)
    rows, hidden = (int(mm_220.shape[0]), int(mm_220.shape[1]))
    output_dtype = torch.float32 if mm_220.dtype == torch.bfloat16 else torch.float16
    base = torch.empty_strided(
        view_shape,
        _contiguous_stride(view_shape),
        device=mm_220.device,
        dtype=output_dtype,
    )
    block_h = triton.next_power_of_2(hidden)
    num_warps = 8 if hidden >= 2048 else 4

    _residual_rmsnorm_aliases_kernel[(rows,)](
        mm_220,
        add_219,
        arg287_1,
        base,
        hidden=hidden,
        eps=EPS,
        is_bf16=mm_220.dtype == torch.bfloat16,
        output_is_f32=output_dtype == torch.float32,
        block_h=block_h,
        num_warps=num_warps,
        num_stages=3,
    )
    return tuple(base.view(shape) for shape in output_shapes)


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
