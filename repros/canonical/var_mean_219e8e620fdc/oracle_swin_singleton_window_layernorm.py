"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin hidden-size-1024 population LayerNorm affine scope, including the `[6272,1024] -> [128,7,7,1024]` view, correction=0 fp32 mean plus centered-variance, `var + 1e-5` before `libdevice.rsqrt`, affine epilogue, singleton window metadata views/permute/views, and final contiguous `[6272,1024]` output, whereas Inductor already emits one fused persistent row-reduction kernel for this same full scope but pays a small residual cost in the generic normalization/layout schedule; Inductor cannot pick this today because it has no guarded singleton-window LayerNorm template that erases the no-op window metadata path and uses the row-blocked schedule measured here while preserving exact Inductor numerics; the fix is NEW_PATTERN: add a shape/layout-specialized singleton-window LayerNorm lowering, or teach the existing normalization template to choose this row-blocked schedule for the same metadata-only window path."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)


ROWS = 6272
BATCH = 128
HEIGHT = 7
WIDTH = 7
HIDDEN = 1024
EPS = 1.0e-5
DTYPE = torch.float32
CLASSIFICATION = "NEW_PATTERN"

INPUT_SHAPE = (ROWS, HIDDEN)
WEIGHT_SHAPE = (HIDDEN,)
VIEW_SHAPE = (BATCH, HEIGHT, WIDTH, HIDDEN)
WINDOW_VIEW_SHAPE = (BATCH, 1, HEIGHT, 1, WIDTH, HIDDEN)
FLATTEN_VIEW_SHAPE = (-1, HEIGHT, WIDTH, HIDDEN)
TOKEN_VIEW_SHAPE = (-1, HEIGHT * WIDTH, HIDDEN)
OUTPUT_SHAPE = (ROWS, HIDDEN)
OUTPUT_STRIDE = (HIDDEN, 1)
BLOCK_H = HIDDEN


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:
    from torch._inductor.runtime.triton_helpers import libdevice

    @triton.autotune(
        configs=[
            triton.Config({"ROW_BLOCK": 1}, num_warps=1, num_stages=1),
            triton.Config({"ROW_BLOCK": 2}, num_warps=1, num_stages=1),
            triton.Config({"ROW_BLOCK": 4}, num_warps=1, num_stages=1),
            triton.Config({"ROW_BLOCK": 8}, num_warps=1, num_stages=1),
        ],
        key=["total_rows"],
    )
    @triton.jit
    def _swin_singleton_window_layernorm_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, BLOCK)[None, :]
        row_mask = rows < total_rows
        offsets = rows * hidden + cols

        x = tl.load(
            input_ptr + offsets,
            mask=row_mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            eviction_policy="evict_last",
        ).to(tl.float32)

        # Match Inductor's generated correction=0 lowering: fp32 mean, then
        # centered squared-difference population variance, then libdevice.rsqrt.
        x_for_mean = tl.where(row_mask, x, 0.0)
        sum_x = tl.sum(x_for_mean, axis=1)[:, None].to(tl.float32)
        mean = sum_x / tl.full((1, 1), 1024, tl.int32).to(tl.float32)
        centered_for_var = x - mean
        sum_centered_sq = tl.sum(
            tl.where(row_mask, centered_for_var * centered_for_var, 0.0),
            axis=1,
        )[:, None].to(tl.float32)
        variance = sum_centered_sq / tl.full((1, 1), 1024.0, tl.float32)
        invstd = libdevice.rsqrt(variance + tl.full((1, 1), eps, tl.float32))

        centered = x - mean
        output = (centered * invstd) * weight + bias
        tl.store(output_ptr + offsets, output, mask=row_mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected_shape}")
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {expected_dtype}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int, int],
    tuple[int, int, int, int, int, int],
    tuple[int, int, int, int],
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        mm_2,
        weight,
        bias,
        shape0_raw,
        shape1_raw,
        shape2_raw,
        shape3_raw,
        shape4_raw,
    ) = inputs

    mm_2 = _require_tensor("mm_2", mm_2, INPUT_SHAPE, DTYPE)
    weight = _require_tensor("arg333_1", weight, WEIGHT_SHAPE, DTYPE)
    bias = _require_tensor("arg334_1", bias, WEIGHT_SHAPE, DTYPE)
    if not (mm_2.device == weight.device == bias.device):
        raise ValueError("all tensor inputs must be on the same device")

    shape0 = _shape_tuple(shape0_raw)
    shape1 = _shape_tuple(shape1_raw)
    shape2 = _shape_tuple(shape2_raw)
    shape3 = _shape_tuple(shape3_raw)
    shape4 = _shape_tuple(shape4_raw)

    expected_shapes = (
        VIEW_SHAPE,
        WINDOW_VIEW_SHAPE,
        FLATTEN_VIEW_SHAPE,
        TOKEN_VIEW_SHAPE,
        OUTPUT_SHAPE,
    )
    actual_shapes = (shape0, shape1, shape2, shape3, shape4)
    for index, (actual, expected) in enumerate(zip(actual_shapes, expected_shapes), start=3):
        if actual != expected:
            raise ValueError(f"shape parameter {index} is {actual}, expected {expected}")

    return mm_2, weight, bias, shape0, shape1, shape2, shape3, shape4


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    mm_2, weight, bias, shape0, shape1, shape2, shape3, shape4 = _validate_inputs(inputs)
    x = torch.ops.aten.view.default(mm_2, shape0)
    variance, mean = torch.ops.aten.var_mean.correction(
        x,
        [3],
        correction=0,
        keepdim=True,
    )
    x = torch.ops.aten.sub.Tensor(x, mean)
    x = torch.ops.aten.mul.Tensor(
        x,
        torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS)),
    )
    x = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(x, weight), bias)
    x = torch.ops.aten.view.default(x, shape1)
    x = torch.ops.aten.permute.default(x, [0, 1, 3, 2, 4, 5])
    x = torch.ops.aten.view.default(x, shape2)
    x = torch.ops.aten.view.default(x, shape3)
    return torch.ops.aten.view.default(x, shape4)


def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single float32 contiguous `[6272, 1024]` output.
    """
    mm_2, weight, bias, _shape0, _shape1, _shape2, _shape3, shape4 = _validate_inputs(inputs)
    if triton is None or not mm_2.is_cuda:
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        shape4,
        OUTPUT_STRIDE,
        device=mm_2.device,
        dtype=DTYPE,
    )
    grid = lambda meta: (triton.cdiv(ROWS, meta["ROW_BLOCK"]),)
    _swin_singleton_window_layernorm_kernel[grid](
        mm_2,
        weight,
        bias,
        output,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        BLOCK=BLOCK_H,
    )
    return output


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
