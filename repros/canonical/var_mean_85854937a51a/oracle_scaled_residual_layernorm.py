"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DINOv2 inference broadcast-scale residual LayerNorm scope in one row-wise Triton kernel, including the `[175360, 768]` to `[128, 1370, 768]` view, `add_77 + addmm * arg165_1` producer, population `var_mean(correction=0, keepdim=True)` over hidden size 768, eps=1e-6 rsqrt, affine epilogue, and final contiguous `[175360, 768]` view, whereas Inductor currently lowers the decomposed view/mul/add/var_mean/affine/view chain through generic normalization scheduling that does not form this producer-plus-epilogue row plan; Inductor cannot do this today because the scheduler/codegen norm template does not fold broadcast hidden-vector scale plus residual producers into the fixed-hidden LayerNorm lowering while preserving direct view-compatible output stores; the fix is SCHEDULER_FUSION: teach Inductor's LayerNorm template to fuse broadcast scale/residual producers into a fixed-hidden normalization epilogue with direct final-layout stores."""
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

BATCH = 128
TOKENS = 1370
HIDDEN = 768
ROWS = BATCH * TOKENS
ADDMM_SHAPE = (ROWS, HIDDEN)
BASE_SHAPE = (BATCH, TOKENS, HIDDEN)
AFFINE_SHAPE = (HIDDEN,)
OUTPUT_SHAPE = (ROWS, HIDDEN)
BASE_STRIDE = (TOKENS * HIDDEN, HIDDEN, 1)
OUTPUT_STRIDE = (HIDDEN, 1)
EPS = 1.0e-6
DTYPE = torch.float32
BLOCK_H = 1024
ROW_BLOCK = 1
CLASSIFICATION = "SCHEDULER_FUSION"

if triton is not None:

    @triton.jit
    def _scaled_residual_layernorm_kernel(
        addmm_ptr,
        scale_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask
        offsets = rows * hidden + cols

        addmm = tl.load(addmm_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        scaled = addmm * scale
        x = residual + scaled
        x_for_reduce = tl.where(mask, x, 0.0)

        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        square = tl.where(mask, centered * centered, 0.0)
        variance = tl.sum(square, axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + offsets, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int, int],
    tuple[int, int],
]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects 7 inputs, got {len(inputs)}")

    addmm_45, arg165_1, add_77, arg166_1, arg167_1, base_shape, output_shape = inputs
    tensor_inputs = (addmm_45, arg165_1, add_77, arg166_1, arg167_1)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")

    expected_shapes = (ADDMM_SHAPE, AFFINE_SHAPE, BASE_SHAPE, AFFINE_SHAPE, AFFINE_SHAPE)
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")
        if value.dtype != DTYPE:
            raise TypeError(f"input {index} dtype {value.dtype} != {DTYPE}")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if value.device != addmm_45.device:
            raise ValueError(f"input {index} device {value.device} != {addmm_45.device}")
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    base_shape_tuple = _shape_tuple(base_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if base_shape_tuple != BASE_SHAPE:
        raise ValueError(f"unexpected first view shape parameter: {base_shape!r}")
    if output_shape_tuple != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {output_shape!r}")

    return addmm_45, arg165_1, add_77, arg166_1, arg167_1, base_shape_tuple, output_shape_tuple


@oracle_impl(hardware="H100", shapes="(T([175360, 768], f32), T([768], f32), T([128, 1370, 768], f32), T([768], f32), T([768], f32), S([128, 1370, 768]), S([175360, 768]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward broadcast-scale residual LayerNorm computation.

    SCOPE INVARIANT: accepts the same 7 inputs as Repro.forward() and returns
    the same single float32 `[175360, 768]` contiguous view output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_scaled_residual_layernorm.py")

    addmm_45, arg165_1, add_77, arg166_1, arg167_1, base_shape, output_shape = _validate_inputs(inputs)
    base = torch.empty_strided(
        base_shape,
        BASE_STRIDE,
        device=addmm_45.device,
        dtype=DTYPE,
    )
    _scaled_residual_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        addmm_45,
        arg165_1,
        add_77,
        arg166_1,
        arg167_1,
        base,
        total_rows=ROWS,
        hidden=HIDDEN,
        eps=EPS,
        block_h=BLOCK_H,
        row_block=ROW_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return base.view(output_shape)


def _check_layout(instance: torch.nn.Module, inputs: tuple[Any, ...] | list[Any]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()
    ok = (
        isinstance(expected, torch.Tensor)
        and isinstance(actual, torch.Tensor)
        and tuple(actual.shape) == tuple(expected.shape)
        and actual.stride() == expected.stride()
        and actual.dtype == expected.dtype
    )
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected_stride={expected.stride()}, oracle_stride={actual.stride()})"
    )
    return ok


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
        ok = _check_layout(instance, inputs) and ok
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
