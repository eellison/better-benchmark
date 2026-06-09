"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ViT class-token plus patch-token positional LayerNorm scope in one Triton row kernel, including expand/view/permute/cat semantics, position add, population var_mean, affine epilogue, and final contiguous flattened output, whereas tuned Inductor already lowers this fixed-width norm-template region into a faster one-kernel memory-traffic envelope; Inductor cannot materially improve this local repro through scheduler fusion, scatter-reduce, split-K, or algebraic elimination based on this measurement because the remaining work is mandatory patch/position/affine reads, row statistics, rsqrt, and output stores; the fix is BANDWIDTH_BOUND: record this full-scope attempt as not_true_floor and treat the repro as at the compiler floor unless a broader norm-template/codegen change beats the current baseline."""
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
DTYPE = torch.float32

if triton is not None:

    @triton.jit
    def _vit_patch_layernorm_kernel(
        class_token_ptr,
        conv_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        conv_stride_b: tl.constexpr,
        conv_stride_c: tl.constexpr,
        conv_stride_p: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        tokens_for_row = rows % tokens
        batches = rows // tokens
        cols = tl.arange(0, BLOCK_C)
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask[:, None] & col_mask[None, :]

        is_class_token = tokens_for_row == 0
        class_values = tl.load(
            class_token_ptr + cols[None, :],
            mask=mask & is_class_token[:, None],
            other=0.0,
        ).to(tl.float32)
        patch_values = tl.load(
            conv_ptr
            + batches[:, None] * conv_stride_b
            + (tokens_for_row[:, None] - 1) * conv_stride_p
            + cols[None, :] * conv_stride_c,
            mask=mask & (tokens_for_row[:, None] != 0),
            other=0.0,
        ).to(tl.float32)
        position = tl.load(
            position_ptr + tokens_for_row[:, None] * hidden + cols[None, :],
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        x = class_values + patch_values + position

        x_masked = tl.where(mask, x, 0.0)
        sum_x = tl.sum(x_masked, axis=1)
        sum_x2 = tl.sum(x_masked * x_masked, axis=1)
        mean = sum_x / hidden
        variance = sum_x2 / hidden - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = (x - mean[:, None]) * invstd[:, None] * weight[None, :] + bias[None, :]

        tl.store(out_ptr + rows[:, None] * hidden + cols[None, :], y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_inputs(
    inputs: tuple[Any, ...] | list[Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    class_token, convolution, position, weight, bias, expand_shape, view_shape, output_shape = inputs
    tensor_inputs = (class_token, convolution, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first five repro inputs must be tensors")
    if not all(value.is_cuda for value in tensor_inputs):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if not all(value.dtype == DTYPE for value in tensor_inputs):
        dtypes = [value.dtype for value in tensor_inputs]
        raise TypeError(f"all tensor inputs must be torch.float32, got {dtypes}")

    if len(class_token.shape) != 3 or class_token.shape[0] != 1 or class_token.shape[1] != 1:
        raise ValueError(f"class token shape {tuple(class_token.shape)} must be (1, 1, hidden)")
    if len(convolution.shape) != 4:
        raise ValueError(f"convolution shape {tuple(convolution.shape)} must be 4D")
    if len(position.shape) != 3 or position.shape[0] != 1:
        raise ValueError(f"position shape {tuple(position.shape)} must be (1, tokens, hidden)")
    if len(weight.shape) != 1 or len(bias.shape) != 1:
        raise ValueError("LayerNorm weight and bias must be rank-1 tensors")

    batch = int(convolution.shape[0])
    hidden = int(convolution.shape[1])
    height = int(convolution.shape[2])
    width = int(convolution.shape[3])
    spatial = height * width
    tokens = spatial + 1
    rows = batch * tokens

    expected_tensor_shapes = (
        (1, 1, hidden),
        (batch, hidden, height, width),
        (1, tokens, hidden),
        (hidden,),
        (hidden,),
    )
    for index, (value, expected_shape) in enumerate(zip(tensor_inputs, expected_tensor_shapes)):
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected_shape}")

    if not class_token.is_contiguous():
        raise ValueError(f"class token must be contiguous, got stride={class_token.stride()}")
    if not position.is_contiguous():
        raise ValueError(f"position tensor must be contiguous, got stride={position.stride()}")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError(f"LayerNorm weight/bias must be contiguous, got {weight.stride()} and {bias.stride()}")
    if convolution.stride(2) != width * convolution.stride(3):
        raise ValueError(
            "convolution H/W dimensions must be viewable as the repro's flattened patch axis, "
            f"got shape={tuple(convolution.shape)} stride={convolution.stride()}"
        )

    expand_shape_tuple = _shape_tuple(expand_shape)
    view_shape_tuple = _shape_tuple(view_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    if expand_shape_tuple != (batch, -1, -1):
        raise ValueError(f"unexpected expand shape parameter: {expand_shape!r}")
    if view_shape_tuple != (batch, hidden, spatial):
        raise ValueError(f"unexpected convolution view shape parameter: {view_shape!r}")
    if output_shape_tuple != (rows, hidden):
        raise ValueError(f"unexpected output view shape parameter: {output_shape!r}")

    return class_token, convolution, position, weight, bias, output_shape_tuple


@oracle_impl(hardware="H100", shapes="(T([1, 1, 192], f32), T([128, 192, 14, 14], f32), T([1, 197, 192], f32), T([192], f32), T([192], f32), S([128, -1, -1]), S([128, 192, 196]), S([25216, 192]))")
def oracle_forward(inputs):
    """Run the complete Repro.forward class-token/patch LayerNorm computation.

    SCOPE INVARIANT: accepts the same 8 inputs as Repro.forward() and returns
    the same single contiguous float32 `[batch * tokens, hidden]` output.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_vit_patch_layernorm.py")

    class_token, convolution, position, weight, bias, output_shape = _validate_inputs(inputs)
    rows, hidden = output_shape
    tokens = int(position.shape[1])
    block_c = _next_power_of_2(hidden)
    block_r = 16 if hidden <= 256 else 1
    output = torch.empty_strided(
        output_shape,
        (hidden, 1),
        device=convolution.device,
        dtype=DTYPE,
    )
    _vit_patch_layernorm_kernel[(triton.cdiv(rows, block_r),)](
        class_token,
        convolution,
        position,
        weight,
        bias,
        output,
        total_rows=rows,
        tokens=tokens,
        hidden=hidden,
        conv_stride_b=int(convolution.stride(0)),
        conv_stride_c=int(convolution.stride(1)),
        conv_stride_p=int(convolution.stride(3)),
        eps=EPS,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        num_warps=8 if block_r > 1 else 1,
        num_stages=3,
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
