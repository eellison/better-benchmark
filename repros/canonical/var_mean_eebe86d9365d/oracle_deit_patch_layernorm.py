"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeiT distilled-token patch LayerNorm scope by row-blocking eight adjacent token rows in one Triton program, including both expanded token rows, the convolution patch view/permute gather for either listed convolution stride, positional add, fp32 population var_mean over hidden size 768, eps=1e-6 affine epilogue, and final contiguous [25344, 768] view, whereas Inductor lowers the decomposed expand/cat/view/permute/add/var_mean graph through its generic normalization schedule without a DeiT patch-token template that amortizes affine loads and patch-layout arithmetic across neighboring rows; Inductor cannot do this today because norm-template canonicalization does not recognize this two-token-plus-patch assembly as one row-tiled LayerNorm lowering; the fix is NEW_PATTERN: add a guarded DeiT/ViT distilled patch LayerNorm template that folds the token expansion, patch gather, row-blocked statistics, affine epilogue, and final view into specialized codegen."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

BATCH = 128
TOKENS = 198
PATCHES = 196
CHANNELS = 768
HEIGHT = 14
WIDTH = 14
ROWS = BATCH * TOKENS
EPS = 1.0e-6
BLOCK_H = 1024
ROW_BLOCK = 8

if triton is not None:

    @triton.jit
    def _deit_patch_layernorm_kernel(
        cls_token_ptr,
        dist_token_ptr,
        convolution_ptr,
        position_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        conv_s0: tl.constexpr,
        conv_s1: tl.constexpr,
        conv_patch_s: tl.constexpr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        total_rows: tl.constexpr,
        eps: tl.constexpr,
        block_h: tl.constexpr,
        row_block: tl.constexpr,
    ):
        rows = tl.program_id(0) * row_block + tl.arange(0, row_block)[:, None]
        cols = tl.arange(0, block_h)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < hidden
        mask = row_mask & col_mask

        batch = rows // tokens
        token = rows - batch * tokens
        patch = token - 2

        is_cls = token == 0
        is_dist = token == 1
        cls_values = tl.load(
            cls_token_ptr + cols,
            mask=col_mask & is_cls,
            other=0.0,
        )
        dist_values = tl.load(
            dist_token_ptr + cols,
            mask=col_mask & is_dist,
            other=0.0,
        )
        patch_values = tl.load(
            convolution_ptr + batch * conv_s0 + cols * conv_s1 + patch * conv_patch_s,
            mask=mask & (token >= 2),
            other=0.0,
        )
        position_values = tl.load(
            position_ptr + token * hidden + cols,
            mask=mask,
            other=0.0,
        )

        x = (cls_values + dist_values + patch_values + position_values).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None] / hidden
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(output_ptr + rows * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, int],
]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        cls_token,
        dist_token,
        convolution,
        position,
        weight,
        bias,
        expand_shape0,
        expand_shape1,
        conv_view_shape,
        output_shape,
    ) = inputs

    tensor_inputs = (cls_token, dist_token, convolution, position, weight, bias)
    if not all(isinstance(value, torch.Tensor) for value in tensor_inputs):
        raise TypeError("the first six repro inputs must be tensors")

    expected_shapes = (
        (1, 1, CHANNELS),
        (1, 1, CHANNELS),
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (1, TOKENS, CHANNELS),
        (CHANNELS,),
        (CHANNELS,),
    )
    for index, (value, expected) in enumerate(zip(tensor_inputs, expected_shapes)):
        if tuple(value.shape) != expected:
            raise ValueError(f"input {index} shape {tuple(value.shape)} != {expected}")
        if value.dtype != torch.float32:
            raise TypeError(f"input {index} dtype {value.dtype} != torch.float32")
        if not value.is_cuda:
            raise RuntimeError("CUDA tensors are required for the Triton oracle")
        if value.device != cls_token.device:
            raise ValueError(f"input {index} device {value.device} != {cls_token.device}")

    for index, value in ((0, cls_token), (1, dist_token), (3, position), (4, weight), (5, bias)):
        if not value.is_contiguous():
            raise ValueError(f"input {index} must be contiguous, got stride={value.stride()}")

    if convolution.stride(2) != WIDTH * convolution.stride(3):
        raise ValueError(
            "convolution H/W strides must be view-compatible with [B, C, H*W], "
            f"got stride={convolution.stride()}"
        )

    expand_shape0_tuple = _shape_tuple(expand_shape0)
    expand_shape1_tuple = _shape_tuple(expand_shape1)
    conv_view_shape_tuple = _shape_tuple(conv_view_shape)
    output_shape_tuple = _shape_tuple(output_shape)
    valid_expand_shape = (BATCH, -1, -1)
    resolved_expand_shape = (BATCH, 1, CHANNELS)
    if expand_shape0_tuple not in (valid_expand_shape, resolved_expand_shape):
        raise ValueError(f"unexpected first expand shape parameter: {expand_shape0!r}")
    if expand_shape1_tuple not in (valid_expand_shape, resolved_expand_shape):
        raise ValueError(f"unexpected second expand shape parameter: {expand_shape1!r}")
    if conv_view_shape_tuple != (BATCH, CHANNELS, PATCHES):
        raise ValueError(f"unexpected convolution view shape parameter: {conv_view_shape!r}")
    if output_shape_tuple != (ROWS, CHANNELS):
        raise ValueError(f"unexpected output shape parameter: {output_shape!r}")

    return cls_token, dist_token, convolution, position, weight, bias, output_shape_tuple


def oracle_forward(inputs):
    """Run the complete Repro.forward token/patch assembly plus LayerNorm.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same single f32 `[25344, 768]` output with contiguous `(768, 1)` stride.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_deit_patch_layernorm.py")

    cls_token, dist_token, convolution, position, weight, bias, output_shape = _validate_inputs(inputs)
    output = torch.empty_strided(
        output_shape,
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    _deit_patch_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        cls_token,
        dist_token,
        convolution,
        position,
        weight,
        bias,
        output,
        conv_s0=convolution.stride(0),
        conv_s1=convolution.stride(1),
        conv_patch_s=convolution.stride(3),
        hidden=CHANNELS,
        tokens=TOKENS,
        total_rows=ROWS,
        eps=EPS,
        block_h=BLOCK_H,
        row_block=ROW_BLOCK,
        num_warps=4,
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
