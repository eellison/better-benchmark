"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete DeiT distilled-token patch LayerNorm scope in one Triton row-block kernel, including both expanded token rows, the strided convolution reshape/permute patch gather, positional add, fp32 population var_mean over hidden size 768, eps=1e-6 rsqrt normalization, affine epilogue, and final contiguous [25344, 768] reshape, whereas Inductor lowers the decomposed expand/reshape/permute/cat/add/var_mean graph through generic producers and a normalization schedule instead of one specialized patch-token LayerNorm producer; Inductor cannot do this today because norm-template canonicalization does not recognize this distilled-token strided patch assembly and cannot fold its token expansion, patch-index arithmetic, statistics, affine, and final view into one generated kernel; the fix is NEW_PATTERN: add a guarded DeiT/ViT distilled patch LayerNorm template that emits the fused token expansion, strided patch gather, row-blocked statistics, affine epilogue, and final view directly."""
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
CHANNELS = 768
PATCH_H = 14
PATCH_W = 14
PATCHES = PATCH_H * PATCH_W
TOKENS = PATCHES + 2
ROWS = BATCH * TOKENS
CONV_SHAPE = (BATCH, CHANNELS, PATCH_H, PATCH_W)
TOKEN_SHAPE = (1, 1, CHANNELS)
POS_SHAPE = (1, TOKENS, CHANNELS)
AFFINE_SHAPE = (CHANNELS,)
CONV_RESHAPE = (BATCH, CHANNELS, PATCHES)
EXPAND_SHAPE = (BATCH, -1, -1)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
EPS = 1.0e-6
BLOCK_H = 1024
ROW_BLOCK = 8

if triton is not None:

    @triton.jit
    def _deit_distilled_patch_layernorm_kernel(
        cls_ptr,
        dist_ptr,
        conv_ptr,
        pos_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
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

        cls_values = tl.load(
            cls_ptr + cols,
            mask=col_mask & (token == 0),
            other=0.0,
        )
        dist_values = tl.load(
            dist_ptr + cols,
            mask=col_mask & (token == 1),
            other=0.0,
        )
        patch_values = tl.load(
            conv_ptr + batch * conv_s0 + cols * conv_s1 + patch * conv_patch_s,
            mask=mask & (token >= 2),
            other=0.0,
        )
        position_values = tl.load(
            pos_ptr + token * hidden + cols,
            mask=mask,
            other=0.0,
        )

        x = (cls_values + dist_values + patch_values + position_values).to(tl.float32)
        x_for_reduce = tl.where(mask, x, 0.0)
        mean = tl.sum(x_for_reduce, axis=1)[:, None] / hidden
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None]
            / hidden
        )
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"shape parameter {value!r} is not iterable") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != dtype:
        raise TypeError(f"{name} has dtype {value.dtype}, expected {dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    (
        cls_token,
        dist_token,
        convolution,
        pos_embed,
        weight,
        bias,
        cls_expand_shape,
        dist_expand_shape,
        conv_reshape_shape,
        output_shape,
    ) = inputs

    cls_token = _require_tensor("arg4_1", cls_token, TOKEN_SHAPE, torch.float32)
    dist_token = _require_tensor("arg5_1", dist_token, TOKEN_SHAPE, torch.float32)
    convolution = _require_tensor("convolution", convolution, CONV_SHAPE, torch.float32)
    pos_embed = _require_tensor("arg3_1", pos_embed, POS_SHAPE, torch.float32)
    weight = _require_tensor("arg6_1", weight, AFFINE_SHAPE, torch.float32)
    bias = _require_tensor("arg7_1", bias, AFFINE_SHAPE, torch.float32)

    contiguous_inputs = (cls_token, dist_token, pos_embed, weight, bias)
    for index, value in enumerate(contiguous_inputs):
        if not value.is_contiguous():
            raise ValueError(
                f"non-convolution tensor {index} must be contiguous, got stride={value.stride()}"
            )

    if convolution.stride(2) != PATCH_W * convolution.stride(3):
        raise ValueError(
            "convolution H/W strides must be reshape-compatible with [B, C, H*W], "
            f"got stride={convolution.stride()}"
        )

    tensor_inputs = (cls_token, dist_token, convolution, pos_embed, weight, bias)
    device = cls_token.device
    if any(value.device != device for value in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(cls_expand_shape) != EXPAND_SHAPE:
        raise ValueError(f"unexpected class-token expand shape: {cls_expand_shape!r}")
    if _shape_tuple(dist_expand_shape) != EXPAND_SHAPE:
        raise ValueError(f"unexpected dist-token expand shape: {dist_expand_shape!r}")
    if _shape_tuple(conv_reshape_shape) != CONV_RESHAPE:
        raise ValueError(f"unexpected convolution reshape shape: {conv_reshape_shape!r}")
    if _shape_tuple(output_shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output reshape shape: {output_shape!r}")

    return cls_token, dist_token, convolution, pos_embed, weight, bias


@oracle_impl(hardware="H100", shapes="(T([1, 1, 768], f32), T([1, 1, 768], f32), T([128, 768, 14, 14], f32, stride=(150528, 1, 10752, 768)), T([1, 198, 768], f32), T([768], f32), T([768], f32), S([128, -1, -1]), S([128, -1, -1]), S([128, 768, 196]), S([25344, 768]))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_deit_distilled_patch_layernorm.py")

    cls_token, dist_token, convolution, pos_embed, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution.device,
        dtype=torch.float32,
    )

    _deit_distilled_patch_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        cls_token,
        dist_token,
        convolution,
        pos_embed,
        weight,
        bias,
        out,
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
