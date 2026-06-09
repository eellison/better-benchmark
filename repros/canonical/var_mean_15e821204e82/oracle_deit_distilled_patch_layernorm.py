"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete DeiT distilled-token patch LayerNorm scope in one Triton row-block kernel, including the strided convolution reshape/permute patch gather, both expanded token rows, positional add, fp32 population var_mean over hidden size 768, eps=1e-6 libdevice rsqrt normalization, affine epilogue, and final contiguous [25344, 768] reshape, whereas Inductor already lowers the decomposed reshape/permute/expand/cat/add/var_mean graph to a similarly fused normalization schedule for this full scope; the oracle does not beat Inductor under CUDAGraph timing because the remaining work is dominated by mandatory patch/token/position/affine reads, one hidden-size reduction, rsqrt, and output stores rather than an avoidable intermediate; the fix is BANDWIDTH_BOUND: record this as an at-floor DeiT/ViT distilled patch LayerNorm case unless broader normalization-template, launch-overhead, or memory-traffic changes move both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    libdevice = None

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
AFFINE_OUTPUT_SHAPE = (BATCH, TOKENS, CHANNELS)
AFFINE_OUTPUT_STRIDE = (TOKENS * CHANNELS, CHANNELS, 1)
EPS = 1.0e-6
BLOCK_H = 1024
ROW_BLOCK = 8


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Size):
        return tuple(int(dim) for dim in value)
    if isinstance(value, (list, tuple)):
        return tuple(int(dim) for dim in value)
    raise TypeError(f"expected shape parameter list/tuple, got {type(value).__name__}")


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
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
        convolution,
        cls_token,
        dist_token,
        pos_embed,
        weight,
        bias,
        conv_reshape_shape,
        cls_expand_shape,
        dist_expand_shape,
        output_shape,
    ) = inputs

    convolution = _require_tensor("convolution", convolution, CONV_SHAPE)
    cls_token = _require_tensor("primals_5", cls_token, TOKEN_SHAPE)
    dist_token = _require_tensor("primals_6", dist_token, TOKEN_SHAPE)
    pos_embed = _require_tensor("primals_4", pos_embed, POS_SHAPE)
    weight = _require_tensor("primals_7", weight, AFFINE_SHAPE)
    bias = _require_tensor("primals_8", bias, AFFINE_SHAPE)

    if convolution.stride(2) != PATCH_W * convolution.stride(3):
        raise ValueError(
            "convolution H/W strides must be reshape-compatible with [B, C, H*W], "
            f"got stride={convolution.stride()}"
        )
    for name, tensor in (
        ("primals_5", cls_token),
        ("primals_6", dist_token),
        ("primals_4", pos_embed),
        ("primals_7", weight),
        ("primals_8", bias),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous, got stride={tensor.stride()}")

    device = convolution.device
    tensor_inputs = (cls_token, dist_token, pos_embed, weight, bias)
    if any(tensor.device != device for tensor in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    if _shape_tuple(conv_reshape_shape) != CONV_RESHAPE:
        raise ValueError(f"unexpected convolution reshape shape: {conv_reshape_shape!r}")
    if _shape_tuple(cls_expand_shape) != EXPAND_SHAPE:
        raise ValueError(f"unexpected class-token expand shape: {cls_expand_shape!r}")
    if _shape_tuple(dist_expand_shape) != EXPAND_SHAPE:
        raise ValueError(f"unexpected dist-token expand shape: {dist_expand_shape!r}")
    if _shape_tuple(output_shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output reshape shape: {output_shape!r}")

    return convolution, cls_token, dist_token, pos_embed, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    (
        convolution,
        primals_5,
        primals_6,
        primals_4,
        primals_7,
        primals_8,
        shape_param_0,
        shape_param_1,
        shape_param_2,
        shape_param_3,
    ) = inputs
    reshape_default = torch.ops.aten.reshape.default(convolution, shape_param_0)
    permute_default = torch.ops.aten.permute.default(reshape_default, [0, 2, 1])
    expand_default = torch.ops.aten.expand.default(primals_5, shape_param_1)
    expand_default_1 = torch.ops.aten.expand.default(primals_6, shape_param_2)
    cat_default = torch.ops.aten.cat.default(
        [expand_default, expand_default_1, permute_default],
        1,
    )
    add_tensor = torch.ops.aten.add.Tensor(cat_default, primals_4)
    var_mean_correction = torch.ops.aten.var_mean.correction(
        add_tensor,
        [2],
        correction=0,
        keepdim=True,
    )
    variance = var_mean_correction[0]
    mean = var_mean_correction[1]
    add_tensor_1 = torch.ops.aten.add.Tensor(variance, EPS)
    rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1)
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, mean)
    mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, primals_7)
    add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, primals_8)
    return torch.ops.aten.reshape.default(add_tensor_2, shape_param_3)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _deit_distilled_patch_layernorm_kernel(
        conv_ptr,
        cls_ptr,
        dist_ptr,
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
        invstd = libdevice.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        y = centered * invstd * weight + bias

        tl.store(out_ptr + rows * hidden + cols, y, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([128, 768, 14, 14], f32, stride=(150528, 1, 10752, 768)), T([1, 1, 768], f32), T([1, 1, 768], f32), T([1, 198, 768], f32), T([768], f32), T([768], f32), S([128, 768, 196]), S([128, -1, -1]), S([128, -1, -1]), S([25344, 768]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the full Repro.forward scope and return the single flattened output."""
    if triton is None or libdevice is None:
        return _torch_reference(inputs)

    convolution, cls_token, dist_token, pos_embed, weight, bias = _validate_inputs(inputs)
    affine_out = torch.empty_strided(
        AFFINE_OUTPUT_SHAPE,
        AFFINE_OUTPUT_STRIDE,
        device=convolution.device,
        dtype=torch.float32,
    )

    _deit_distilled_patch_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        convolution,
        cls_token,
        dist_token,
        pos_embed,
        weight,
        bias,
        affine_out,
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
    return torch.ops.aten.reshape.default(affine_out, inputs[9])


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
