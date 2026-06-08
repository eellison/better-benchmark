"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ViT patch-token positional-add LayerNorm scope in one Triton row-block kernel, loading the reshape/permute convolution patches, folding the broadcast [1,256,768] add, correction=0 var_mean, eps=1e-6 rsqrt, affine epilogue, clone-contiguous layout, and final [32768,768] view, whereas Inductor currently lowers the decomposed view/permute/add/var_mean graph through its generic reduction-normalization path with Welford-style row statistics and extra patch-token index arithmetic; Inductor cannot do this today because norm-template canonicalization does not recognize this NCHW patch-flatten plus positional-broadcast pattern as a fixed hidden-size LayerNorm producer; the fix is NEW_PATTERN: add a guarded ViT patch-position LayerNorm template that emits row-blocked patch gathers, direct correction=0 statistics, affine epilogue, and contiguous flattened output directly."""
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


BATCH = 128
CHANNELS = 768
HEIGHT = 16
WIDTH = 16
TOKENS = HEIGHT * WIDTH
ROWS = BATCH * TOKENS
EPS = 1.0e-6
BLOCK_C = 1024
ROW_BLOCK = 16


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution = _require_tensor(
        "convolution",
        inputs[0],
        (BATCH, CHANNELS, HEIGHT, WIDTH),
    )
    pos_embed = _require_tensor("arg3_1", inputs[1], (1, TOKENS, CHANNELS))
    weight = _require_tensor("arg4_1", inputs[2], (CHANNELS,))
    bias = _require_tensor("arg5_1", inputs[3], (CHANNELS,))
    if _shape_tuple(inputs[4]) != (BATCH, CHANNELS, TOKENS):
        raise ValueError(f"unexpected convolution view shape parameter: {inputs[4]!r}")

    if convolution.stride(2) != WIDTH * convolution.stride(3):
        raise ValueError(
            "convolution H/W strides must be reshape-compatible with [B, C, H*W], "
            f"got stride={convolution.stride()}"
        )
    if tuple(weight.stride()) != (1,) or tuple(bias.stride()) != (1,):
        raise ValueError("LayerNorm weight and bias must be contiguous")

    device = convolution.device
    if pos_embed.device != device or weight.device != device or bias.device != device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return convolution, pos_embed, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, arg3_1, arg4_1, arg5_1, shape_param = inputs
    view_default = torch.ops.aten.view.default(convolution, shape_param)
    permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1])
    add_tensor = torch.ops.aten.add.Tensor(permute_default, arg3_1)
    var_mean_correction = torch.ops.aten.var_mean.correction(
        add_tensor,
        [2],
        correction=0,
        keepdim=True,
    )
    variance = var_mean_correction[0]
    mean = var_mean_correction[1]
    sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, mean)
    add_tensor_1 = torch.ops.aten.add.Tensor(variance, EPS)
    rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1)
    mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default)
    mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, arg4_1)
    add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, arg5_1)
    clone_default = torch.ops.aten.clone.default(
        add_tensor_2,
        memory_format=torch.contiguous_format,
    )
    return torch.ops.aten._unsafe_view.default(clone_default, [ROWS, CHANNELS])


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _patch_pos_layernorm_kernel(
        conv_ptr,
        pos_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        conv_stride_n: tl.constexpr,
        conv_stride_c: tl.constexpr,
        conv_patch_stride: tl.constexpr,
        pos_stride_t: tl.constexpr,
        pos_stride_c: tl.constexpr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        total_rows: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)[:, None]
        cols = tl.arange(0, BLOCK_C)[None, :]
        row_mask = rows < total_rows
        col_mask = cols < channels
        mask = row_mask & col_mask

        batch = rows // tokens
        token = rows - batch * tokens

        conv_offsets = (
            batch * conv_stride_n
            + cols * conv_stride_c
            + token * conv_patch_stride
        )
        pos_offsets = token * pos_stride_t + cols * pos_stride_c

        conv = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
        pos = tl.load(pos_ptr + pos_offsets, mask=mask, other=0.0).to(tl.float32)
        x = conv + pos

        mean = tl.sum(tl.where(mask, x, 0.0), axis=1)[:, None] / channels
        centered = x - mean
        variance = (
            tl.sum(tl.where(mask, centered * centered, 0.0), axis=1)[:, None]
            / channels
        )
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias
        tl.store(out_ptr + rows * channels + cols, out, mask=mask)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward patch-token LayerNorm computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous float32[32768, 768] output tensor.
    """
    if triton is None:
        return _torch_reference(inputs)

    convolution, pos_embed, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    _patch_pos_layernorm_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        convolution,
        pos_embed,
        weight,
        bias,
        out,
        conv_stride_n=convolution.stride(0),
        conv_stride_c=convolution.stride(1),
        conv_patch_stride=convolution.stride(3),
        pos_stride_t=pos_embed.stride(1),
        pos_stride_c=pos_embed.stride(2),
        channels=CHANNELS,
        tokens=TOKENS,
        total_rows=ROWS,
        eps=EPS,
        BLOCK_C=BLOCK_C,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=8,
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
