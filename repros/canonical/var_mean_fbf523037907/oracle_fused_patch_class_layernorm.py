"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete BEiT patch/class-token LayerNorm scope in one Triton row-reduction kernel, folding the NCHW channels-last flatten/transpose, class-token prepend, hidden-size-768 population var_mean, affine epilogue, and final contiguous [25216, 768] store, whereas Inductor currently schedules the class-token cat and strided patch-token producer separately from the normalization template; Inductor cannot do this today because its normalization scheduler does not canonicalize fixed prepend-cat plus metadata-view producers into the LayerNorm row load plan; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to sink static prepend-cat and view producers into the row-statistics and affine output kernel."""
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
HEIGHT = 14
WIDTH = 14
PATCHES = HEIGHT * WIDTH
TOKENS = PATCHES + 1
ROWS = BATCH * TOKENS
EPS = 1.0e-6
BLOCK_C = 1024


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _patch_class_layernorm_kernel(
        conv_ptr,
        cls_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        conv_stride_n: tl.constexpr,
        conv_stride_c: tl.constexpr,
        conv_stride_h: tl.constexpr,
        conv_stride_w: tl.constexpr,
        cls_stride_c: tl.constexpr,
        channels: tl.constexpr,
        tokens: tl.constexpr,
        width: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < channels

        batch = row // tokens
        token = row - batch * tokens
        is_class_token = token == 0

        patch = tl.maximum(token - 1, 0)
        h = patch // width
        w = patch - h * width
        conv_offsets = (
            batch * conv_stride_n
            + cols * conv_stride_c
            + h * conv_stride_h
            + w * conv_stride_w
        )
        class_offsets = cols * cls_stride_c

        patch_value = tl.load(
            conv_ptr + conv_offsets,
            mask=mask & ~is_class_token,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        class_value = tl.load(
            cls_ptr + class_offsets,
            mask=mask & is_class_token,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        value = tl.where(is_class_token, class_value, patch_value)

        reduced_value = tl.where(mask, value, 0.0)
        mean = tl.sum(reduced_value, axis=0) / channels
        centered = tl.where(mask, value - mean, 0.0)
        variance = tl.sum(centered * centered, axis=0) / channels
        inv_std = libdevice.rsqrt(variance + eps)

        weight = tl.load(
            weight_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        bias = tl.load(
            bias_ptr + cols,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        out = (value - mean) * inv_std * weight + bias
        tl.store(out_ptr + row * channels + cols, out, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    conv_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    conv_stride = (CHANNELS * PATCHES, 1, WIDTH * CHANNELS, CHANNELS)
    convolution = _require_tensor("convolution", inputs[0], conv_shape, conv_stride)
    class_token = _require_tensor("primals_4", inputs[1], (1, 1, CHANNELS), (CHANNELS, CHANNELS, 1))
    weight = _require_tensor("primals_6", inputs[2], (CHANNELS,), (1,))
    bias = _require_tensor("primals_7", inputs[3], (CHANNELS,), (1,))

    if _shape_tuple(inputs[4]) != (BATCH, CHANNELS, PATCHES):
        raise ValueError(f"unexpected first reshape shape parameter: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != (BATCH, -1, -1):
        raise ValueError(f"unexpected expand shape parameter: {inputs[5]!r}")
    if _shape_tuple(inputs[6]) != (ROWS, CHANNELS):
        raise ValueError(f"unexpected final reshape shape parameter: {inputs[6]!r}")

    device = convolution.device
    if any(value.device != device for value in (class_token, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return convolution, class_token, weight, bias


def _oracle_forward_torch(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, primals_4, primals_6, primals_7, shape0, shape1, shape2 = inputs
    reshape_default = torch.ops.aten.reshape.default(convolution, shape0)
    permute_default = torch.ops.aten.permute.default(reshape_default, [0, 2, 1])
    expand_default = torch.ops.aten.expand.default(primals_4, shape1)
    cat_default = torch.ops.aten.cat.default([expand_default, permute_default], 1)
    var_mean = torch.ops.aten.var_mean.correction(
        cat_default, [2], correction=0, keepdim=True
    )
    variance = var_mean[0]
    mean = var_mean[1]
    inv_std = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(cat_default, mean), inv_std
    )
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, primals_6), primals_7)
    return torch.ops.aten.reshape.default(affine, shape2)


@oracle_impl(hardware="H100", shapes="(T([128, 768, 14, 14], f32, stride=(150528, 1, 10752, 768)), T([1, 1, 768], f32), T([768], f32), T([768], f32), S([128, 768, 196]), S([128, -1, -1]), S([25216, 768]))")
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
    tensor_inputs = [value for value in inputs if isinstance(value, torch.Tensor)]
    if triton is None or not tensor_inputs or not all(value.is_cuda for value in tensor_inputs):
        return _oracle_forward_torch(inputs)

    convolution, class_token, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (ROWS, CHANNELS),
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    _patch_class_layernorm_kernel[(ROWS,)](
        convolution,
        class_token,
        weight,
        bias,
        output,
        conv_stride_n=convolution.stride(0),
        conv_stride_c=convolution.stride(1),
        conv_stride_h=convolution.stride(2),
        conv_stride_w=convolution.stride(3),
        cls_stride_c=class_token.stride(2),
        channels=CHANNELS,
        tokens=TOKENS,
        width=WIDTH,
        eps=EPS,
        BLOCK_C=BLOCK_C,
        num_warps=8,
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
