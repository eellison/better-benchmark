"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ViT patch-position LayerNorm scope in one Triton row-reduction kernel, folding the NCHW channels-last flatten/transpose, positional embedding add, hidden-size-768 population var_mean, eps=1e-6 rsqrt, affine epilogue, and final contiguous [32768, 768] store, whereas Inductor currently emits one generic fused normalization kernel for the decomposed reshape/permute/add/var_mean/affine/view graph without this dedicated patch-token row load plan; Inductor cannot do this today because its normalization scheduler does not canonicalize channels-last patch flatten plus broadcast positional add producers into the fixed-hidden LayerNorm row schedule; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to sink static patch-view and positional-broadcast producers into the row-statistics and affine output kernel."""
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

BATCH = 128
CHANNELS = 768
HEIGHT = 16
WIDTH = 16
PATCHES = HEIGHT * WIDTH
ROWS = BATCH * PATCHES
EPS = 1.0e-6
BLOCK_C = 1024
CONV_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
CONV_STRIDE = (CHANNELS * PATCHES, 1, WIDTH * CHANNELS, CHANNELS)
PATCH_VIEW_SHAPE = (BATCH, CHANNELS, PATCHES)
OUTPUT_SHAPE = (ROWS, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)

# Import shared oracle infrastructure. Run first:
#   python -m pip install --no-build-isolation -e .
# Use the installed oracle_harness package; run editable install before checks.
# Do not add custom benchmark functions. bench_oracle() owns timing so CUDAGraph,
# GPU locking, and interleaved oracle/compile measurement are preserved.
from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


def _require_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if stride is not None and tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    convolution = _require_tensor("convolution", inputs[0], CONV_SHAPE, CONV_STRIDE)
    pos_embed = _require_tensor("primals_4", inputs[1], (1, PATCHES, CHANNELS))
    weight = _require_tensor("primals_5", inputs[2], (CHANNELS,), (1,))
    bias = _require_tensor("primals_6", inputs[3], (CHANNELS,), (1,))

    if pos_embed.stride(1) != CHANNELS or pos_embed.stride(2) != 1:
        raise ValueError(
            f"primals_4 must be contiguous across patch/channel, got stride={pos_embed.stride()}"
        )
    if _shape_tuple(inputs[4]) != PATCH_VIEW_SHAPE:
        raise ValueError(f"unexpected first reshape shape parameter: {inputs[4]!r}")
    if _shape_tuple(inputs[5]) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected final reshape shape parameter: {inputs[5]!r}")

    device = convolution.device
    if any(value.device != device for value in (pos_embed, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return convolution, pos_embed, weight, bias


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
        conv_stride_h: tl.constexpr,
        conv_stride_w: tl.constexpr,
        pos_stride_p: tl.constexpr,
        pos_stride_c: tl.constexpr,
        channels: tl.constexpr,
        patches: tl.constexpr,
        width: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < channels

        batch = row // patches
        patch = row - batch * patches
        h = patch // width
        w = patch - h * width

        conv_offsets = (
            batch * conv_stride_n
            + cols * conv_stride_c
            + h * conv_stride_h
            + w * conv_stride_w
        )
        pos_offsets = patch * pos_stride_p + cols * pos_stride_c

        conv_value = tl.load(
            conv_ptr + conv_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        pos_value = tl.load(
            pos_ptr + pos_offsets,
            mask=mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        value = conv_value + pos_value

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
        normalized = (value - mean) * inv_std
        scaled = normalized * weight
        output = scaled + bias
        tl.store(out_ptr + row * channels + cols, output, mask=mask)


def _torch_full_scope(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, pos_embed, weight, bias = _validate_inputs(inputs)
    patch_shape = _shape_tuple(inputs[4])
    output_shape = _shape_tuple(inputs[5])
    reshape_default = torch.ops.aten.reshape.default(convolution, patch_shape)
    permute_default = torch.ops.aten.permute.default(reshape_default, [0, 2, 1])
    added = torch.ops.aten.add.Tensor(permute_default, pos_embed)
    var_mean = torch.ops.aten.var_mean.correction(
        added, [2], correction=0, keepdim=True
    )
    variance = var_mean[0]
    mean = var_mean[1]
    inv_std = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(variance, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(added, mean), inv_std)
    affine = torch.ops.aten.add.Tensor(torch.ops.aten.mul.Tensor(normalized, weight), bias)
    return torch.ops.aten.reshape.default(affine, output_shape)


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward patch-position LayerNorm computation.

    SCOPE INVARIANT: accepts the same six inputs as Repro.forward() and returns
    the same single fp32 `[32768,768]` contiguous tensor.
    """
    convolution, pos_embed, weight, bias = _validate_inputs(inputs)
    if (
        triton is None
        or libdevice is None
        or not convolution.is_cuda
        or not pos_embed.is_cuda
        or not weight.is_cuda
        or not bias.is_cuda
    ):
        return _torch_full_scope(inputs)

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=convolution.device,
        dtype=torch.float32,
    )
    _patch_pos_layernorm_kernel[(ROWS,)](
        convolution,
        pos_embed,
        weight,
        bias,
        output,
        conv_stride_n=convolution.stride(0),
        conv_stride_c=convolution.stride(1),
        conv_stride_h=convolution.stride(2),
        conv_stride_w=convolution.stride(3),
        pos_stride_p=pos_embed.stride(1),
        pos_stride_c=pos_embed.stride(2),
        channels=CHANNELS,
        patches=PATCHES,
        width=WIDTH,
        eps=EPS,
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return output


def _layout_signature(value: torch.Tensor) -> tuple[torch.dtype, tuple[int, ...], tuple[int, ...]]:
    return value.dtype, tuple(value.shape), tuple(value.stride())


def _check_layout(instance: torch.nn.Module, inputs: list[Any] | tuple[Any, ...]) -> bool:
    with torch.no_grad():
        expected = instance(*inputs)
        actual = oracle_forward(inputs)
        if actual.is_cuda:
            torch.cuda.synchronize()

    if not isinstance(expected, torch.Tensor) or not isinstance(actual, torch.Tensor):
        print("  SCOPE_MISMATCH: expected and oracle outputs must both be tensors")
        return False

    ok = _layout_signature(expected) == _layout_signature(actual)
    print(
        f"  output 0 layout: {'PASS' if ok else 'FAIL'} "
        f"(expected={_layout_signature(expected)} oracle={_layout_signature(actual)})"
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
