"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXt pooled residual LayerNorm scope in one Triton row kernel, fusing the residual add, 7x7 spatial mean, 640-channel population var_mean, affine epilogue, and final contiguous [128,640] output store, whereas Inductor currently lowers the decomposed add/spatial-mean/channel-LayerNorm/reshape graph through generic normalization scheduling around a pooled intermediate; Inductor cannot do this today because its normalization scheduler does not canonicalize a fixed spatial average producer into the channel LayerNorm row template; the fix is SCHEDULER_FUSION: teach the norm template to fold fixed small spatial reductions and residual producers into the row-statistics load plan when that beats materializing the pooled tensor."""
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
    has_stochastic_ops,
)


BATCH = 128
CHANNELS = 640
HEIGHT = 7
WIDTH = 7
SPATIAL = HEIGHT * WIDTH
EPS = 1.0e-6
BLOCK_C = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.jit
    def _pooled_layernorm_kernel(
        conv_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        n_stride: tl.constexpr,
        c_stride: tl.constexpr,
        h_stride: tl.constexpr,
        w_stride: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        spatial: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < channels

        acc = tl.zeros((BLOCK_C,), tl.float32)
        for h in tl.static_range(0, height):
            for w in tl.static_range(0, width):
                offsets = row * n_stride + cols * c_stride + h * h_stride + w * w_stride
                conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
                acc += conv + residual

        pooled = tl.where(mask, acc / spatial, 0.0)
        mean = tl.sum(pooled, axis=0) / channels
        centered = pooled - mean
        variance = tl.sum(tl.where(mask, centered * centered, 0.0), axis=0) / channels
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = centered * invstd * weight + bias
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
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    image_shape = (BATCH, CHANNELS, HEIGHT, WIDTH)
    image_stride = (CHANNELS * SPATIAL, 1, WIDTH * CHANNELS, CHANNELS)
    convolution = _require_tensor("convolution_45", inputs[0], image_shape, image_stride)
    residual = _require_tensor("add_85", inputs[1], image_shape, image_stride)
    weight = _require_tensor("arg157_1", inputs[2], (CHANNELS,), (1,))
    bias = _require_tensor("arg158_1", inputs[3], (CHANNELS,), (1,))
    if _shape_tuple(inputs[4]) != (BATCH, CHANNELS):
        raise ValueError(f"unexpected reshape shape parameter: {inputs[4]!r}")

    device = convolution.device
    if residual.device != device or weight.device != device or bias.device != device:
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return convolution, residual, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    convolution, residual, weight, bias = _validate_inputs(inputs)
    x = convolution + residual
    pooled = torch.mean(x, dim=(-1, -2), keepdim=True)
    y = torch.permute(torch.as_strided(pooled, (BATCH, CHANNELS, 1, 1), (CHANNELS, 1, CHANNELS, CHANNELS)), (0, 2, 3, 1))
    variance, mean = torch.var_mean(y, dim=[3], correction=0, keepdim=True)
    y = (y - mean) * torch.rsqrt(variance + EPS)
    y = y * weight + bias
    y = torch.permute(y, (0, 3, 1, 2))
    return torch.reshape(y, (BATCH, CHANNELS))


@oracle_impl(hardware="H100", shapes="(T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([128, 640, 7, 7], f32, stride=(31360, 1, 4480, 640)), T([640], f32), T([640], f32), S([128, 640]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> torch.Tensor:
    """Run the complete Repro.forward pooled residual LayerNorm computation.

    SCOPE INVARIANT: accepts the same inputs as Repro.forward() and returns the
    same single contiguous float32[BATCH, CHANNELS] output tensor.
    """
    if triton is None:
        return _torch_reference(inputs)

    convolution, residual, weight, bias = _validate_inputs(inputs)
    out = torch.empty_strided(
        (BATCH, CHANNELS),
        (CHANNELS, 1),
        device=convolution.device,
        dtype=torch.float32,
    )
    _pooled_layernorm_kernel[(BATCH,)](
        convolution,
        residual,
        weight,
        bias,
        out,
        n_stride=convolution.stride(0),
        c_stride=convolution.stride(1),
        h_stride=convolution.stride(2),
        w_stride=convolution.stride(3),
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        spatial=SPATIAL,
        eps=EPS,
        BLOCK_C=BLOCK_C,
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
