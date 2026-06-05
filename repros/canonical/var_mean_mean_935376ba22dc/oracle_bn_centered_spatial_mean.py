"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Visformer training-BatchNorm fragment in one channel-tiled kernel by retaining the 7x7 per-batch/channel tile after the `add_175 + convolution_56` statistics reduction, directly writing the BN rsqrt, centered activation, affine spatial mean, and in-place running-stat outputs, whereas Inductor emits separate norm-template statistics and consumer kernels that reread both inputs for the centered/spatial-mean outputs; Inductor cannot do this today because its normalization reduction template does not fuse dependent full-tensor and spatial-reduction consumers into the same channel program while preserving mutable `copy_` aliases; the fix is SCHEDULER_FUSION: teach the BN-training template to sink these linear consumers into the stat kernel when the reduction tile is small enough to retain."""
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

N = 128
CHANNELS = 768
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871

NCHW_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
NCHW_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
SPATIAL_MEAN_SHAPE = (N, CHANNELS)
SPATIAL_MEAN_STRIDE = (CHANNELS, 1)

FUSED_BLOCK_N = 128
FUSED_BLOCK_HW = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_centered_spatial_mean_kernel(
        add_ptr,
        conv_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_out_ptr,
        spatial_mean_out_ptr,
        centered_out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_index = tl.arange(0, BLOCK_N)
        n_offsets = n_index[:, None]
        hw_offsets = tl.arange(0, BLOCK_HW)[None, :]
        hw_mask = hw_offsets < hw_size
        offsets = (n_offsets * channels + channel) * hw_size + hw_offsets

        x = (
            tl.load(add_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
            + tl.load(conv_ptr + offsets, mask=hw_mask, other=0.0).to(tl.float32)
        )
        x = tl.where(hw_mask, x, 0.0)
        row_sum = tl.sum(x, axis=1)
        row_sum_sq = tl.sum(x * x, axis=1)
        total_sum = tl.sum(row_sum, axis=0)
        total_sum_sq = tl.sum(row_sum_sq, axis=0)

        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(invstd_out_ptr + channel, invstd)
        tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )

        centered = x - mean
        tl.store(centered_out_ptr + offsets, centered, mask=hw_mask)

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        pooled = (row_sum / hw_size - mean) * invstd
        pooled = pooled * weight + bias
        tl.store(spatial_mean_out_ptr + n_index * channels + channel, pooled)


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    add_175, convolution_56, arg200_1, arg201_1, arg202_1, arg203_1, shape_param = inputs
    add_tensor = _expect_f32_tensor("add_175", add_175, NCHW_SHAPE, NCHW_STRIDE)
    convolution = _expect_f32_tensor("convolution_56", convolution_56, NCHW_SHAPE, NCHW_STRIDE)
    running_mean = _expect_f32_tensor("arg200_1", arg200_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg201_1", arg201_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg202_1", arg202_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg203_1", arg203_1, STAT_SHAPE, STAT_STRIDE)

    if list(shape_param) != [N, CHANNELS]:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    device = add_tensor.device
    if any(t.device != device for t in (convolution, running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return add_tensor, convolution, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    add_tensor, convolution, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    x = add_tensor + convolution
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    normalized = (x - mean) * invstd
    spatial_mean = normalized * weight[None, :, None, None] + bias[None, :, None, None]
    spatial_mean = torch.mean(spatial_mean, dim=(-1, -2), keepdim=True).view(N, CHANNELS)
    centered = x - mean
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean.squeeze((0, 2, 3)) * MOMENTUM)
    running_var.copy_(
        running_var * (1.0 - MOMENTUM)
        + var.squeeze((0, 2, 3)) * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return invstd.squeeze((0, 2, 3)), spatial_mean, centered, running_mean, running_var


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
        return _torch_reference(inputs)

    add_tensor, convolution, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if not add_tensor.is_cuda:
        return _torch_reference(inputs)

    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=add_tensor.device, dtype=torch.float32)
    spatial_mean = torch.empty_strided(
        SPATIAL_MEAN_SHAPE,
        SPATIAL_MEAN_STRIDE,
        device=add_tensor.device,
        dtype=torch.float32,
    )
    centered = torch.empty_strided(NCHW_SHAPE, NCHW_STRIDE, device=add_tensor.device, dtype=torch.float32)

    _bn_centered_spatial_mean_kernel[(CHANNELS,)](
        add_tensor,
        convolution,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        spatial_mean,
        centered,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=FUSED_BLOCK_N,
        BLOCK_HW=FUSED_BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    return invstd, spatial_mean, centered, running_mean, running_var


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
