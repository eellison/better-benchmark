"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured ResNet training-BatchNorm plus residual-ReLU scope in one channel kernel, returning the BN rsqrt side output, spatial ReLU mean view, le mask, squeezed mean view, and both in-place running-stat copy_ outputs while avoiding a materialized normalized activation, whereas Inductor currently schedules the BN-training var_mean/update template and the residual ReLU spatial-mean/mask consumers as separate regions with exposed intermediate activation traffic; Inductor cannot do this today because its scheduler cannot fuse a normalization reduction template into multiple downstream pointwise and reduction consumers while also preserving mutable copy_ side outputs and sibling stat returns; the fix is SCHEDULER_FUSION: teach the BN-training scheduler template to expose mean/rsqrt/running-stat outputs and sink affine residual ReLU, le, and spatial-mean consumers into the same channel-tiled reduction schedule."""
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


N = 128
CHANNELS = 64
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _bn_training_residual_relu_mean_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        invstd_out_ptr,
        spatial_mean_out_ptr,
        le_mask_out_ptr,
        mean_out_ptr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N)
        hw_offsets = tl.arange(0, BLOCK_HW)
        offsets = (n_offsets[:, None] * channels + channel) * hw + hw_offsets[None, :]

        x = tl.load(x_ptr + offsets).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum2 = tl.sum(x * x, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x2 = tl.sum(row_sum2, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(invstd_out_ptr + channel, invstd)
        tl.store(mean_out_ptr + channel, mean)

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        residual = tl.load(residual_ptr + offsets).to(tl.float32)
        pre_relu = (x - mean) * invstd * weight + bias + residual
        relu = tl.maximum(pre_relu, 0.0)
        pooled = tl.sum(relu, axis=1) * 0.015625

        tl.store(spatial_mean_out_ptr + n_offsets * channels + channel, pooled)
        tl.store(le_mask_out_ptr + offsets, relu <= 0.0)


def _expect_tensor(
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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    convolution_20, arg115_1, arg116_1, arg117_1, arg118_1, relu_16, shape_param = inputs
    nchw_shape = (N, CHANNELS, HEIGHT, WIDTH)
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    x = _expect_tensor("convolution_20", convolution_20, nchw_shape, nchw_stride)
    running_mean = _expect_tensor("arg115_1", arg115_1, (CHANNELS,), (1,))
    running_var = _expect_tensor("arg116_1", arg116_1, (CHANNELS,), (1,))
    weight = _expect_tensor("arg117_1", arg117_1, (CHANNELS,), (1,))
    bias = _expect_tensor("arg118_1", arg118_1, (CHANNELS,), (1,))
    residual = _expect_tensor("relu_16", relu_16, nchw_shape, nchw_stride)

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, residual)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x, running_mean, running_var, weight, bias, residual


@oracle_impl(hardware="H100", shapes="(T([128, 64, 8, 8], f32), T([64], f32), T([64], f32), T([64], f32), T([64], f32), T([128, 64, 8, 8], f32), S([128, 64]))")
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
        raise RuntimeError("Triton is required for oracle_bn_training_relu_mean.py")

    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    spatial_mean = torch.empty_strided((N, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float32)
    le_mask = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=x.device,
        dtype=torch.bool,
    )
    mean_view = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=x.device, dtype=torch.float32)

    _bn_training_residual_relu_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        residual,
        invstd,
        spatial_mean,
        le_mask,
        mean_view,
        channels=CHANNELS,
        hw=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=N,
        BLOCK_HW=HW,
        num_warps=8,
        num_stages=3,
    )
    return invstd, spatial_mean, le_mask, mean_view, running_mean, running_var


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
