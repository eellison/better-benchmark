"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 training-BatchNorm var_mean, mutable running-stat copy_ updates, affine hard-swish, final 7x7 spatial mean, and the captured 4D as_strided output with a two-kernel Triton schedule that carries only per-channel mean/invstd between the statistics update and spatial-mean epilogue, whereas Inductor currently lowers the same scope through generic normalization and downstream spatial-reduction schedules; Inductor cannot do this today because norm-template scheduling does not keep training-BN side-effect returns and an immediate hard-swish spatial-mean consumer inside one reusable full-scope plan; the fix is SCHEDULER_FUSION: add a BN-training template that exposes mean/invstd/running-stat epilogues while sinking affine hard-swish plus fixed 7x7 spatial mean into the specialized lowering."""
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
N = 512
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
TOTAL_ROWS = N * CHANNELS
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK_R = 16384
OUTPUT_BLOCK_ROWS = 32
BLOCK_HW = 64

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (N, CHANNELS, 1, 1)
OUTPUT_STRIDE = (CHANNELS, 1, CHANNELS, CHANNELS)

if triton is not None:

    @triton.jit
    def _channel_stats_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_R_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        r_offsets = tl.arange(0, BLOCK_R_)
        sum_x = tl.full((), 0.0, tl.float32)
        sum_x2 = tl.full((), 0.0, tl.float32)

        for r_base in tl.range(0, elements_per_channel, BLOCK_R_):
            r = r_base + r_offsets
            r_mask = r < elements_per_channel
            hw = r - (r // hw_size) * hw_size
            n = r // hw_size
            x_offsets = n * channels * hw_size + channel * hw_size + hw
            x = tl.load(x_ptr + x_offsets, mask=r_mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.where(var < 0.0, 0.0, var)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(
            running_mean_ptr + channel,
            old_running_mean * (1.0 - momentum) + mean * momentum,
        )
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum)
            + var * running_var_correction * momentum,
        )
        tl.store(stats_ptr + channel, mean)
        tl.store(stats_ptr + channels + channel, invstd)

    @triton.jit
    def _hardswish_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS_ + tl.arange(0, BLOCK_ROWS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        hw_mask = hw_offsets < hw_size
        channel = rows - (rows // channels) * channels

        x = tl.load(
            x_ptr + rows[:, None] * hw_size + hw_offsets[None, :],
            mask=row_mask[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(stats_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

        affine = (x - mean[:, None]) * invstd[:, None]
        affine = affine * weight[:, None] + bias[:, None]
        relu6 = affine + 3.0
        relu6 = tl.where(relu6 < 0.0, 0.0, relu6)
        relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
        hardswish = affine * relu6 * 0.16666666666666666
        pooled = tl.sum(tl.where(row_mask[:, None] & hw_mask[None, :], hardswish, 0.0), axis=1)
        pooled = pooled * (1.0 / 49.0)
        tl.store(out_ptr + rows, pooled, mask=row_mask)


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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_61, arg305_1, arg306_1, arg307_1, arg308_1 = inputs
    x = _expect_f32_tensor("convolution_61", convolution_61, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg305_1", arg305_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg306_1", arg306_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg307_1", arg307_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg308_1", arg308_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    affine = (x - mean) * invstd
    affine = affine * weight[None, :, None, None] + bias[None, :, None, None]
    relu6 = torch.clamp_max(torch.clamp_min(affine + 3.0, 0.0), 6.0)
    out = (affine * relu6 * (1.0 / 6.0)).mean(dim=(-1, -2), keepdim=True)
    out = torch.as_strided(out, OUTPUT_SHAPE, OUTPUT_STRIDE)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(
        running_var * (1.0 - MOMENTUM)
        + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return out, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([512, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32), T([960], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _channel_stats_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        stats,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_R_=STAT_BLOCK_R,
        num_warps=8,
        num_stages=1,
    )
    _hardswish_spatial_mean_kernel[(triton.cdiv(TOTAL_ROWS, OUTPUT_BLOCK_ROWS),)](
        x,
        weight,
        bias,
        stats,
        output,
        total_rows=TOTAL_ROWS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_ROWS_=OUTPUT_BLOCK_ROWS,
        BLOCK_HW_=BLOCK_HW,
        num_warps=1,
        num_stages=1,
    )
    return output, running_mean, running_var


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
