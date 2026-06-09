"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete ShuffleNetV2 training-BatchNorm var_mean, running-stat copy_ side effects, affine ReLU, and returned `[512,1024]` spatial mean with Triton statistics plus fused normalize/ReLU/pool epilogue kernels while preserving the in-place running-mean and running-var output aliases, whereas Inductor already emits a comparable schedule and is slightly faster on the required harness; Inductor cannot materially improve this local repro through a narrower norm-template change because the remaining work is mandatory stat reads/reductions, running-stat writes, affine activation, spatial mean, and output traffic rather than avoidable intermediate materialization; the fix is BANDWIDTH_BOUND: record this as at floor, with no targeted scheduler fix for this repro."""
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


N = 512
CHANNELS = 1024
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
TOTAL_ROWS = N * CHANNELS
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (N, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
STATS_XBLOCK = 2
STATS_RBLOCK = 1024
OUTPUT_XBLOCK = 64
OUTPUT_RBLOCK = 64


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _stats_update_kernel(
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
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        r_offsets = tl.arange(0, RBLOCK)
        sum_acc = tl.zeros([XBLOCK], tl.float32)
        sum_sq_acc = tl.zeros([XBLOCK], tl.float32)

        for r_start in tl.range(0, elements_per_channel, RBLOCK):
            r_index = r_start + r_offsets
            hw = r_index % hw_size
            n_index = r_index // hw_size
            mask = (c_offsets[:, None] < channels) & (r_index[None, :] < elements_per_channel)
            offsets = (n_index[None, :] * channels + c_offsets[:, None]) * hw_size + hw[None, :]
            x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
            sum_acc += tl.sum(x, axis=1)
            sum_sq_acc += tl.sum(x * x, axis=1)

        mean = sum_acc / elements_per_channel
        var = sum_sq_acc / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)
        c_mask = c_offsets < channels

        old_running_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(stats_ptr + c_offsets, mean, mask=c_mask)
        tl.store(stats_ptr + channels + c_offsets, invstd, mask=c_mask)
        tl.store(
            running_mean_ptr + c_offsets,
            old_running_mean * (1.0 - momentum) + mean * momentum,
            mask=c_mask,
        )
        tl.store(
            running_var_ptr + c_offsets,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask,
        )

    @triton.jit
    def _relu_spatial_mean_kernel(
        x_ptr,
        stats_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        total_rows: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
        hw_offsets = tl.arange(0, RBLOCK)
        row_mask = row_offsets < total_rows
        channel = row_offsets % channels
        n_index = row_offsets // channels
        hw_mask = hw_offsets < hw_size
        load_mask = row_mask[:, None] & hw_mask[None, :]
        x_offsets = (n_index[:, None] * channels + channel[:, None]) * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + x_offsets, mask=load_mask, other=0.0).to(tl.float32)
        mean = tl.load(stats_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * invstd[:, None]
        y = y * weight[:, None] + bias[:, None]
        relu = tl.where(y != y, y, tl.maximum(y, 0.0))
        pooled = tl.sum(tl.where(load_mask, relu, 0.0), axis=1) / hw_size
        tl.store(out_ptr + row_offsets, pooled, mask=row_mask)


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_55, arg333_1, arg334_1, arg335_1, arg336_1 = inputs
    x = _expect_f32_tensor("convolution_55", convolution_55, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg333_1", arg333_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg334_1", arg334_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg335_1", arg335_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg336_1", arg336_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([512, 1024, 7, 7], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32))")
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
        raise RuntimeError("Triton is required for oracle_bn_training_relu_spatial_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    stats = torch.empty_strided((2, CHANNELS), (CHANNELS, 1), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)

    _stats_update_kernel[(triton.cdiv(CHANNELS, STATS_XBLOCK),)](
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
        XBLOCK=STATS_XBLOCK,
        RBLOCK=STATS_RBLOCK,
        num_warps=16,
        num_stages=1,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(TOTAL_ROWS, OUTPUT_XBLOCK),)](
        x,
        stats,
        weight,
        bias,
        output,
        channels=CHANNELS,
        hw_size=HW,
        total_rows=TOTAL_ROWS,
        XBLOCK=OUTPUT_XBLOCK,
        RBLOCK=OUTPUT_RBLOCK,
        num_warps=2,
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
