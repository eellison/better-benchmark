"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full captured GhostNet training-BatchNorm scope by reducing per-channel variance and mean over f32[512,80,7,7], updating both running-stat copy_ aliases, returning invstd and mean side outputs, and writing the final f32[512,160,7,7] concat-plus-residual tensor directly, whereas Inductor currently schedules the var_mean/running-stat update, broadcast affine normalization, channel cat, and residual add as generic producer-consumer work around intermediate tensors; Inductor cannot do this today because its norm-template scheduler does not preserve mutable BN side outputs while fusing the following fixed channel concat and residual add into the normalization epilogue; the fix is SCHEDULER_FUSION: teach the BN-training template to expose mean/invstd/running-stat epilogues and sink static channel-cat plus residual-add stores into the same full-scope schedule."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


N = 512
CHANNELS = 80
OUT_CHANNELS = 160
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
OUTPUT_NUMEL = N * OUT_CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (N, OUT_CHANNELS, HEIGHT, WIDTH)
OUTPUT_STRIDE = (OUT_CHANNELS * HW, HW, WIDTH, 1)

PARTIAL_BLOCK = 1024
PARTIAL_BLOCKS = (ELEMENTS_PER_CHANNEL + PARTIAL_BLOCK - 1) // PARTIAL_BLOCK
FINAL_BLOCK = 32
OUTPUT_BLOCK = 256

if triton is not None:

    @triton.jit
    def _bn_partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sum2_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_id = tl.program_id(1)
        r = block_id * BLOCK + tl.arange(0, BLOCK)
        mask = r < elements_per_channel
        n = r // hw_size
        hw = r - n * hw_size
        x_offsets = n * channels * hw_size + channel * hw_size + hw
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        partial_offset = block_id * channels + channel
        tl.store(partial_sum_ptr + partial_offset, sum_x)
        tl.store(partial_sum2_ptr + partial_offset, sum_x2)

    @triton.jit
    def _bn_finalize_update_kernel(
        partial_sum_ptr,
        partial_sum2_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        partial_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK)
        mask = block_offsets < partial_blocks
        partial_offsets = block_offsets * channels + channel
        partial_sum = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        partial_sum2 = tl.load(partial_sum2_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(partial_sum, axis=0)
        sum_x2 = tl.sum(partial_sum2, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _cat_residual_output_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        skip_ptr,
        residual_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        hw = offsets % hw_size
        channel_out = (offsets // hw_size) % out_channels
        n = offsets // (out_channels * hw_size)

        first_half = channel_out < channels
        channel_skip = tl.where(first_half, channel_out, 0)
        channel_bn = tl.where(first_half, 0, channel_out - channels)

        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        skip_offsets = n * channels * hw_size + channel_skip * hw_size + hw
        skip = tl.load(skip_ptr + skip_offsets, mask=mask & first_half, other=0.0).to(tl.float32)

        bn_offsets = n * channels * hw_size + channel_bn * hw_size + hw
        x = tl.load(x_ptr + bn_offsets, mask=mask & ~first_half, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_bn, mask=mask & ~first_half, other=0.0).to(tl.float32)
        bn = (x - mean) * invstd * weight + bias

        value = tl.where(first_half, skip, bn) + residual
        tl.store(out_ptr + offsets, value, mask=mask)


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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    convolution_92, arg499_1, arg500_1, arg501_1, arg502_1, add_411, add_395 = inputs
    x = _expect_f32_tensor("convolution_92", convolution_92, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg499_1", arg499_1, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("arg500_1", arg500_1, VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("arg501_1", arg501_1, VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("arg502_1", arg502_1, VECTOR_SHAPE, VECTOR_STRIDE)
    skip = _expect_f32_tensor("add_411", add_411, INPUT_SHAPE, INPUT_STRIDE)
    residual = _expect_f32_tensor("add_395", add_395, OUTPUT_SHAPE, OUTPUT_STRIDE)

    if any(t.device != x.device for t in (running_mean, running_var, weight, bias, skip, residual)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias, skip, residual


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
        raise RuntimeError("Triton is required for this oracle")

    x, running_mean, running_var, weight, bias, skip, residual = _validate_inputs(inputs)
    mean = torch.empty(VECTOR_SHAPE, device=x.device, dtype=torch.float32)
    invstd = torch.empty(VECTOR_SHAPE, device=x.device, dtype=torch.float32)
    partial_sum = torch.empty((PARTIAL_BLOCKS, CHANNELS), device=x.device, dtype=torch.float32)
    partial_sum2 = torch.empty((PARTIAL_BLOCKS, CHANNELS), device=x.device, dtype=torch.float32)
    output = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)

    _bn_partial_stats_kernel[(CHANNELS, PARTIAL_BLOCKS)](
        x,
        partial_sum,
        partial_sum2,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        BLOCK=PARTIAL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_finalize_update_kernel[(CHANNELS,)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        partial_blocks=PARTIAL_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=FINAL_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _cat_residual_output_kernel[(triton.cdiv(OUTPUT_NUMEL, OUTPUT_BLOCK),)](
        x,
        weight,
        bias,
        skip,
        residual,
        mean,
        invstd,
        output,
        total=OUTPUT_NUMEL,
        channels=CHANNELS,
        out_channels=OUT_CHANNELS,
        hw_size=HW,
        BLOCK=OUTPUT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return invstd, output, mean.view(1, CHANNELS, 1, 1), running_mean, running_var


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
