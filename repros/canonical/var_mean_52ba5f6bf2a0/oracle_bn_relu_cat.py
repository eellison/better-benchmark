"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full VoVNet training-BatchNorm, running-stat update, affine ReLU, and channel-concat scope with Triton kernels that write the normalized branch directly into the final contiguous `[32,2144,7,7]` concat output while copying the other concat inputs only once, whereas tuned Inductor already lands in the same launch and memory-traffic envelope for this fixed training-BN plus concat shape; Inductor cannot materially improve this repro through a local norm-template canonicalization change because the remaining time is dominated by the required concat-output writes, input copies, per-channel statistics reads, and running-stat stores; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope normalization case unless broader concat placement, memory-traffic, or launch-overhead improvements move the baseline."""
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


# --- Oracle kernel(s) ---
BATCH = 32
BN_CHANNELS = 224
BIG_CHANNELS = 1024
SMALL_CAT_INPUTS = 4
PREFIX_CHANNELS = BIG_CHANNELS + SMALL_CAT_INPUTS * BN_CHANNELS
OUT_CHANNELS = PREFIX_CHANNELS + BN_CHANNELS
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
BIG_ELEMENTS = BATCH * BIG_CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0006381620931717
BN_BLOCK = 2048
COPY_BLOCK = 2048
OUT_SHAPE = (BATCH, OUT_CHANNELS, HEIGHT, WIDTH)
OUT_STRIDE = (OUT_CHANNELS * HW, HW, WIDTH, 1)

if triton is not None:

    @triton.jit
    def _copy_big_prefix_kernel(
        relu_32_ptr,
        out_ptr,
        out_channels: tl.constexpr,
        big_channels: tl.constexpr,
        hw_size: tl.constexpr,
        total_elements: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_elements
        elements_per_batch = big_channels * hw_size
        n_idx = offsets // elements_per_batch
        rem = offsets - n_idx * elements_per_batch
        out_channel = rem // hw_size
        hw_idx = rem - out_channel * hw_size
        out_offsets = (n_idx * out_channels + out_channel) * hw_size + hw_idx
        src_ptr = relu_32_ptr + (n_idx * big_channels + out_channel) * hw_size + hw_idx
        values = tl.load(src_ptr, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, values, mask=mask)

    @triton.jit
    def _bn_relu_cat_tail_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        relu_33_ptr,
        relu_34_ptr,
        relu_35_ptr,
        relu_36_ptr,
        out_ptr,
        bn_channels: tl.constexpr,
        big_channels: tl.constexpr,
        out_channels: tl.constexpr,
        tail_channel_offset: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_M)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * bn_channels + channel) * hw_size + hw_idx

        relu_33 = tl.load(relu_33_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        relu_34 = tl.load(relu_34_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        relu_35 = tl.load(relu_35_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        relu_36 = tl.load(relu_36_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        small_out = (n_idx * out_channels + big_channels + channel) * hw_size + hw_idx
        tl.store(out_ptr + small_out, relu_33, mask=mask)
        tl.store(out_ptr + small_out + bn_channels * hw_size, relu_34, mask=mask)
        tl.store(out_ptr + small_out + 2 * bn_channels * hw_size, relu_35, mask=mask)
        tl.store(out_ptr + small_out + 3 * bn_channels * hw_size, relu_36, mask=mask)

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
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

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (vals - mean) * invstd
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))

        out_channel = tail_channel_offset + channel
        out_offsets = (n_idx * out_channels + out_channel) * hw_size + hw_idx
        tl.store(out_ptr + out_offsets, y, mask=mask)


def _expect_f32_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    stride: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value).__name__}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    bn_shape = (BATCH, BN_CHANNELS, HEIGHT, WIDTH)
    bn_stride = (BN_CHANNELS * HW, HW, WIDTH, 1)
    big_shape = (BATCH, BIG_CHANNELS, HEIGHT, WIDTH)
    big_stride = (BIG_CHANNELS * HW, HW, WIDTH, 1)
    stat_shape = (BN_CHANNELS,)
    stat_stride = (1,)

    convolution_37 = _expect_f32_tensor("convolution_37", inputs[0], bn_shape, bn_stride)
    running_mean = _expect_f32_tensor("arg225_1", inputs[1], stat_shape, stat_stride)
    running_var = _expect_f32_tensor("arg226_1", inputs[2], stat_shape, stat_stride)
    weight = _expect_f32_tensor("arg227_1", inputs[3], stat_shape, stat_stride)
    bias = _expect_f32_tensor("arg228_1", inputs[4], stat_shape, stat_stride)
    relu_32 = _expect_f32_tensor("relu_32", inputs[5], big_shape, big_stride)
    relu_33 = _expect_f32_tensor("relu_33", inputs[6], bn_shape, bn_stride)
    relu_34 = _expect_f32_tensor("relu_34", inputs[7], bn_shape, bn_stride)
    relu_35 = _expect_f32_tensor("relu_35", inputs[8], bn_shape, bn_stride)
    relu_36 = _expect_f32_tensor("relu_36", inputs[9], bn_shape, bn_stride)

    device = convolution_37.device
    for tensor in (
        running_mean,
        running_var,
        weight,
        bias,
        relu_32,
        relu_33,
        relu_34,
        relu_35,
        relu_36,
    ):
        if tensor.device != device:
            raise ValueError("all tensor inputs must be on the same device")

    return (
        convolution_37,
        running_mean,
        running_var,
        weight,
        bias,
        relu_32,
        relu_33,
        relu_34,
        relu_35,
        relu_36,
    )


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    (
        convolution_37,
        running_mean,
        running_var,
        weight,
        bias,
        relu_32,
        relu_33,
        relu_34,
        relu_35,
        relu_36,
    ) = _validate_inputs(inputs)

    var, mean = torch.ops.aten.var_mean.correction(
        convolution_37,
        [0, 2, 3],
        correction=0,
        keepdim=True,
    )
    invstd = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    normalized = torch.ops.aten.mul.Tensor(
        torch.ops.aten.sub.Tensor(convolution_37, mean),
        invstd,
    )
    affine = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(normalized, weight[None, :, None, None]),
        bias[None, :, None, None],
    )
    relu = torch.ops.aten.relu.default(affine)
    out = torch.ops.aten.cat.default([relu_32, relu_33, relu_34, relu_35, relu_36, relu], 1)

    mean_1d = torch.ops.aten.squeeze.dims(mean, [0, 2, 3])
    var_1d = torch.ops.aten.squeeze.dims(var, [0, 2, 3])
    running_mean.copy_(mean_1d * MOMENTUM + running_mean * (1.0 - MOMENTUM))
    running_var.copy_(var_1d * RUNNING_VAR_CORRECTION * MOMENTUM + running_var * (1.0 - MOMENTUM))
    return out, running_mean, running_var


def oracle_forward(inputs):
    """Run the full Repro.forward training-BN, running-stat update, ReLU, and concat scope.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    (
        convolution_37,
        running_mean,
        running_var,
        weight,
        bias,
        relu_32,
        relu_33,
        relu_34,
        relu_35,
        relu_36,
    ) = _validate_inputs(inputs)

    if triton is None or not convolution_37.is_cuda:
        return _torch_reference(inputs)

    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=convolution_37.device,
        dtype=torch.float32,
    )
    _copy_big_prefix_kernel[(triton.cdiv(BIG_ELEMENTS, COPY_BLOCK),)](
        relu_32,
        out,
        out_channels=OUT_CHANNELS,
        big_channels=BIG_CHANNELS,
        hw_size=HW,
        total_elements=BIG_ELEMENTS,
        BLOCK_M=COPY_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _bn_relu_cat_tail_kernel[(BN_CHANNELS,)](
        convolution_37,
        running_mean,
        running_var,
        weight,
        bias,
        relu_33,
        relu_34,
        relu_35,
        relu_36,
        out,
        bn_channels=BN_CHANNELS,
        big_channels=BIG_CHANNELS,
        out_channels=OUT_CHANNELS,
        tail_channel_offset=PREFIX_CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_M=BN_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return out, running_mean, running_var


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
