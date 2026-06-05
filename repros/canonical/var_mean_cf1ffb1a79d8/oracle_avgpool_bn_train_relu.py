"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle folds the fixed 2x2 stride-2 avg_pool2d into the channel training-BatchNorm reduction, updates the running mean/variance copy_ outputs in place, returns invstd and mean side outputs, and writes the affine ReLU tensor in one Triton launch without materializing the pooled tensor, whereas Inductor currently schedules avg_pool2d as a separate producer feeding the BN-training var_mean/update and activation work; Inductor cannot do this today because scheduler fusion does not inline a structured pooling stencil into a multi-output normalization reduction with mutable running-stat returns and a full activation consumer; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to accept fixed-window pooling producers and emit the stats, copy_ side effects, and affine ReLU epilogue from one channel-tiled reduction plan."""
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
    has_stochastic_ops,
)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
N = 64
CHANNELS = 512
IN_HEIGHT = 14
IN_WIDTH = 14
IN_HW = IN_HEIGHT * IN_WIDTH
OUT_HEIGHT = 7
OUT_WIDTH = 7
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ELEMENTS_PER_CHANNEL = N * OUT_HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0003189792663476
BLOCK_K = 4096
INPUT_SHAPE = (N, CHANNELS, IN_HEIGHT, IN_WIDTH)
INPUT_STRIDE = (CHANNELS * IN_HW, IN_HW, IN_WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
RELU_SHAPE = (N, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
RELU_STRIDE = (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1)
MEAN_SHAPE = (1, CHANNELS, 1, 1)
MEAN_STRIDE = (CHANNELS, 1, 1, 1)

if triton is not None:

    @triton.jit
    def _avgpool_bn_train_relu_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_out_ptr,
        relu_out_ptr,
        mean_out_ptr,
        channels: tl.constexpr,
        in_width: tl.constexpr,
        in_hw: tl.constexpr,
        out_width: tl.constexpr,
        out_hw: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_K_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K_)
        mask = offsets < elements_per_channel

        n_idx = offsets // out_hw
        out_hw_idx = offsets - n_idx * out_hw
        out_h = out_hw_idx // out_width
        out_w = out_hw_idx - out_h * out_width

        input_base = n_idx * channels * in_hw + channel * in_hw + (out_h * 2) * in_width + out_w * 2
        x00 = tl.load(x_ptr + input_base, mask=mask, other=0.0).to(tl.float32)
        x01 = tl.load(x_ptr + input_base + 1, mask=mask, other=0.0).to(tl.float32)
        x10 = tl.load(x_ptr + input_base + in_width, mask=mask, other=0.0).to(tl.float32)
        x11 = tl.load(x_ptr + input_base + in_width + 1, mask=mask, other=0.0).to(tl.float32)
        pooled = (x00 + x01 + x10 + x11) * 0.25

        sum_x = tl.sum(pooled, axis=0)
        sum_x2 = tl.sum(pooled * pooled, axis=0)
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
        y = (pooled - mean) * invstd
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        relu_offsets = n_idx * channels * out_hw + channel * out_hw + out_hw_idx
        tl.store(relu_out_ptr + relu_offsets, y, mask=mask)


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
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x = _expect_f32_tensor("convolution_87", inputs[0], INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg530_1", inputs[1], STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg531_1", inputs[2], STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg532_1", inputs[3], STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg533_1", inputs[4], STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]):
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    pooled = torch.ops.aten.avg_pool2d.default(x, [2, 2], [2, 2])
    var, mean = torch.var_mean(pooled, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    relu = torch.relu((pooled - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * 0.9 + mean_1d * 0.1)
    running_var.copy_(running_var * 0.9 + var_1d * RUNNING_VAR_CORRECTION * 0.1)
    return invstd.squeeze((0, 2, 3)), relu, mean_1d[None, :, None, None], running_mean, running_var


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
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    relu = torch.empty_strided(RELU_SHAPE, RELU_STRIDE, device=x.device, dtype=torch.float32)
    mean = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x.device, dtype=torch.float32)

    _avgpool_bn_train_relu_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        relu,
        mean,
        channels=CHANNELS,
        in_width=IN_WIDTH,
        in_hw=IN_HW,
        out_width=OUT_WIDTH,
        out_hw=OUT_HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K_=BLOCK_K,
        num_warps=8,
        num_stages=3,
    )
    return invstd, relu, mean, running_mean, running_var


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
