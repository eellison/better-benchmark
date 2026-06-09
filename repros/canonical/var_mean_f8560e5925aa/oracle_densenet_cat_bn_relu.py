"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet training batchnorm over a virtual channel concat, returns invstd, affine ReLU, saved mean, and in-place running-stat copy_ outputs while reading each concat slice directly, whereas Inductor currently lowers the cat, var_mean, running-stat epilogues, and affine ReLU as separate scheduled work; Inductor cannot do this today because scheduler fusion does not model a mutable training-BN reduction over a virtual cat producer feeding a full activation consumer; the fix is SCHEDULER_FUSION: teach the scheduler a virtual-cat training-BN-affine-ReLU template with explicit running_mean/running_var copy_ epilogues."""
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
N_BATCH = 64
TOTAL_CHANNELS = 704
BASE_CHANNELS = 512
DENSE_CHANNELS = 32
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMS_PER_CHANNEL = N_BATCH * HW
EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.0003189792663476
BLOCK_M = 4096

BASE_SHAPE = (N_BATCH, BASE_CHANNELS, HEIGHT, WIDTH)
DENSE_SHAPE = (N_BATCH, DENSE_CHANNELS, HEIGHT, WIDTH)
OUTPUT_SHAPE = (N_BATCH, TOTAL_CHANNELS, HEIGHT, WIDTH)
BASE_STRIDE = (BASE_CHANNELS * HW, HW, WIDTH, 1)
DENSE_STRIDE = (DENSE_CHANNELS * HW, HW, WIDTH, 1)
OUTPUT_STRIDE = (TOTAL_CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (TOTAL_CHANNELS,)
STAT_STRIDE = (1,)


if triton is not None:

    @triton.jit
    def _bn_relu_virtual_cat_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        x5_ptr,
        x6_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_ptr,
        mean_ptr,
        relu_ptr,
        base_channels: tl.constexpr,
        dense_channels: tl.constexpr,
        total_channels: tl.constexpr,
        hw: tl.constexpr,
        elems_per_channel: tl.constexpr,
        eps: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        active = offsets < elems_per_channel

        batch = offsets // hw
        spatial = offsets - batch * hw

        in0 = channel < base_channels
        local0 = channel
        x0_offsets = batch * (base_channels * hw) + local0 * hw + spatial
        x = tl.load(x0_ptr + x0_offsets, mask=active & in0, other=0.0).to(tl.float32)

        in1 = (channel >= 512) & (channel < 544)
        local1 = channel - 512
        x1_offsets = batch * (dense_channels * hw) + local1 * hw + spatial
        x += tl.load(x1_ptr + x1_offsets, mask=active & in1, other=0.0).to(tl.float32)

        in2 = (channel >= 544) & (channel < 576)
        local2 = channel - 544
        x2_offsets = batch * (dense_channels * hw) + local2 * hw + spatial
        x += tl.load(x2_ptr + x2_offsets, mask=active & in2, other=0.0).to(tl.float32)

        in3 = (channel >= 576) & (channel < 608)
        local3 = channel - 576
        x3_offsets = batch * (dense_channels * hw) + local3 * hw + spatial
        x += tl.load(x3_ptr + x3_offsets, mask=active & in3, other=0.0).to(tl.float32)

        in4 = (channel >= 608) & (channel < 640)
        local4 = channel - 608
        x4_offsets = batch * (dense_channels * hw) + local4 * hw + spatial
        x += tl.load(x4_ptr + x4_offsets, mask=active & in4, other=0.0).to(tl.float32)

        in5 = (channel >= 640) & (channel < 672)
        local5 = channel - 640
        x5_offsets = batch * (dense_channels * hw) + local5 * hw + spatial
        x += tl.load(x5_ptr + x5_offsets, mask=active & in5, other=0.0).to(tl.float32)

        in6 = channel >= 672
        local6 = channel - 672
        x6_offsets = batch * (dense_channels * hw) + local6 * hw + spatial
        x += tl.load(x6_ptr + x6_offsets, mask=active & in6, other=0.0).to(tl.float32)

        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        tl.store(invstd_ptr + channel, invstd)
        tl.store(mean_ptr + channel, mean)
        tl.store(running_mean_ptr + channel, old_mean * 0.9 + mean * 0.1)
        tl.store(
            running_var_ptr + channel,
            old_var * 0.9 + var * running_var_correction * 0.1,
        )

        normalized = (x - mean) * invstd
        affine = normalized * weight + bias
        relu = tl.where((affine > 0.0) | (affine != affine), affine, 0.0)
        relu_offsets = batch * (total_channels * hw) + channel * hw + spatial
        tl.store(relu_ptr + relu_offsets, relu, mask=active)


def _require_f32_tensor(
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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects eleven inputs, got {len(inputs)}")

    tensors = (
        _require_f32_tensor("avg_pool2d_2", inputs[0], BASE_SHAPE, BASE_STRIDE),
        _require_f32_tensor("convolution_89", inputs[1], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("convolution_91", inputs[2], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("convolution_93", inputs[3], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("convolution_95", inputs[4], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("convolution_97", inputs[5], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("convolution_99", inputs[6], DENSE_SHAPE, DENSE_STRIDE),
        _require_f32_tensor("arg602_1", inputs[7], STAT_SHAPE, STAT_STRIDE),
        _require_f32_tensor("arg603_1", inputs[8], STAT_SHAPE, STAT_STRIDE),
        _require_f32_tensor("arg604_1", inputs[9], STAT_SHAPE, STAT_STRIDE),
        _require_f32_tensor("arg605_1", inputs[10], STAT_SHAPE, STAT_STRIDE),
    )

    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(
    avg_pool2d_2: torch.Tensor,
    convolution_89: torch.Tensor,
    convolution_91: torch.Tensor,
    convolution_93: torch.Tensor,
    convolution_95: torch.Tensor,
    convolution_97: torch.Tensor,
    convolution_99: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.cat(
        [
            avg_pool2d_2,
            convolution_89,
            convolution_91,
            convolution_93,
            convolution_95,
            convolution_97,
            convolution_99,
        ],
        dim=1,
    )
    var, mean_4d = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd_4d = torch.rsqrt(var + EPS)
    invstd = invstd_4d.squeeze((0, 2, 3))
    mean = mean_4d.squeeze((0, 2, 3))
    relu = torch.relu((x - mean_4d) * invstd_4d * weight.view(1, TOTAL_CHANNELS, 1, 1) + bias.view(1, TOTAL_CHANNELS, 1, 1))
    running_mean.copy_(running_mean * 0.9 + mean * 0.1)
    running_var.copy_(running_var * 0.9 + var.squeeze((0, 2, 3)) * RUNNING_VAR_CORRECTION * 0.1)
    return invstd, relu, mean.view(1, TOTAL_CHANNELS, 1, 1), running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([64, 512, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([704], f32), T([704], f32), T([704], f32), T([704], f32))")
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
    (
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        running_mean,
        running_var,
        weight,
        bias,
    ) = _validate_inputs(inputs)

    if not avg_pool2d_2.is_cuda:
        return _torch_oracle(
            avg_pool2d_2,
            convolution_89,
            convolution_91,
            convolution_93,
            convolution_95,
            convolution_97,
            convolution_99,
            running_mean,
            running_var,
            weight,
            bias,
        )
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    invstd = torch.empty_strided(
        STAT_SHAPE,
        STAT_STRIDE,
        device=avg_pool2d_2.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        STAT_SHAPE,
        STAT_STRIDE,
        device=avg_pool2d_2.device,
        dtype=torch.float32,
    )
    relu = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=avg_pool2d_2.device,
        dtype=torch.float32,
    )

    _bn_relu_virtual_cat_kernel[(TOTAL_CHANNELS,)](
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        mean,
        relu,
        base_channels=BASE_CHANNELS,
        dense_channels=DENSE_CHANNELS,
        total_channels=TOTAL_CHANNELS,
        hw=HW,
        elems_per_channel=ELEMS_PER_CHANNEL,
        eps=EPS,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_M,
        num_warps=8,
        num_stages=4,
    )

    return invstd, relu, mean.view(1, TOTAL_CHANNELS, 1, 1), running_mean, running_var


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
