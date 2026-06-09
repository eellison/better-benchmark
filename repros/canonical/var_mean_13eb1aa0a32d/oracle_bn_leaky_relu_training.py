"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DCGAN training-BatchNorm leaky-ReLU scope in one per-channel Triton program, including population `var_mean` over `[1024,4,4]`, `rsqrt(var + 1e-5)`, the `[512]` invstd output, the `[1,512,1,1]` mean side output, both in-place running-stat `copy_` updates, and the full affine leaky-ReLU activation, whereas Inductor currently lowers the training-normalization statistics, mutable running-stat side effects, and activation epilogue through generic scheduled regions; Inductor cannot do this today because the normalization scheduler does not preserve this full activation consumer and mutable side-output updates as one compact per-channel training-BN schedule; the fix is SCHEDULER_FUSION: teach the training BatchNorm template to fuse affine activation epilogues while still emitting mean/invstd and running-stat side effects."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile importable without Triton.
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


BATCH = 1024
CHANNELS = 512
HEIGHT = 4
WIDTH = 4
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
TOTAL_ELEMENTS = BATCH * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000610388817677
LEAKY_RELU_SLOPE = 0.2
BLOCK_M = 16384
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
MEAN_VIEW_SHAPE = (1, CHANNELS, 1, 1)
MEAN_VIEW_STRIDE = (CHANNELS, 1, 1, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _bn_leaky_relu_train_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_out_ptr,
        y_out_ptr,
        mean_out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        leaky_relu_slope: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        channel = tl.program_id(0)
        r = tl.arange(0, BLOCK_N)
        mask = r < elements_per_channel
        n = r // hw_size
        hw = r - n * hw_size
        offsets = (n * channels + channel) * hw_size + hw

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        total = elements_per_channel + 0.0
        mean = sum_x / total
        var = sum_x2 / total - mean * mean
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
        y = (x - mean) * invstd
        y = y * weight + bias
        y = tl.where(y > 0.0, y, y * leaky_relu_slope)
        tl.store(y_out_ptr + offsets, y, mask=mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    convolution_3, arg16_1, arg17_1, arg18_1, arg19_1 = inputs
    x = _expect_f32_tensor("convolution_3", convolution_3, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg16_1", arg16_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg17_1", arg17_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg18_1", arg18_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg19_1", arg19_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd_4d = torch.rsqrt(var + EPS)
    y = (x - mean) * invstd_4d
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    y = torch.where(y > 0, y, y * LEAKY_RELU_SLOPE)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    invstd_1d = invstd_4d.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return invstd_1d, y, mean_1d[None, :, None, None], running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([1024, 512, 4, 4], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    The returned running mean and variance are the same input tensors after the
    in-place `copy_` updates, matching the captured training-BN side effects.
    """
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    y = torch.empty_strided(INPUT_SHAPE, INPUT_STRIDE, device=x.device, dtype=torch.float32)
    mean = torch.empty_strided(MEAN_VIEW_SHAPE, MEAN_VIEW_STRIDE, device=x.device, dtype=torch.float32)

    _bn_leaky_relu_train_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        y,
        mean,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        leaky_relu_slope=LEAKY_RELU_SLOPE,
        BLOCK_N=BLOCK_M,
        num_warps=8,
        num_stages=3,
    )
    return invstd, y, mean, running_mean, running_var


# --- CLI entry point ---
def main() -> None:
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
    _ = get_shape_key(inputs)

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
