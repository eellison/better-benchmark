"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full DenseNet training concat-BatchNorm-ReLU scope by reducing the logical [64,736,7,7] concatenation directly from the eight source tensors, updating the running mean/variance copy_ outputs in place, and writing the affine ReLU activation without materializing the cat, whereas Inductor currently treats the virtual cat, BN training statistics/update side effects, and activation epilogue as generic scheduled work with avoidable concat/indexing overhead; Inductor cannot do this today because its scheduler does not keep a multi-input channel concat virtual across a multi-output normalization reduction with mutable running-stat returns and a full activation consumer; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to accept fixed channel-concat producers, expose mean/invstd/running-stat side outputs, and fuse the affine ReLU epilogue into the same channel-tiled reduction plan."""
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
N = 64
C0 = 512
BRANCH_C = 32
BRANCHES = 7
CHANNELS = C0 + BRANCHES * BRANCH_C
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0003189792663476
BLOCK_K = 4096

if triton is not None:

    @triton.jit
    def _bn_train_cat_relu_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        x4_ptr,
        x5_ptr,
        x6_ptr,
        x7_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_out_ptr,
        relu_out_ptr,
        mean_out_ptr,
        c0: tl.constexpr,
        branch_c: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        rel = channel - c0
        in0 = channel < c0
        in1 = (rel >= 0) & (rel < branch_c)
        in2 = (rel >= branch_c) & (rel < 2 * branch_c)
        in3 = (rel >= 2 * branch_c) & (rel < 3 * branch_c)
        in4 = (rel >= 3 * branch_c) & (rel < 4 * branch_c)
        in5 = (rel >= 4 * branch_c) & (rel < 5 * branch_c)
        in6 = (rel >= 5 * branch_c) & (rel < 6 * branch_c)
        in7 = (rel >= 6 * branch_c) & (rel < 7 * branch_c)

        x0_channel = tl.where(in0, channel, 0)
        x1_channel = tl.where(in1, rel, 0)
        x2_channel = tl.where(in2, rel - branch_c, 0)
        x3_channel = tl.where(in3, rel - 2 * branch_c, 0)
        x4_channel = tl.where(in4, rel - 3 * branch_c, 0)
        x5_channel = tl.where(in5, rel - 4 * branch_c, 0)
        x6_channel = tl.where(in6, rel - 5 * branch_c, 0)
        x7_channel = tl.where(in7, rel - 6 * branch_c, 0)

        x0_offsets = (n_idx * c0 + x0_channel) * hw_size + hw_idx
        branch_base = n_idx * branch_c * hw_size + hw_idx
        x1_offsets = branch_base + x1_channel * hw_size
        x2_offsets = branch_base + x2_channel * hw_size
        x3_offsets = branch_base + x3_channel * hw_size
        x4_offsets = branch_base + x4_channel * hw_size
        x5_offsets = branch_base + x5_channel * hw_size
        x6_offsets = branch_base + x6_channel * hw_size
        x7_offsets = branch_base + x7_channel * hw_size

        vals = tl.load(x0_ptr + x0_offsets, mask=mask & in0, other=0.0).to(tl.float32)
        vals += tl.load(x1_ptr + x1_offsets, mask=mask & in1, other=0.0).to(tl.float32)
        vals += tl.load(x2_ptr + x2_offsets, mask=mask & in2, other=0.0).to(tl.float32)
        vals += tl.load(x3_ptr + x3_offsets, mask=mask & in3, other=0.0).to(tl.float32)
        vals += tl.load(x4_ptr + x4_offsets, mask=mask & in4, other=0.0).to(tl.float32)
        vals += tl.load(x5_ptr + x5_offsets, mask=mask & in5, other=0.0).to(tl.float32)
        vals += tl.load(x6_ptr + x6_offsets, mask=mask & in6, other=0.0).to(tl.float32)
        vals += tl.load(x7_ptr + x7_offsets, mask=mask & in7, other=0.0).to(tl.float32)

        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
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
        y = (vals - mean) * invstd
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        out_offsets = (n_idx * channels + channel) * hw_size + hw_idx
        tl.store(relu_out_ptr + out_offsets, y, mask=mask)


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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 12:
        raise ValueError(f"{REPRO_ID} expects 12 inputs, got {len(inputs)}")

    (
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        convolution_101,
        arg614_1,
        arg615_1,
        arg616_1,
        arg617_1,
    ) = inputs

    x0_stride = (C0 * HW, HW, WIDTH, 1)
    branch_stride = (BRANCH_C * HW, HW, WIDTH, 1)
    x0 = _expect_tensor("avg_pool2d_2", avg_pool2d_2, (N, C0, HEIGHT, WIDTH), x0_stride)
    branches = (
        _expect_tensor("convolution_89", convolution_89, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_91", convolution_91, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_93", convolution_93, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_95", convolution_95, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_97", convolution_97, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_99", convolution_99, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
        _expect_tensor("convolution_101", convolution_101, (N, BRANCH_C, HEIGHT, WIDTH), branch_stride),
    )
    running_mean = _expect_tensor("arg614_1", arg614_1, (CHANNELS,), (1,))
    running_var = _expect_tensor("arg615_1", arg615_1, (CHANNELS,), (1,))
    weight = _expect_tensor("arg616_1", arg616_1, (CHANNELS,), (1,))
    bias = _expect_tensor("arg617_1", arg617_1, (CHANNELS,), (1,))

    tensors = (x0, *branches, running_mean, running_var, weight, bias)
    device = x0.device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")

    return tensors


def _torch_reference(inputs: list[Any] | tuple[Any, ...]):
    (
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        convolution_101,
        arg614_1,
        arg615_1,
        arg616_1,
        arg617_1,
    ) = inputs
    cat = torch.cat(
        [
            avg_pool2d_2,
            convolution_89,
            convolution_91,
            convolution_93,
            convolution_95,
            convolution_97,
            convolution_99,
            convolution_101,
        ],
        dim=1,
    )
    var, mean = torch.var_mean(cat, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    relu = torch.relu((cat - mean) * invstd * arg616_1[None, :, None, None] + arg617_1[None, :, None, None])
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    arg614_1.copy_(arg614_1 * 0.9 + mean_1d * 0.1)
    arg615_1.copy_(arg615_1 * 0.9 + var_1d * RUNNING_VAR_CORRECTION * 0.1)
    return invstd.squeeze((0, 2, 3)), relu, mean_1d[None, :, None, None], arg614_1, arg615_1


@oracle_impl(hardware="H100", shapes="(T([64, 512, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([64, 32, 7, 7], f32), T([736], f32), T([736], f32), T([736], f32), T([736], f32))")
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
    tensors = _validate_inputs(inputs)
    if triton is None or not tensors[0].is_cuda:
        return _torch_reference(inputs)

    (
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        convolution_101,
        running_mean,
        running_var,
        weight,
        bias,
    ) = tensors

    invstd = torch.empty_strided((CHANNELS,), (1,), device=avg_pool2d_2.device, dtype=torch.float32)
    relu = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=avg_pool2d_2.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (1, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=avg_pool2d_2.device,
        dtype=torch.float32,
    )

    _bn_train_cat_relu_kernel[(CHANNELS,)](
        avg_pool2d_2,
        convolution_89,
        convolution_91,
        convolution_93,
        convolution_95,
        convolution_97,
        convolution_99,
        convolution_101,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        relu,
        mean,
        c0=C0,
        branch_c=BRANCH_C,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_K=BLOCK_K,
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
