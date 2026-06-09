"""
Oracle for var_mean_765fb8f2c85e: BN training forward + affine + ReLU + MaxPool + AvgPool.

Repro pattern core:
    x[N, C, H, W] -> BN training stats -> running stats update -> affine -> ReLU
    -> max_pool -> avg_pool (full Repro scope)

Inputs: convolution_4[128,192,71,71], arg27_1[192], arg28_1[192], arg29_1[192], arg30_1[192]
Outputs: (pool_offsets i8[128,192,35,35], avg_pool f32[128,192,35,35],
          new_running_mean f32[192], new_running_var f32[192])

The oracle fuses BN stats + affine + ReLU using Triton kernels, then applies
the max_pool + avg_pool tail via PyTorch ops. Running stats are updated in-place
via copy_ to match the repro scope contract exactly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

# --- Configuration (auto-derived from file location) ---
REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)

# Default shape constants
DEFAULT_EPS = 0.001
DEFAULT_MOMENTUM = 0.1


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _stats_nchw_kernel(
        x_ptr,
        mean_ptr,
        var_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        block: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, block)
        mask = offsets < elems_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        flat_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
        vals = tl.load(x_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        tl.store(mean_ptr + channel, mean)
        tl.store(var_ptr + channel, tl.maximum(var, 0.0))


    @triton.jit
    def _running_stats_kernel(
        mean_ptr,
        var_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_running_mean_ptr,
        out_running_var_ptr,
        channels: tl.constexpr,
        elems_per_channel: tl.constexpr,
        momentum: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.arange(0, block)
        mask = offsets < channels
        mean = tl.load(mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean = tl.load(running_mean_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        correction = elems_per_channel / (elems_per_channel - 1.0)
        tl.store(out_running_mean_ptr + offsets, old_mean * (1.0 - momentum) + mean * momentum, mask=mask)
        tl.store(out_running_var_ptr + offsets, old_var * (1.0 - momentum) + var * correction * momentum, mask=mask)


    @triton.jit
    def _affine_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        var_ptr,
        y_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        block: tl.constexpr,
    ):
        program = tl.program_id(0)
        offsets = program * block + tl.arange(0, block)
        mask = offsets < total
        channel = (offsets // hw_size) % channels
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * tl.rsqrt(var + eps) * weight + bias
        tl.store(y_ptr + offsets, tl.maximum(y, 0.0), mask=mask)


def _triton_bn_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton BN training forward + running stats + affine + ReLU."""
    n_batches, channels, height, width = x.shape
    hw_size = height * width
    elems_per_channel = n_batches * hw_size
    block = triton.next_power_of_2(elems_per_channel)

    mean = torch.empty((channels,), device=x.device, dtype=torch.float32)
    var = torch.empty((channels,), device=x.device, dtype=torch.float32)
    y = torch.empty_like(x)
    new_running_mean = torch.empty_like(running_mean)
    new_running_var = torch.empty_like(running_var)

    _stats_nchw_kernel[(channels,)](
        x, mean, var,
        channels=channels, hw_size=hw_size,
        elems_per_channel=elems_per_channel, block=block,
        num_warps=8,
    )
    _running_stats_kernel[(1,)](
        mean, var, running_mean, running_var,
        new_running_mean, new_running_var,
        channels=channels, elems_per_channel=elems_per_channel,
        momentum=momentum, block=triton.next_power_of_2(channels),
        num_warps=8,
    )
    affine_block = 256
    grid = (triton.cdiv(x.numel(), affine_block),)
    _affine_relu_kernel[grid](
        x, weight, bias, mean, var, y,
        total=x.numel(), channels=channels, hw_size=hw_size,
        eps=eps, block=affine_block, num_warps=4,
    )
    return y, new_running_mean, new_running_var


def _torch_bn_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = DEFAULT_EPS,
    momentum: float = DEFAULT_MOMENTUM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch BN training forward + affine + ReLU (fallback)."""
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    y = torch.relu((x - mean) * inv_std * weight[None, :, None, None] + bias[None, :, None, None])

    elems_per_channel = x.shape[0] * x.shape[2] * x.shape[3]
    correction = elems_per_channel / (elems_per_channel - 1)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    new_running_mean = running_mean * (1.0 - momentum) + mean_1d * momentum
    new_running_var = running_var * (1.0 - momentum) + var_1d * correction * momentum
    return y, new_running_mean, new_running_var


def oracle_forward(inputs):
    """Run the full Repro.forward scope: BN + affine + ReLU + MaxPool + AvgPool.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Inputs: [convolution_4, arg27_1, arg28_1, arg29_1, arg30_1]
    Returns: (pool_offsets, avg_pool, copy_running_mean, copy_running_var)
    """
    convolution_4, arg27_1, arg28_1, arg29_1, arg30_1 = inputs

    # BN + affine + ReLU
    if triton is not None and convolution_4.is_cuda and convolution_4.is_contiguous():
        relu_out, new_running_mean, new_running_var = _triton_bn_relu(
            convolution_4, arg27_1, arg28_1, arg29_1, arg30_1,
        )
    else:
        relu_out, new_running_mean, new_running_var = _torch_bn_relu(
            convolution_4, arg27_1, arg28_1, arg29_1, arg30_1,
        )

    # MaxPool: kernel=3, stride=2, padding=0, dilation=1, ceil_mode=False
    pool_result = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu_out, [3, 3], [2, 2], [0, 0], [1, 1], False
    )
    pool_offsets = pool_result[1]
    pool_values = pool_result[0]

    # AvgPool: kernel=3, stride=1, padding=1
    avg_pool = torch.ops.aten.avg_pool2d.default(pool_values, [3, 3], [1, 1], [1, 1])

    # Running stats copy_ (matching repro mutation contract)
    arg27_1.copy_(new_running_mean)
    arg28_1.copy_(new_running_var)

    return (pool_offsets, avg_pool, arg27_1, arg28_1)


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
