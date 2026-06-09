"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet training-BatchNorm-ReLU-cat-spatial-mean scope by reducing BatchNorm statistics with split per-channel reductions, updating both running-stat copy_ outputs in place, and producing the final `[512,960,1,1]` spatial mean directly from the original `relu_37` branch plus the normalized affine ReLU branch without materializing either the ReLU activation or the logical concat, whereas Inductor currently schedules the BatchNorm statistics/update region, affine ReLU, concatenation, and following spatial mean as generic fused regions around large intermediate tensors; Inductor cannot do this today because its norm-template scheduler does not preserve this producer/consumer chain across mutable BatchNorm side outputs and a downstream concat mean; the fix is SCHEDULER_FUSION: teach the BN-training template to expose running-stat side effects while fusing fixed-shape affine/ReLU consumers and logical channel-concat spatial reductions into one planned schedule."""
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
CHANNELS = 480
CAT_CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK = 1024
STAT_BLOCKS = 25
FINAL_BLOCK = 32
MEAN_BLOCK_N = 16
MEAN_BLOCK = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        channel = tl.program_id(1)
        offsets = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        part_sum = tl.sum(vals, axis=0)
        part_sum_sq = tl.sum(vals * vals, axis=0)
        out_offset = channel * stat_blocks + block_id
        tl.store(partial_ptr + out_offset, part_sum)
        tl.store(partial_ptr + channels * stat_blocks + out_offset, part_sum_sq)

    @triton.jit
    def _finalize_stats_kernel(
        partial_ptr,
        running_mean_ptr,
        running_var_ptr,
        invstd_out_ptr,
        mean_out_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_P)
        mask = offsets < stat_blocks
        base = channel * stat_blocks + offsets

        sums = tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_ptr + channels * stat_blocks + base, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(sums, axis=0)
        total_sum_sq = tl.sum(sums_sq, axis=0)
        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(invstd_out_ptr + channel, invstd)
        tl.store(mean_out_ptr + channel, mean)

    @triton.jit
    def _relu_spatial_mean_kernel(
        relu_ptr,
        out_ptr,
        n_size: tl.constexpr,
        channels: tl.constexpr,
        cat_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW)
        mask = (n_offsets[:, None] < n_size) & (hw_offsets[None, :] < hw_size)
        in_offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]
        vals = tl.load(relu_ptr + in_offsets, mask=mask, other=0.0).to(tl.float32)
        spatial_mean = tl.sum(vals, axis=1) / hw_size
        out_offsets = n_offsets * cat_channels + channel
        tl.store(out_ptr + out_offsets, spatial_mean, mask=n_offsets < n_size)

    @triton.jit
    def _bn_relu_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        invstd_ptr,
        bn_mean_ptr,
        out_ptr,
        n_size: tl.constexpr,
        channels: tl.constexpr,
        cat_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        channel = tl.program_id(1)
        hw_offsets = tl.arange(0, BLOCK_HW)
        mask = (n_offsets[:, None] < n_size) & (hw_offsets[None, :] < hw_size)

        x_offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(bn_mean_ptr + channel).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))
        y = tl.where(mask, y, 0.0)

        spatial_mean = tl.sum(y, axis=1) / hw_size
        out_offsets = n_offsets * cat_channels + channels + channel
        tl.store(out_ptr + out_offsets, spatial_mean, mask=n_offsets < n_size)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_88, arg483_1, arg484_1, arg485_1, arg486_1, relu_37 = inputs
    x_stride = (CHANNELS * HW, HW, WIDTH, 1)
    x = _expect_f32_tensor("convolution_88", convolution_88, (N, CHANNELS, HEIGHT, WIDTH), x_stride)
    running_mean = _expect_f32_tensor("arg483_1", arg483_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg484_1", arg484_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg485_1", arg485_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg486_1", arg486_1, (CHANNELS,), (1,))
    relu = _expect_f32_tensor("relu_37", relu_37, (N, CHANNELS, HEIGHT, WIDTH), x_stride)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, relu)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias, relu


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias, relu = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = torch.relu((x - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
    cat_mean = torch.cat((relu, y), dim=1).mean(dim=(2, 3), keepdim=True)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return cat_mean, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([512, 480, 7, 7], f32), T([480], f32), T([480], f32), T([480], f32), T([480], f32), T([512, 480, 7, 7], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation."""
    x, running_mean, running_var, weight, bias, relu = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    partial = torch.empty_strided(
        (2, CHANNELS, STAT_BLOCKS),
        (CHANNELS * STAT_BLOCKS, STAT_BLOCKS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    mean = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, CAT_CHANNELS, 1, 1),
        (CAT_CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(STAT_BLOCKS, CHANNELS)](
        x,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        BLOCK_M=STAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _finalize_stats_kernel[(CHANNELS,)](
        partial,
        running_mean,
        running_var,
        invstd,
        mean,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_P=FINAL_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    mean_grid = (triton.cdiv(N, MEAN_BLOCK_N), CHANNELS)
    _relu_spatial_mean_kernel[mean_grid](
        relu,
        out,
        n_size=N,
        channels=CHANNELS,
        cat_channels=CAT_CHANNELS,
        hw_size=HW,
        BLOCK_N=MEAN_BLOCK_N,
        BLOCK_HW=MEAN_BLOCK,
        num_warps=2,
        num_stages=3,
    )
    _bn_relu_spatial_mean_kernel[mean_grid](
        x,
        weight,
        bias,
        invstd,
        mean,
        out,
        n_size=N,
        channels=CHANNELS,
        cat_channels=CAT_CHANNELS,
        hw_size=HW,
        BLOCK_N=MEAN_BLOCK_N,
        BLOCK_HW=MEAN_BLOCK,
        num_warps=2,
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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
            shape_key = get_shape_key(inputs)
            if shape_key:
                print(f"Shape: {shape_key}")
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
