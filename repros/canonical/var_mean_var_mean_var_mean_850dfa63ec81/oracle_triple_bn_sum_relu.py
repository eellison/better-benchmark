"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full RepVGG triple training-BatchNorm scope, including all three independent channel `var_mean` reductions over `[128,384,14,14]`, six running-stat `copy_` side effects, three invstd side outputs, three saved mean views, and the summed affine ReLU activation in two shape-specialized Triton passes, whereas tuned Inductor already lowers the same norm-template graph into the required channel-statistics and activation-store memory envelope; Inductor cannot materially improve this repro through a local scheduler fusion, scatter-reduce, split-K, algebraic-elimination, or recompute rewrite because the remaining work is dominated by mandatory reads of three activations for statistics, rereads for the output activation, running-stat/vector stores, and the full activation store; the fix is BANDWIDTH_BOUND: record this as an at-floor norm-template case unless broader normalization-template or launch-overhead work moves the family."""
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


N_BATCH = 128
CHANNELS = 384
HEIGHT = 14
WIDTH = 14
SPATIAL = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N_BATCH * SPATIAL
TOTAL_ELEMENTS = N_BATCH * CHANNELS * SPATIAL
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361

PARTIAL_BLOCK_M = 256
PARTIAL_BLOCK_C = 8
NUM_PARTIALS = (ELEMENTS_PER_CHANNEL + PARTIAL_BLOCK_M - 1) // PARTIAL_BLOCK_M
FINAL_BLOCK_C = 32
EPILOGUE_BLOCK = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _triple_partial_stats_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        num_partials: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        partial_id = tl.program_id(1)
        channel_offsets = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        m_offsets = partial_id * BLOCK_M + tl.arange(0, BLOCK_M)
        active = (m_offsets[:, None] < elements_per_channel) & (
            channel_offsets[None, :] < channels
        )

        n_idx = m_offsets // hw_size
        hw_idx = m_offsets - n_idx * hw_size
        flat = (
            n_idx[:, None] * channels * hw_size
            + channel_offsets[None, :] * hw_size
            + hw_idx[:, None]
        )

        x0 = tl.load(x0_ptr + flat, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(x0, axis=0)
        sumsq0 = tl.sum(x0 * x0, axis=0)
        x1 = tl.load(x1_ptr + flat, mask=active, other=0.0).to(tl.float32)
        sum1 = tl.sum(x1, axis=0)
        sumsq1 = tl.sum(x1 * x1, axis=0)
        x2 = tl.load(x2_ptr + flat, mask=active, other=0.0).to(tl.float32)
        sum2 = tl.sum(x2, axis=0)
        sumsq2 = tl.sum(x2 * x2, axis=0)

        partial_base = partial_id * channels + channel_offsets
        partial_stride = num_partials * channels
        store_mask = channel_offsets < channels
        tl.store(partial_ptr + partial_base, sum0, mask=store_mask)
        tl.store(partial_ptr + partial_stride + partial_base, sum1, mask=store_mask)
        tl.store(partial_ptr + 2 * partial_stride + partial_base, sum2, mask=store_mask)
        tl.store(partial_ptr + 3 * partial_stride + partial_base, sumsq0, mask=store_mask)
        tl.store(partial_ptr + 4 * partial_stride + partial_base, sumsq1, mask=store_mask)
        tl.store(partial_ptr + 5 * partial_stride + partial_base, sumsq2, mask=store_mask)

    @triton.jit
    def _triple_finalize_stats_kernel(
        partial_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        running_mean2_ptr,
        running_var2_ptr,
        stats_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        num_partials: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        channel_offsets = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = channel_offsets < channels
        partial_stride = num_partials * channels

        sum0 = tl.zeros((BLOCK_C,), tl.float32)
        sum1 = tl.zeros((BLOCK_C,), tl.float32)
        sum2 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq0 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq1 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq2 = tl.zeros((BLOCK_C,), tl.float32)

        for partial_id in tl.static_range(0, num_partials):
            base = partial_id * channels + channel_offsets
            sum0 += tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
            sum1 += tl.load(partial_ptr + partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sum2 += tl.load(partial_ptr + 2 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq0 += tl.load(partial_ptr + 3 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq1 += tl.load(partial_ptr + 4 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq2 += tl.load(partial_ptr + 5 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)

        scale = 1.0 / elements_per_channel
        mean0 = sum0 * scale
        mean1 = sum1 * scale
        mean2 = sum2 * scale
        var0 = tl.maximum(sumsq0 * scale - mean0 * mean0, 0.0)
        var1 = tl.maximum(sumsq1 * scale - mean1 * mean1, 0.0)
        var2 = tl.maximum(sumsq2 * scale - mean2 * mean2, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)
        invstd2 = tl.rsqrt(var2 + eps)

        old_mean0 = tl.load(running_mean0_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var0 = tl.load(running_var0_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean1 = tl.load(running_mean1_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var1 = tl.load(running_var1_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean2 = tl.load(running_mean2_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var2 = tl.load(running_var2_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)

        tl.store(
            running_mean0_ptr + channel_offsets,
            old_mean0 * (1.0 - momentum) + mean0 * momentum,
            mask=mask,
        )
        tl.store(
            running_var0_ptr + channel_offsets,
            old_var0 * (1.0 - momentum) + var0 * running_var_correction * momentum,
            mask=mask,
        )
        tl.store(
            running_mean1_ptr + channel_offsets,
            old_mean1 * (1.0 - momentum) + mean1 * momentum,
            mask=mask,
        )
        tl.store(
            running_var1_ptr + channel_offsets,
            old_var1 * (1.0 - momentum) + var1 * running_var_correction * momentum,
            mask=mask,
        )
        tl.store(
            running_mean2_ptr + channel_offsets,
            old_mean2 * (1.0 - momentum) + mean2 * momentum,
            mask=mask,
        )
        tl.store(
            running_var2_ptr + channel_offsets,
            old_var2 * (1.0 - momentum) + var2 * running_var_correction * momentum,
            mask=mask,
        )

        tl.store(stats_ptr + channel_offsets, mean0, mask=mask)
        tl.store(stats_ptr + channels + channel_offsets, mean1, mask=mask)
        tl.store(stats_ptr + 2 * channels + channel_offsets, mean2, mask=mask)
        tl.store(stats_ptr + 3 * channels + channel_offsets, invstd0, mask=mask)
        tl.store(stats_ptr + 4 * channels + channel_offsets, invstd1, mask=mask)
        tl.store(stats_ptr + 5 * channels + channel_offsets, invstd2, mask=mask)

    @triton.jit
    def _triple_bn_sum_relu_kernel(
        x0_ptr,
        weight0_ptr,
        bias0_ptr,
        x1_ptr,
        weight1_ptr,
        bias1_ptr,
        x2_ptr,
        weight2_ptr,
        bias2_ptr,
        stats_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total_elements
        channel = (offsets // hw_size) % channels

        x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        mean0 = tl.load(stats_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        mean1 = tl.load(stats_ptr + channels + channel, mask=mask, other=0.0).to(tl.float32)
        mean2 = tl.load(stats_ptr + 2 * channels + channel, mask=mask, other=0.0).to(tl.float32)
        invstd0 = tl.load(stats_ptr + 3 * channels + channel, mask=mask, other=0.0).to(tl.float32)
        invstd1 = tl.load(stats_ptr + 4 * channels + channel, mask=mask, other=0.0).to(tl.float32)
        invstd2 = tl.load(stats_ptr + 5 * channels + channel, mask=mask, other=0.0).to(tl.float32)

        weight0 = tl.load(weight0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y0 = (x0 - mean0) * invstd0 * weight0 + bias0
        y1 = (x1 - mean1) * invstd1 * weight1 + bias1
        y2 = (x2 - mean2) * invstd2 * weight2 + bias2
        summed = y0 + y1 + y2
        relu = tl.where((summed > 0.0) | (summed != summed), summed, 0.0)
        tl.store(out_ptr + offsets, relu, mask=mask)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 15:
        raise ValueError(f"{REPRO_ID} expects 15 inputs, got {len(inputs)}")

    nchw_shape = (N_BATCH, CHANNELS, HEIGHT, WIDTH)
    nchw_stride = (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1)
    vec_shape = (CHANNELS,)
    vec_stride = (1,)

    tensors = (
        _require_f32_tensor("relu_19", inputs[0], nchw_shape, nchw_stride),
        _require_f32_tensor("arg322_1", inputs[1], vec_shape, vec_stride),
        _require_f32_tensor("arg323_1", inputs[2], vec_shape, vec_stride),
        _require_f32_tensor("arg324_1", inputs[3], vec_shape, vec_stride),
        _require_f32_tensor("arg325_1", inputs[4], vec_shape, vec_stride),
        _require_f32_tensor("convolution_40", inputs[5], nchw_shape, nchw_stride),
        _require_f32_tensor("arg328_1", inputs[6], vec_shape, vec_stride),
        _require_f32_tensor("arg329_1", inputs[7], vec_shape, vec_stride),
        _require_f32_tensor("arg330_1", inputs[8], vec_shape, vec_stride),
        _require_f32_tensor("arg331_1", inputs[9], vec_shape, vec_stride),
        _require_f32_tensor("convolution_41", inputs[10], nchw_shape, nchw_stride),
        _require_f32_tensor("arg334_1", inputs[11], vec_shape, vec_stride),
        _require_f32_tensor("arg335_1", inputs[12], vec_shape, vec_stride),
        _require_f32_tensor("arg336_1", inputs[13], vec_shape, vec_stride),
        _require_f32_tensor("arg337_1", inputs[14], vec_shape, vec_stride),
    )

    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    var0, mean0 = torch.var_mean(x0, dim=(0, 2, 3), correction=0, keepdim=True)
    var1, mean1 = torch.var_mean(x1, dim=(0, 2, 3), correction=0, keepdim=True)
    var2, mean2 = torch.var_mean(x2, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd0 = torch.rsqrt(var0 + EPS)
    invstd1 = torch.rsqrt(var1 + EPS)
    invstd2 = torch.rsqrt(var2 + EPS)

    out = torch.relu(
        (x0 - mean0) * invstd0 * weight0.view(1, CHANNELS, 1, 1)
        + bias0.view(1, CHANNELS, 1, 1)
        + (x1 - mean1) * invstd1 * weight1.view(1, CHANNELS, 1, 1)
        + bias1.view(1, CHANNELS, 1, 1)
        + (x2 - mean2) * invstd2 * weight2.view(1, CHANNELS, 1, 1)
        + bias2.view(1, CHANNELS, 1, 1)
    )

    mean0_1d = mean0.squeeze((0, 2, 3))
    mean1_1d = mean1.squeeze((0, 2, 3))
    mean2_1d = mean2.squeeze((0, 2, 3))
    var0_1d = var0.squeeze((0, 2, 3))
    var1_1d = var1.squeeze((0, 2, 3))
    var2_1d = var2.squeeze((0, 2, 3))
    running_mean0.copy_(running_mean0 * (1.0 - MOMENTUM) + mean0_1d * MOMENTUM)
    running_var0.copy_(
        running_var0 * (1.0 - MOMENTUM)
        + var0_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    running_mean1.copy_(running_mean1 * (1.0 - MOMENTUM) + mean1_1d * MOMENTUM)
    running_var1.copy_(
        running_var1 * (1.0 - MOMENTUM)
        + var1_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    running_mean2.copy_(running_mean2 * (1.0 - MOMENTUM) + mean2_1d * MOMENTUM)
    running_var2.copy_(
        running_var2 * (1.0 - MOMENTUM)
        + var2_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )

    return (
        invstd0.squeeze((0, 2, 3)),
        invstd1.squeeze((0, 2, 3)),
        invstd2.squeeze((0, 2, 3)),
        out,
        mean2,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )


def _run_triton_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_triple_bn_sum_relu.py")

    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        x2,
        running_mean2,
        running_var2,
        weight2,
        bias2,
    ) = inputs

    stats = torch.empty_strided(
        (6, CHANNELS),
        (CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    partial = torch.empty_strided(
        (6, NUM_PARTIALS, CHANNELS),
        (NUM_PARTIALS * CHANNELS, CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (N_BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1),
        device=x0.device,
        dtype=torch.float32,
    )

    _triple_partial_stats_kernel[(
        triton.cdiv(CHANNELS, PARTIAL_BLOCK_C),
        NUM_PARTIALS,
    )](
        x0,
        x1,
        x2,
        partial,
        channels=CHANNELS,
        hw_size=SPATIAL,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        num_partials=NUM_PARTIALS,
        BLOCK_M=PARTIAL_BLOCK_M,
        BLOCK_C=PARTIAL_BLOCK_C,
        num_warps=8,
        num_stages=4,
    )
    _triple_finalize_stats_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        stats,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        num_partials=NUM_PARTIALS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=1,
        num_stages=4,
    )
    _triple_bn_sum_relu_kernel[(triton.cdiv(TOTAL_ELEMENTS, EPILOGUE_BLOCK),)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        x2,
        weight2,
        bias2,
        stats,
        out,
        total_elements=TOTAL_ELEMENTS,
        channels=CHANNELS,
        hw_size=SPATIAL,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    return (
        stats[3],
        stats[4],
        stats[5],
        out,
        stats[2].view(1, CHANNELS, 1, 1),
        stats[1].view(1, CHANNELS, 1, 1),
        stats[0].view(1, CHANNELS, 1, 1),
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
    )


def oracle_forward(inputs):
    """Run the full Repro.forward computation."""
    tensors = _validate_inputs(inputs)
    if not tensors[0].is_cuda:
        return _torch_oracle(tensors)
    return _run_triton_oracle(tensors)


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
