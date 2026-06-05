"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete four-branch Inception training-BatchNorm scope by grouping the sibling per-channel population var_mean reductions, eps=1e-3 affine ReLU epilogues, eight in-place running-stat copy_ outputs, and the final `[128,768,17,17]` channel cat into shared shape-specialized Triton stats/finalize/epilogue launches that write the cat output directly, whereas Inductor still treats the independent norm-template branches and their mutable copy_ side effects as separate scheduler units around the static cat consumer; Inductor cannot do this today because the scheduler does not co-schedule sibling training-BN templates that feed a fixed channel concatenation while preserving all running-stat aliases; the fix is SCHEDULER_FUSION: teach the norm-template scheduler to group static sibling BN branches, emit their copy_ epilogues, and sink the channel-cat layout into one shared full-scope output plan."""
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
N_BATCH = 128
BRANCHES = 4
CHANNELS = 192
OUT_CHANNELS = BRANCHES * CHANNELS
HEIGHT = 17
WIDTH = 17
SPATIAL = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N_BATCH * SPATIAL
TOTAL_OUTPUT_ELEMENTS = N_BATCH * OUT_CHANNELS * SPATIAL
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000270336027683

NCHW_SHAPE = (N_BATCH, CHANNELS, HEIGHT, WIDTH)
NCHW_STRIDE = (CHANNELS * SPATIAL, SPATIAL, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (N_BATCH, OUT_CHANNELS, HEIGHT, WIDTH)
OUTPUT_STRIDE = (OUT_CHANNELS * SPATIAL, SPATIAL, WIDTH, 1)

PARTIAL_BLOCK_M = 256
PARTIAL_BLOCK_C = 8
NUM_PARTIALS = (ELEMENTS_PER_CHANNEL + PARTIAL_BLOCK_M - 1) // PARTIAL_BLOCK_M
FINAL_BLOCK_C = 16
EPILOGUE_BLOCK_M = 1024

if triton is not None:

    @triton.jit
    def _quad_partial_stats_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
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
        mask = (m_offsets[:, None] < elements_per_channel) & (
            channel_offsets[None, :] < channels
        )

        n_idx = m_offsets // hw_size
        hw_idx = m_offsets - n_idx * hw_size
        flat_offsets = (
            n_idx[:, None] * channels * hw_size
            + channel_offsets[None, :] * hw_size
            + hw_idx[:, None]
        )

        x0 = tl.load(x0_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(x0, axis=0)
        sumsq0 = tl.sum(x0 * x0, axis=0)
        x1 = tl.load(x1_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum1 = tl.sum(x1, axis=0)
        sumsq1 = tl.sum(x1 * x1, axis=0)
        x2 = tl.load(x2_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum2 = tl.sum(x2, axis=0)
        sumsq2 = tl.sum(x2 * x2, axis=0)
        x3 = tl.load(x3_ptr + flat_offsets, mask=mask, other=0.0).to(tl.float32)
        sum3 = tl.sum(x3, axis=0)
        sumsq3 = tl.sum(x3 * x3, axis=0)

        base = partial_id * channels + channel_offsets
        stride = num_partials * channels
        store_mask = channel_offsets < channels
        tl.store(partial_ptr + base, sum0, mask=store_mask)
        tl.store(partial_ptr + stride + base, sum1, mask=store_mask)
        tl.store(partial_ptr + 2 * stride + base, sum2, mask=store_mask)
        tl.store(partial_ptr + 3 * stride + base, sum3, mask=store_mask)
        tl.store(partial_ptr + 4 * stride + base, sumsq0, mask=store_mask)
        tl.store(partial_ptr + 5 * stride + base, sumsq1, mask=store_mask)
        tl.store(partial_ptr + 6 * stride + base, sumsq2, mask=store_mask)
        tl.store(partial_ptr + 7 * stride + base, sumsq3, mask=store_mask)

    @triton.jit
    def _quad_finalize_stats_kernel(
        partial_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        running_mean2_ptr,
        running_var2_ptr,
        running_mean3_ptr,
        running_var3_ptr,
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
        sum3 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq0 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq1 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq2 = tl.zeros((BLOCK_C,), tl.float32)
        sumsq3 = tl.zeros((BLOCK_C,), tl.float32)

        for partial_id in tl.static_range(0, num_partials):
            base = partial_id * channels + channel_offsets
            sum0 += tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
            sum1 += tl.load(partial_ptr + partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sum2 += tl.load(partial_ptr + 2 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sum3 += tl.load(partial_ptr + 3 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq0 += tl.load(partial_ptr + 4 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq1 += tl.load(partial_ptr + 5 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq2 += tl.load(partial_ptr + 6 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)
            sumsq3 += tl.load(partial_ptr + 7 * partial_stride + base, mask=mask, other=0.0).to(tl.float32)

        inv_count = 1.0 / elements_per_channel
        mean0 = sum0 * inv_count
        mean1 = sum1 * inv_count
        mean2 = sum2 * inv_count
        mean3 = sum3 * inv_count
        var0 = tl.maximum(sumsq0 * inv_count - mean0 * mean0, 0.0)
        var1 = tl.maximum(sumsq1 * inv_count - mean1 * mean1, 0.0)
        var2 = tl.maximum(sumsq2 * inv_count - mean2 * mean2, 0.0)
        var3 = tl.maximum(sumsq3 * inv_count - mean3 * mean3, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)
        invstd2 = tl.rsqrt(var2 + eps)
        invstd3 = tl.rsqrt(var3 + eps)

        old_mean0 = tl.load(running_mean0_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var0 = tl.load(running_var0_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean1 = tl.load(running_mean1_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var1 = tl.load(running_var1_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean2 = tl.load(running_mean2_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var2 = tl.load(running_var2_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_mean3 = tl.load(running_mean3_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
        old_var3 = tl.load(running_var3_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)

        one_minus_momentum = 1.0 - momentum
        tl.store(
            running_mean0_ptr + channel_offsets,
            old_mean0 * one_minus_momentum + mean0 * momentum,
            mask=mask,
        )
        tl.store(
            running_var0_ptr + channel_offsets,
            old_var0 * one_minus_momentum + var0 * running_var_correction * momentum,
            mask=mask,
        )
        tl.store(
            running_mean1_ptr + channel_offsets,
            old_mean1 * one_minus_momentum + mean1 * momentum,
            mask=mask,
        )
        tl.store(
            running_var1_ptr + channel_offsets,
            old_var1 * one_minus_momentum + var1 * running_var_correction * momentum,
            mask=mask,
        )
        tl.store(
            running_mean2_ptr + channel_offsets,
            old_mean2 * one_minus_momentum + mean2 * momentum,
            mask=mask,
        )
        tl.store(
            running_var2_ptr + channel_offsets,
            old_var2 * one_minus_momentum + var2 * running_var_correction * momentum,
            mask=mask,
        )
        tl.store(
            running_mean3_ptr + channel_offsets,
            old_mean3 * one_minus_momentum + mean3 * momentum,
            mask=mask,
        )
        tl.store(
            running_var3_ptr + channel_offsets,
            old_var3 * one_minus_momentum + var3 * running_var_correction * momentum,
            mask=mask,
        )

        tl.store(stats_ptr + channel_offsets, mean0, mask=mask)
        tl.store(stats_ptr + channels + channel_offsets, mean1, mask=mask)
        tl.store(stats_ptr + 2 * channels + channel_offsets, mean2, mask=mask)
        tl.store(stats_ptr + 3 * channels + channel_offsets, mean3, mask=mask)
        tl.store(stats_ptr + 4 * channels + channel_offsets, invstd0, mask=mask)
        tl.store(stats_ptr + 5 * channels + channel_offsets, invstd1, mask=mask)
        tl.store(stats_ptr + 6 * channels + channel_offsets, invstd2, mask=mask)
        tl.store(stats_ptr + 7 * channels + channel_offsets, invstd3, mask=mask)

    @triton.jit
    def _quad_bn_relu_cat_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        weight0_ptr,
        bias0_ptr,
        weight1_ptr,
        bias1_ptr,
        weight2_ptr,
        bias2_ptr,
        weight3_ptr,
        bias3_ptr,
        stats_ptr,
        out_ptr,
        total_channels: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        out_channel = tl.program_id(0)
        block_m = tl.program_id(1)
        m_offsets = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = m_offsets < elements_per_channel
        branch = out_channel // channels
        local_channel = out_channel - branch * channels

        n_idx = m_offsets // hw_size
        hw_idx = m_offsets - n_idx * hw_size
        input_offsets = n_idx * channels * hw_size + local_channel * hw_size + hw_idx
        output_offsets = n_idx * total_channels * hw_size + out_channel * hw_size + hw_idx

        is0 = branch == 0
        is1 = branch == 1
        is2 = branch == 2
        is3 = branch == 3

        x = tl.load(x0_ptr + input_offsets, mask=mask & is0, other=0.0).to(tl.float32)
        x += tl.load(x1_ptr + input_offsets, mask=mask & is1, other=0.0).to(tl.float32)
        x += tl.load(x2_ptr + input_offsets, mask=mask & is2, other=0.0).to(tl.float32)
        x += tl.load(x3_ptr + input_offsets, mask=mask & is3, other=0.0).to(tl.float32)

        stats_offset = branch * channels + local_channel
        mean = tl.load(stats_ptr + stats_offset).to(tl.float32)
        invstd = tl.load(stats_ptr + 4 * channels + stats_offset).to(tl.float32)

        weight = tl.load(weight0_ptr + local_channel, mask=is0, other=0.0).to(tl.float32)
        weight += tl.load(weight1_ptr + local_channel, mask=is1, other=0.0).to(tl.float32)
        weight += tl.load(weight2_ptr + local_channel, mask=is2, other=0.0).to(tl.float32)
        weight += tl.load(weight3_ptr + local_channel, mask=is3, other=0.0).to(tl.float32)
        bias = tl.load(bias0_ptr + local_channel, mask=is0, other=0.0).to(tl.float32)
        bias += tl.load(bias1_ptr + local_channel, mask=is1, other=0.0).to(tl.float32)
        bias += tl.load(bias2_ptr + local_channel, mask=is2, other=0.0).to(tl.float32)
        bias += tl.load(bias3_ptr + local_channel, mask=is3, other=0.0).to(tl.float32)

        y = (x - mean) * invstd * weight + bias
        relu = tl.where((y > 0.0) | (y != y), y, 0.0)
        tl.store(out_ptr + output_offsets, relu, mask=mask)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 20:
        raise ValueError(f"{REPRO_ID} expects 20 inputs, got {len(inputs)}")

    names = (
        "convolution_60",
        "arg363_1",
        "arg364_1",
        "arg365_1",
        "arg366_1",
        "convolution_63",
        "arg381_1",
        "arg382_1",
        "arg383_1",
        "arg384_1",
        "convolution_68",
        "arg411_1",
        "arg412_1",
        "arg413_1",
        "arg414_1",
        "convolution_69",
        "arg417_1",
        "arg418_1",
        "arg419_1",
        "arg420_1",
    )
    tensors = tuple(
        _expect_f32_tensor(
            name,
            value,
            NCHW_SHAPE if index % 5 == 0 else VECTOR_SHAPE,
            NCHW_STRIDE if index % 5 == 0 else VECTOR_STRIDE,
        )
        for index, (name, value) in enumerate(zip(names, inputs))
    )

    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    outputs = []
    copy_outputs = []
    for branch in range(BRANCHES):
        base = branch * 5
        x = inputs[base]
        running_mean = inputs[base + 1]
        running_var = inputs[base + 2]
        weight = inputs[base + 3]
        bias = inputs[base + 4]

        var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
        invstd = torch.rsqrt(var + EPS)
        y = torch.relu((x - mean) * invstd * weight.view(1, CHANNELS, 1, 1) + bias.view(1, CHANNELS, 1, 1))
        outputs.append(y)

        mean_1d = mean.squeeze((0, 2, 3))
        var_1d = var.squeeze((0, 2, 3))
        running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
        running_var.copy_(
            running_var * (1.0 - MOMENTUM)
            + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
        )
        copy_outputs.extend((running_mean, running_var))

    return (torch.cat(outputs, dim=1), *copy_outputs)


def _run_triton_oracle(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

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
        x3,
        running_mean3,
        running_var3,
        weight3,
        bias3,
    ) = inputs

    partial = torch.empty_strided(
        (2 * BRANCHES, NUM_PARTIALS, CHANNELS),
        (NUM_PARTIALS * CHANNELS, CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(
        (2 * BRANCHES, CHANNELS),
        (CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )

    _quad_partial_stats_kernel[(
        triton.cdiv(CHANNELS, PARTIAL_BLOCK_C),
        NUM_PARTIALS,
    )](
        x0,
        x1,
        x2,
        x3,
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
    _quad_finalize_stats_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        running_mean3,
        running_var3,
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
    _quad_bn_relu_cat_kernel[(
        OUT_CHANNELS,
        triton.cdiv(ELEMENTS_PER_CHANNEL, EPILOGUE_BLOCK_M),
    )](
        x0,
        x1,
        x2,
        x3,
        weight0,
        bias0,
        weight1,
        bias1,
        weight2,
        bias2,
        weight3,
        bias3,
        stats,
        output,
        total_channels=OUT_CHANNELS,
        channels=CHANNELS,
        hw_size=SPATIAL,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        BLOCK_M=EPILOGUE_BLOCK_M,
        num_warps=4,
        num_stages=4,
    )
    return (
        output,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        running_mean2,
        running_var2,
        running_mean3,
        running_var3,
    )


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
