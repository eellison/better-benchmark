"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNet training batchnorm var_mean, running-stat copy_ updates, affine ReLU, and low-memory 3x3 stride-2 maxpool-with-offsets scope with tiled channel reductions and a fused normalization-to-pool stencil that writes only final pooled values and int8 offsets, whereas Inductor materializes the normalized/ReLU activation before the pooling stencil and handles the mutable running-stat outputs as separate epilogue work; Inductor cannot do this today because scheduler fusion does not inline a reduction producer with mutation side outputs into a multi-output low-memory maxpool consumer while preserving offset semantics; the fix is SCHEDULER_FUSION: teach the scheduler a training-BN-affine-ReLU-to-low-memory-maxpool template with explicit running-stat copy_ epilogues."""
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
N_BATCH = 32
CHANNELS = 64
HEIGHT = 112
WIDTH = 112
OUT_HEIGHT = 56
OUT_WIDTH = 56
HW = HEIGHT * WIDTH
ELEMS_PER_CHANNEL = N_BATCH * HW
EPS = 1.0e-5
RUNNING_VAR_CORRECTION = 1.0000024912370735
PARTIAL_BLOCK = 2048
NUM_PARTIALS = (ELEMS_PER_CHANNEL + PARTIAL_BLOCK - 1) // PARTIAL_BLOCK
FINAL_BLOCK = 256
POOL_BLOCK_C = 2
POOL_BLOCK_OUT = 128
INPUT_SHAPE = (N_BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (N_BATCH, CHANNELS, OUT_HEIGHT, OUT_WIDTH)
OUTPUT_STRIDE = (CHANNELS * OUT_HEIGHT * OUT_WIDTH, OUT_HEIGHT * OUT_WIDTH, OUT_WIDTH, 1)


if triton is not None:

    @triton.jit
    def _bn_partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        elems_per_channel: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        num_partials: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        partial_id = tl.program_id(1)
        offsets = partial_id * BLOCK_M + tl.arange(0, BLOCK_M)
        active = offsets < elems_per_channel

        batch = offsets // hw
        spatial = offsets - batch * hw
        flat = batch * (channels * hw) + channel * hw + spatial
        x = tl.load(x_ptr + flat, mask=active, other=0.0).to(tl.float32)

        partial_offset = channel * num_partials + partial_id
        tl.store(partial_sum_ptr + partial_offset, tl.sum(x, axis=0))
        tl.store(partial_sumsq_ptr + partial_offset, tl.sum(x * x, axis=0))


    @triton.jit
    def _bn_finalize_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        inv_std_ptr,
        running_mean_ptr,
        running_var_ptr,
        elems_per_channel: tl.constexpr,
        num_partials: tl.constexpr,
        eps: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        active = offsets < num_partials
        partial_base = channel * num_partials + offsets

        sum_x = tl.sum(tl.load(partial_sum_ptr + partial_base, mask=active, other=0.0), axis=0)
        sum_x2 = tl.sum(tl.load(partial_sumsq_ptr + partial_base, mask=active, other=0.0), axis=0)
        mean = sum_x / elems_per_channel
        var = sum_x2 / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(mean_ptr + channel, mean)
        tl.store(inv_std_ptr + channel, inv_std)
        tl.store(running_mean_ptr + channel, old_mean * 0.9 + mean * 0.1)
        tl.store(running_var_ptr + channel, old_var * 0.9 + var * running_var_correction * 0.1)


    @triton.jit
    def _bn_relu_maxpool_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        inv_std_ptr,
        values_ptr,
        offsets_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        batch = tl.program_id(0)
        channel_block = tl.program_id(1)
        out_block = tl.program_id(2)

        channel_offsets = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        out_offsets = out_block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        out_active = out_offsets < (out_height * out_width)
        channel_active = channel_offsets < channels
        out_h = out_offsets // out_width
        out_w = out_offsets - out_h * out_width

        input_base = batch * (channels * hw) + channel_offsets[:, None] * hw

        mean = tl.load(mean_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        inv_std = tl.load(inv_std_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=channel_active, other=0.0).to(tl.float32)

        best = tl.full((BLOCK_C, BLOCK_OUT), -float("inf"), tl.float32)
        best_offset = tl.zeros((BLOCK_C, BLOCK_OUT), tl.int32)

        for kh in tl.static_range(0, 3):
            in_h = out_h * 2 + kh - 1
            valid_h = (in_h >= 0) & (in_h < height)
            load_h = tl.minimum(tl.maximum(in_h, 0), height - 1)
            for kw in tl.static_range(0, 3):
                in_w = out_w * 2 + kw - 1
                valid_out = out_active & valid_h & (in_w >= 0) & (in_w < width)
                load_w = tl.minimum(tl.maximum(in_w, 0), width - 1)
                valid = channel_active[:, None] & valid_out[None, :]
                x = tl.load(
                    x_ptr + input_base + load_h[None, :] * width + load_w[None, :],
                    mask=valid,
                    other=0.0,
                ).to(tl.float32)

                affine = (x - mean[:, None]) * inv_std[:, None] * weight[:, None] + bias[:, None]
                relu = tl.where((affine > 0.0) | (affine != affine), affine, 0.0)
                take = valid & ((relu > best) | (relu != relu))
                best = tl.where(take, relu, best)
                best_offset = tl.where(take, kh * 3 + kw, best_offset)

        out_hw = out_height * out_width
        out_base = batch * (channels * out_hw) + channel_offsets[:, None] * out_hw + out_offsets[None, :]
        store_mask = channel_active[:, None] & out_active[None, :]
        tl.store(values_ptr + out_base, best, mask=store_mask)
        tl.store(offsets_ptr + out_base, best_offset.to(tl.int8), mask=store_mask)


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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x = _require_f32_tensor("convolution", inputs[0], INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _require_f32_tensor("arg3_1", inputs[1], STAT_SHAPE, STAT_STRIDE)
    running_var = _require_f32_tensor("arg4_1", inputs[2], STAT_SHAPE, STAT_STRIDE)
    weight = _require_f32_tensor("arg5_1", inputs[3], STAT_SHAPE, STAT_STRIDE)
    bias = _require_f32_tensor("arg6_1", inputs[4], STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_oracle(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    normalized = (x - mean) * torch.rsqrt(var + EPS)
    relu = torch.relu(normalized * weight.view(1, CHANNELS, 1, 1) + bias.view(1, CHANNELS, 1, 1))
    values, offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        relu,
        [3, 3],
        [2, 2],
        [1, 1],
        [1, 1],
        False,
    )
    running_mean.copy_(running_mean * 0.9 + mean.squeeze((0, 2, 3)) * 0.1)
    running_var.copy_(running_var * 0.9 + var.squeeze((0, 2, 3)) * RUNNING_VAR_CORRECTION * 0.1)
    return values, offsets, running_mean, running_var


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
    if not x.is_cuda:
        return _torch_oracle(x, running_mean, running_var, weight, bias)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    partial_sum = torch.empty((CHANNELS, NUM_PARTIALS), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((CHANNELS, NUM_PARTIALS), device=x.device, dtype=torch.float32)
    mean = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    inv_std = torch.empty((CHANNELS,), device=x.device, dtype=torch.float32)
    values = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    offsets = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.int8,
    )

    _bn_partial_stats_kernel[(CHANNELS, NUM_PARTIALS)](
        x,
        partial_sum,
        partial_sumsq,
        elems_per_channel=ELEMS_PER_CHANNEL,
        channels=CHANNELS,
        hw=HW,
        num_partials=NUM_PARTIALS,
        BLOCK_M=PARTIAL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_finalize_stats_kernel[(CHANNELS,)](
        partial_sum,
        partial_sumsq,
        mean,
        inv_std,
        running_mean,
        running_var,
        elems_per_channel=ELEMS_PER_CHANNEL,
        num_partials=NUM_PARTIALS,
        eps=EPS,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=FINAL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _bn_relu_maxpool_kernel[(
        N_BATCH,
        triton.cdiv(CHANNELS, POOL_BLOCK_C),
        triton.cdiv(OUT_HEIGHT * OUT_WIDTH, POOL_BLOCK_OUT),
    )](
        x,
        weight,
        bias,
        mean,
        inv_std,
        values,
        offsets,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        hw=HW,
        BLOCK_C=POOL_BLOCK_C,
        BLOCK_OUT=POOL_BLOCK_OUT,
        num_warps=8,
        num_stages=3,
    )
    return values, offsets, running_mean, running_var


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
