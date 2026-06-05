"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Inception training batchnorm affine ReLU branch fanout, in-place running-stat updates, channel concatenation semantics, and final spatial mean as direct branch-to-output reductions, whereas Inductor materializes normalized branch tensors and channel cats before reducing the 8x8 spatial tail; Inductor cannot do this today because scheduler fusion does not commute the spatial mean through channel cats and sink the affine ReLU producer into the consumer while preserving mutable BN running-stat outputs; the fix is SCHEDULER_FUSION: add a training-BN-affine-ReLU-to-spatial-mean fusion/template that emits final channel slices and explicit running-stat copy_ epilogues."""
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
HEIGHT = 8
WIDTH = 8
SPATIAL = HEIGHT * WIDTH
REDUCTION_SIZE = N_BATCH * SPATIAL
TOTAL_CHANNELS = 2048
EPS = 0.001
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804

STATS_BLOCK_M = 256
STATS_BLOCK_C = 16
NUM_PARTIALS = REDUCTION_SIZE // STATS_BLOCK_M
FINAL_BLOCK_C = 32
OUT_BLOCK_N = 4
OUT_BLOCK_C = 16

OUTPUT_SHAPE = (N_BATCH, TOTAL_CHANNELS)
OUTPUT_STRIDE = (TOTAL_CHANNELS, 1)

BRANCH_SPECS = (
    ("branch0", 0, 1, 2, 3, 4, 320, 0),
    ("branch1", 5, 6, 7, 8, 9, 384, 320),
    ("branch2", 10, 11, 12, 13, 14, 384, 704),
    ("branch3", 15, 16, 17, 18, 19, 384, 1088),
    ("branch4", 20, 21, 22, 23, 24, 384, 1472),
    ("branch5", 25, 26, 27, 28, 29, 192, 1856),
)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        out_offset,
        C: tl.constexpr,
        TOTAL_C: tl.constexpr,
        M: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        partial_id = tl.program_id(1)
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        offsets_m = partial_id * BLOCK_M + tl.arange(0, BLOCK_M)
        active = (offsets_m[:, None] < M) & (channels[None, :] < C)

        values = tl.load(
            x_ptr + offsets_m[:, None] * C + channels[None, :],
            mask=active,
            other=0.0,
        ).to(tl.float32)
        sum_x = tl.sum(values, axis=0)
        sum_x2 = tl.sum(values * values, axis=0)

        store_offsets = partial_id * TOTAL_C + out_offset + channels
        tl.store(partial_sum_ptr + store_offsets, sum_x, mask=channels < C)
        tl.store(partial_sumsq_ptr + store_offsets, sum_x2, mask=channels < C)


    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        inv_std_ptr,
        running_mean_ptr,
        running_var_ptr,
        out_offset,
        C: tl.constexpr,
        TOTAL_C: tl.constexpr,
        M: tl.constexpr,
        NUM_P: tl.constexpr,
        EPSILON: tl.constexpr,
        VAR_CORRECTION: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        channel_block = tl.program_id(0)
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = channels < C

        sum_x = tl.zeros((BLOCK_C,), tl.float32)
        sum_x2 = tl.zeros((BLOCK_C,), tl.float32)
        for partial_id in tl.static_range(0, NUM_P):
            load_offsets = partial_id * TOTAL_C + out_offset + channels
            sum_x += tl.load(partial_sum_ptr + load_offsets, mask=mask, other=0.0).to(tl.float32)
            sum_x2 += tl.load(partial_sumsq_ptr + load_offsets, mask=mask, other=0.0).to(tl.float32)

        mean = sum_x / M
        var = sum_x2 / M - mean * mean
        var = tl.maximum(var, 0.0)
        inv_std = tl.rsqrt(var + EPSILON)

        old_mean = tl.load(running_mean_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + channels, mask=mask, other=0.0).to(tl.float32)
        tl.store(mean_ptr + out_offset + channels, mean, mask=mask)
        tl.store(inv_std_ptr + out_offset + channels, inv_std, mask=mask)
        tl.store(running_mean_ptr + channels, old_mean * 0.9 + mean * 0.1, mask=mask)
        tl.store(
            running_var_ptr + channels,
            old_var * 0.9 + var * VAR_CORRECTION * 0.1,
            mask=mask,
        )


    @triton.jit
    def _bn_relu_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        inv_std_ptr,
        out_ptr,
        out_offset,
        C: tl.constexpr,
        N: tl.constexpr,
        TOTAL_C: tl.constexpr,
        SPATIAL_SIZE: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch_block = tl.program_id(0)
        channel_block = tl.program_id(1)
        batches = batch_block * BLOCK_N + tl.arange(0, BLOCK_N)
        channels = channel_block * BLOCK_C + tl.arange(0, BLOCK_C)
        batch_mask = batches < N
        channel_mask = channels < C
        mask = batch_mask[:, None] & channel_mask[None, :]

        mean = tl.load(mean_ptr + out_offset + channels, mask=channel_mask, other=0.0).to(tl.float32)
        inv_std = tl.load(inv_std_ptr + out_offset + channels, mask=channel_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
        for spatial in tl.static_range(0, SPATIAL_SIZE):
            x = tl.load(
                x_ptr + (batches[:, None] * SPATIAL_SIZE + spatial) * C + channels[None, :],
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            affine = (x - mean[None, :]) * inv_std[None, :] * weight[None, :] + bias[None, :]
            relu = tl.where((affine > 0.0) | (affine != affine), affine, 0.0)
            acc += relu

        store_offsets = batches[:, None] * TOTAL_C + out_offset + channels[None, :]
        tl.store(out_ptr + store_offsets, acc * (1.0 / SPATIAL_SIZE), mask=mask)


def _expected_input_stride(channels: int) -> tuple[int, int, int, int]:
    return (channels * SPATIAL, 1, WIDTH * channels, channels)


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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]]:
    if len(inputs) != 31:
        raise ValueError(f"{REPRO_ID} expects 31 inputs, got {len(inputs)}")

    shape_param = inputs[30]
    if tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"_shape_param_0 is {shape_param!r}, expected {list(OUTPUT_SHAPE)}")

    branches = []
    for name, x_i, running_mean_i, running_var_i, weight_i, bias_i, channels, out_offset in BRANCH_SPECS:
        x = _require_f32_tensor(
            f"{name}.activation",
            inputs[x_i],
            (N_BATCH, channels, HEIGHT, WIDTH),
            _expected_input_stride(channels),
        )
        running_mean = _require_f32_tensor(
            f"{name}.running_mean",
            inputs[running_mean_i],
            (channels,),
            (1,),
        )
        running_var = _require_f32_tensor(
            f"{name}.running_var",
            inputs[running_var_i],
            (channels,),
            (1,),
        )
        weight = _require_f32_tensor(f"{name}.weight", inputs[weight_i], (channels,), (1,))
        bias = _require_f32_tensor(f"{name}.bias", inputs[bias_i], (channels,), (1,))
        device = x.device
        if any(t.device != device for t in (running_mean, running_var, weight, bias)):
            raise ValueError(f"{name} tensors are not all on {device}")
        branches.append((x, running_mean, running_var, weight, bias, channels, out_offset))

    first_device = branches[0][0].device
    if any(branch[0].device != first_device for branch in branches):
        raise ValueError("all activation inputs must be on the same device")
    return branches


def _torch_oracle(
    branches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, ...]:
    pooled = []
    for x, running_mean, running_var, weight, bias, channels, _out_offset in branches:
        var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
        inv_std = torch.rsqrt(var + EPS)
        y = torch.relu((x - mean) * inv_std * weight.view(1, channels, 1, 1) + bias.view(1, channels, 1, 1))
        pooled.append(torch.mean(y, dim=(-1, -2)))

        mean_1d = mean.squeeze((0, 2, 3))
        var_1d = var.squeeze((0, 2, 3))
        running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
        running_var.copy_(
            running_var * (1.0 - MOMENTUM)
            + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
        )

    out = torch.cat(pooled, dim=1).reshape(OUTPUT_SHAPE)
    return (out,) + tuple(branch[i] for branch in branches for i in (1, 2))


def _run_triton_oracle(
    branches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]],
) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    device = branches[0][0].device
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=device, dtype=torch.float32)
    partial_sum = torch.empty((NUM_PARTIALS, TOTAL_CHANNELS), device=device, dtype=torch.float32)
    partial_sumsq = torch.empty((NUM_PARTIALS, TOTAL_CHANNELS), device=device, dtype=torch.float32)
    mean = torch.empty((TOTAL_CHANNELS,), device=device, dtype=torch.float32)
    inv_std = torch.empty((TOTAL_CHANNELS,), device=device, dtype=torch.float32)

    for x, _running_mean, _running_var, _weight, _bias, channels, out_offset in branches:
        _partial_stats_kernel[(triton.cdiv(channels, STATS_BLOCK_C), NUM_PARTIALS)](
            x,
            partial_sum,
            partial_sumsq,
            out_offset,
            C=channels,
            TOTAL_C=TOTAL_CHANNELS,
            M=REDUCTION_SIZE,
            BLOCK_M=STATS_BLOCK_M,
            BLOCK_C=STATS_BLOCK_C,
            num_warps=8,
            num_stages=4,
        )

    for _x, running_mean, running_var, _weight, _bias, channels, out_offset in branches:
        _finalize_stats_kernel[(triton.cdiv(channels, FINAL_BLOCK_C),)](
            partial_sum,
            partial_sumsq,
            mean,
            inv_std,
            running_mean,
            running_var,
            out_offset,
            C=channels,
            TOTAL_C=TOTAL_CHANNELS,
            M=REDUCTION_SIZE,
            NUM_P=NUM_PARTIALS,
            EPSILON=EPS,
            VAR_CORRECTION=RUNNING_VAR_CORRECTION,
            BLOCK_C=FINAL_BLOCK_C,
            num_warps=1,
            num_stages=4,
        )

    for x, _running_mean, _running_var, weight, bias, channels, out_offset in branches:
        _bn_relu_spatial_mean_kernel[
            (triton.cdiv(N_BATCH, OUT_BLOCK_N), triton.cdiv(channels, OUT_BLOCK_C))
        ](
            x,
            weight,
            bias,
            mean,
            inv_std,
            out,
            out_offset,
            C=channels,
            N=N_BATCH,
            TOTAL_C=TOTAL_CHANNELS,
            SPATIAL_SIZE=SPATIAL,
            BLOCK_N=OUT_BLOCK_N,
            BLOCK_C=OUT_BLOCK_C,
            num_warps=4,
            num_stages=4,
        )

    return (out,) + tuple(branch[i] for branch in branches for i in (1, 2))


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
    branches = _validate_inputs(inputs)
    if not branches[0][0].is_cuda:
        return _torch_oracle(branches)
    return _run_triton_oracle(branches)


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
