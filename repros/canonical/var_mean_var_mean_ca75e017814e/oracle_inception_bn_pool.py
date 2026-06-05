"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Inception training-BatchNorm block, including both per-channel var/mean reductions, mutable running-stat copy_ outputs, the low-memory max-pool offsets branch, channel concatenation, and the padded avg-pool tail by writing the final pooled concat layout directly, whereas Inductor already measures within the same envelope for this full-scope shape; Inductor cannot materially improve it today because the remaining time is dominated by required BN reductions, max-pool/avg-pool memory traffic, final output stores, and launch overhead rather than an actionable local norm-template scheduling gap; the fix is BANDWIDTH_BOUND: record this as an at-floor full-scope oracle and revisit only if broader norm, pooling, or launch-overhead work moves the baseline."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch._inductor.inductor_prims  # noqa: F401

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
C1 = 320
C2 = 192
C3 = 768
TOTAL_C = 1280
HEIGHT = 8
WIDTH = 8
POOL_IN_HEIGHT = 17
POOL_IN_WIDTH = 17
SPATIAL = HEIGHT * WIDTH
POOL_IN_SPATIAL = POOL_IN_HEIGHT * POOL_IN_WIDTH
REDUCTION_SIZE = N_BATCH * SPATIAL
EPS = 0.001
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804

STAT_BLOCK = 8192
OUTPUT_BLOCK = 256

OUTPUT_SHAPE = (N_BATCH, TOTAL_C, HEIGHT, WIDTH)
OUTPUT_STRIDE = (TOTAL_C * SPATIAL, SPATIAL, WIDTH, 1)
POOL_SHAPE = (N_BATCH, C3, HEIGHT, WIDTH)
POOL_STRIDE = (C3 * SPATIAL, SPATIAL, WIDTH, 1)


if triton is not None:

    @triton.jit
    def _dual_bn_stats_update_kernel(
        x1_ptr,
        x2_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        running_mean2_ptr,
        running_var2_ptr,
        mean1_ptr,
        invstd1_ptr,
        mean2_ptr,
        invstd2_ptr,
        c1: tl.constexpr,
        c2: tl.constexpr,
        hw_size: tl.constexpr,
        elems_per_channel: tl.constexpr,
        correction: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        block_size: tl.constexpr,
    ):
        stat_id = tl.program_id(0)
        first = stat_id < c1
        channel1 = tl.where(first, stat_id, 0)
        channel2 = tl.where(first, 0, stat_id - c1)

        offsets = tl.arange(0, block_size)
        mask = offsets < elems_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        x1_offsets = n_idx * c1 * hw_size + channel1 * hw_size + hw_idx
        x2_offsets = n_idx * c2 * hw_size + channel2 * hw_size + hw_idx
        values1 = tl.load(x1_ptr + x1_offsets, mask=mask & first, other=0.0).to(tl.float32)
        values2 = tl.load(x2_ptr + x2_offsets, mask=mask & ~first, other=0.0).to(tl.float32)
        values = values1 + values2

        total_sum = tl.sum(values, axis=0)
        total_sum_sq = tl.sum(values * values, axis=0)
        mean = total_sum / elems_per_channel
        var = total_sum_sq / elems_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean1 = tl.load(running_mean1_ptr + channel1, mask=first, other=0.0)
        old_var1 = tl.load(running_var1_ptr + channel1, mask=first, other=0.0)
        old_mean2 = tl.load(running_mean2_ptr + channel2, mask=~first, other=0.0)
        old_var2 = tl.load(running_var2_ptr + channel2, mask=~first, other=0.0)
        new_mean1 = old_mean1 * (1.0 - momentum) + mean * momentum
        new_var1 = old_var1 * (1.0 - momentum) + var * correction * momentum
        new_mean2 = old_mean2 * (1.0 - momentum) + mean * momentum
        new_var2 = old_var2 * (1.0 - momentum) + var * correction * momentum

        tl.store(running_mean1_ptr + channel1, new_mean1, mask=first)
        tl.store(running_var1_ptr + channel1, new_var1, mask=first)
        tl.store(mean1_ptr + channel1, mean, mask=first)
        tl.store(invstd1_ptr + channel1, invstd, mask=first)
        tl.store(running_mean2_ptr + channel2, new_mean2, mask=~first)
        tl.store(running_var2_ptr + channel2, new_var2, mask=~first)
        tl.store(mean2_ptr + channel2, mean, mask=~first)
        tl.store(invstd2_ptr + channel2, invstd, mask=~first)

    @triton.jit
    def _bn_relu_avgpool3x3_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        out_offset: tl.constexpr,
        c: tl.constexpr,
        total_c: tl.constexpr,
        h: tl.constexpr,
        w: tl.constexpr,
        hw_size: tl.constexpr,
        total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        ow = offsets % w
        tmp = offsets // w
        oh = tmp % h
        tmp = tmp // h
        channel = tmp % c
        batch = tmp // c

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        base = batch * c * hw_size + channel * hw_size

        acc = tl.zeros((block_size,), tl.float32)
        for kh in tl.static_range(0, 3):
            ih = oh + kh - 1
            valid_h = (ih >= 0) & (ih < h)
            for kw in tl.static_range(0, 3):
                iw = ow + kw - 1
                valid = mask & valid_h & (iw >= 0) & (iw < w)
                x = tl.load(x_ptr + base + ih * w + iw, mask=valid, other=0.0).to(tl.float32)
                y = (x - mean) * invstd * weight + bias
                y = tl.where((y > 0.0) | (y != y), y, 0.0)
                acc += tl.where(valid, y, 0.0)

        store_offsets = batch * total_c * hw_size + (out_offset + channel) * hw_size + oh * w + ow
        tl.store(out_ptr + store_offsets, acc * (1.0 / 9.0), mask=mask)

    @triton.jit
    def _maxpool2d_kernel(
        x_ptr,
        values_ptr,
        indices_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        in_h: tl.constexpr,
        in_w: tl.constexpr,
        out_h: tl.constexpr,
        out_w: tl.constexpr,
        in_hw: tl.constexpr,
        out_hw: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        ow = offsets % out_w
        tmp = offsets // out_w
        oh = tmp % out_h
        tmp = tmp // out_h
        channel = tmp % channels
        batch = tmp // channels
        base = batch * channels * in_hw + channel * in_hw

        max_val = tl.full((block_size,), -float("inf"), tl.float32)
        best_idx = tl.full((block_size,), 0, tl.int32)
        for kh in tl.static_range(0, 3):
            ih = oh * 2 + kh
            for kw in tl.static_range(0, 3):
                iw = ow * 2 + kw
                x = tl.load(x_ptr + base + ih * in_w + iw, mask=mask, other=-float("inf")).to(tl.float32)
                better = mask & ((x > max_val) | (x != x))
                max_val = tl.where(better, x, max_val)
                best_idx = tl.where(better, kh * 3 + kw, best_idx)

        tl.store(values_ptr + offsets, max_val, mask=mask)
        tl.store(indices_ptr + offsets, best_idx.to(tl.int8), mask=mask)

    @triton.jit
    def _avgpool3x3_kernel(
        x_ptr,
        out_ptr,
        out_offset: tl.constexpr,
        channels: tl.constexpr,
        total_c: tl.constexpr,
        h: tl.constexpr,
        w: tl.constexpr,
        hw_size: tl.constexpr,
        total: tl.constexpr,
        block_size: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < total

        ow = offsets % w
        tmp = offsets // w
        oh = tmp % h
        tmp = tmp // h
        channel = tmp % channels
        batch = tmp // channels
        base = batch * channels * hw_size + channel * hw_size

        acc = tl.zeros((block_size,), tl.float32)
        for kh in tl.static_range(0, 3):
            ih = oh + kh - 1
            valid_h = (ih >= 0) & (ih < h)
            for kw in tl.static_range(0, 3):
                iw = ow + kw - 1
                valid = mask & valid_h & (iw >= 0) & (iw < w)
                value = tl.load(x_ptr + base + ih * w + iw, mask=valid, other=0.0).to(tl.float32)
                acc += tl.where(valid, value, 0.0)

        store_offsets = batch * total_c * hw_size + (out_offset + channel) * hw_size + oh * w + ow
        tl.store(out_ptr + store_offsets, acc * (1.0 / 9.0), mask=mask)


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
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects 11 inputs, got {len(inputs)}")

    expected = (
        ("convolution_71", (N_BATCH, C1, HEIGHT, WIDTH), (C1 * SPATIAL, SPATIAL, WIDTH, 1)),
        ("arg429_1", (C1,), (1,)),
        ("arg430_1", (C1,), (1,)),
        ("arg431_1", (C1,), (1,)),
        ("arg432_1", (C1,), (1,)),
        ("convolution_75", (N_BATCH, C2, HEIGHT, WIDTH), (C2 * SPATIAL, SPATIAL, WIDTH, 1)),
        ("arg453_1", (C2,), (1,)),
        ("arg454_1", (C2,), (1,)),
        ("arg455_1", (C2,), (1,)),
        ("arg456_1", (C2,), (1,)),
        (
            "cat_7",
            (N_BATCH, C3, POOL_IN_HEIGHT, POOL_IN_WIDTH),
            (C3 * POOL_IN_SPATIAL, POOL_IN_SPATIAL, POOL_IN_WIDTH, 1),
        ),
    )
    tensors = tuple(
        _require_f32_tensor(name, value, shape, stride)
        for value, (name, shape, stride) in zip(inputs, expected)
    )
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all inputs must be on the same device")
    return tensors


def _bn_relu(
    x: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    channels: int,
) -> torch.Tensor:
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    y = torch.relu((x - mean) * torch.rsqrt(var + EPS) * weight.view(1, channels, 1, 1) + bias.view(1, channels, 1, 1))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean.squeeze((0, 2, 3)) * MOMENTUM)
    running_var.copy_(
        running_var * (1.0 - MOMENTUM)
        + var.squeeze((0, 2, 3)) * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return y


def _torch_oracle(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    (
        convolution_71,
        arg429_1,
        arg430_1,
        arg431_1,
        arg432_1,
        convolution_75,
        arg453_1,
        arg454_1,
        arg455_1,
        arg456_1,
        cat_7,
    ) = tensors

    relu_1 = _bn_relu(convolution_71, arg429_1, arg430_1, arg431_1, arg432_1, C1)
    relu_2 = _bn_relu(convolution_75, arg453_1, arg454_1, arg455_1, arg456_1, C2)
    max_values, max_indices = torch.ops.prims._low_memory_max_pool_with_offsets.default(
        cat_7,
        [3, 3],
        [2, 2],
        [0, 0],
        [1, 1],
        False,
    )
    out = F.avg_pool2d(torch.cat((relu_1, relu_2, max_values), dim=1), kernel_size=3, stride=1, padding=1)
    return (max_indices, out, arg429_1, arg430_1, arg453_1, arg454_1)


def _run_triton_oracle(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    (
        convolution_71,
        arg429_1,
        arg430_1,
        arg431_1,
        arg432_1,
        convolution_75,
        arg453_1,
        arg454_1,
        arg455_1,
        arg456_1,
        cat_7,
    ) = tensors

    device = convolution_71.device
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=device, dtype=torch.float32)
    max_values = torch.empty_strided(POOL_SHAPE, POOL_STRIDE, device=device, dtype=torch.float32)
    max_indices = torch.empty_strided(POOL_SHAPE, POOL_STRIDE, device=device, dtype=torch.int8)
    mean1 = torch.empty((C1,), device=device, dtype=torch.float32)
    invstd1 = torch.empty((C1,), device=device, dtype=torch.float32)
    mean2 = torch.empty((C2,), device=device, dtype=torch.float32)
    invstd2 = torch.empty((C2,), device=device, dtype=torch.float32)

    _dual_bn_stats_update_kernel[(C1 + C2,)](
        convolution_71,
        convolution_75,
        arg429_1,
        arg430_1,
        arg453_1,
        arg454_1,
        mean1,
        invstd1,
        mean2,
        invstd2,
        c1=C1,
        c2=C2,
        hw_size=SPATIAL,
        elems_per_channel=REDUCTION_SIZE,
        correction=RUNNING_VAR_CORRECTION,
        eps=EPS,
        momentum=MOMENTUM,
        block_size=STAT_BLOCK,
        num_warps=8,
        num_stages=4,
    )

    total1 = N_BATCH * C1 * SPATIAL
    _bn_relu_avgpool3x3_kernel[(triton.cdiv(total1, OUTPUT_BLOCK),)](
        convolution_71,
        arg431_1,
        arg432_1,
        mean1,
        invstd1,
        out,
        0,
        c=C1,
        total_c=TOTAL_C,
        h=HEIGHT,
        w=WIDTH,
        hw_size=SPATIAL,
        total=total1,
        block_size=OUTPUT_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    total2 = N_BATCH * C2 * SPATIAL
    _bn_relu_avgpool3x3_kernel[(triton.cdiv(total2, OUTPUT_BLOCK),)](
        convolution_75,
        arg455_1,
        arg456_1,
        mean2,
        invstd2,
        out,
        C1,
        c=C2,
        total_c=TOTAL_C,
        h=HEIGHT,
        w=WIDTH,
        hw_size=SPATIAL,
        total=total2,
        block_size=OUTPUT_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    total3 = N_BATCH * C3 * SPATIAL
    _maxpool2d_kernel[(triton.cdiv(total3, OUTPUT_BLOCK),)](
        cat_7,
        max_values,
        max_indices,
        total=total3,
        channels=C3,
        in_h=POOL_IN_HEIGHT,
        in_w=POOL_IN_WIDTH,
        out_h=HEIGHT,
        out_w=WIDTH,
        in_hw=POOL_IN_SPATIAL,
        out_hw=SPATIAL,
        block_size=OUTPUT_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _avgpool3x3_kernel[(triton.cdiv(total3, OUTPUT_BLOCK),)](
        max_values,
        out,
        C1 + C2,
        channels=C3,
        total_c=TOTAL_C,
        h=HEIGHT,
        w=WIDTH,
        hw_size=SPATIAL,
        total=total3,
        block_size=OUTPUT_BLOCK,
        num_warps=4,
        num_stages=4,
    )

    return (max_indices, out, arg429_1, arg430_1, arg453_1, arg454_1)


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
