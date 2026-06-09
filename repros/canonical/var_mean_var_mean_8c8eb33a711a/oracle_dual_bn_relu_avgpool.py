"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNeSt dual training-BatchNorm scope, including both population var/mean reductions over `[32,1024,14,14]`, four running-stat copy_ side effects, affine branch sum, ReLU, and final `2x2` avg_pool output in one channel-specialized Triton kernel while preserving the returned inverse-std tensors, mean tensors, and running-stat aliases, whereas Inductor lowers the captured sibling var_mean normalizations and downstream pooling through generic norm and pooling schedules; Inductor cannot do this today because norm-template canonicalization does not fuse paired mutable BN-training reductions with an immediate post-add spatial pooling consumer in one full-scope schedule; the fix is SCHEDULER_FUSION: teach the scheduler/norm template to keep paired BN stats, running-stat updates, affine sum/ReLU, and fixed pooling consumers in a single channel plan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

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


N = 32
CHANNELS = 1024
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
OUT_HEIGHT = 7
OUT_WIDTH = 7
OUT_HW = OUT_HEIGHT * OUT_WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
STAT_BLOCK = 8192
OUT_BLOCK = 2048


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _dual_bn_relu_avgpool_kernel(
        x0_ptr,
        x1_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        weight0_ptr,
        bias0_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        weight1_ptr,
        bias1_ptr,
        invstd0_ptr,
        invstd1_ptr,
        mean0_ptr,
        mean1_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        out_height: tl.constexpr,
        out_width: tl.constexpr,
        n_batches: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        STAT_BLOCK_: tl.constexpr,
        OUT_BLOCK_: tl.constexpr,
    ):
        channel = tl.program_id(0)

        stat_offsets = tl.arange(0, STAT_BLOCK_)
        stat_mask = stat_offsets < elements_per_channel
        n_offsets = stat_offsets // hw_size
        hw_offsets = stat_offsets - n_offsets * hw_size
        input_offsets = (n_offsets * channels + channel) * hw_size + hw_offsets

        x0 = tl.load(x0_ptr + input_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + input_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        count = elements_per_channel + 0.0
        mean0 = tl.sum(x0, axis=0) / count
        mean1 = tl.sum(x1, axis=0) / count

        centered0 = tl.where(stat_mask, x0 - mean0, 0.0)
        centered1 = tl.where(stat_mask, x1 - mean1, 0.0)
        var0 = tl.sum(centered0 * centered0, axis=0) / count
        var1 = tl.sum(centered1 * centered1, axis=0) / count
        var0 = tl.maximum(var0, 0.0)
        var1 = tl.maximum(var1, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)

        old_running_mean0 = tl.load(running_mean0_ptr + channel).to(tl.float32)
        old_running_var0 = tl.load(running_var0_ptr + channel).to(tl.float32)
        old_running_mean1 = tl.load(running_mean1_ptr + channel).to(tl.float32)
        old_running_var1 = tl.load(running_var1_ptr + channel).to(tl.float32)
        one_minus_momentum = 1.0 - momentum
        tl.store(running_mean0_ptr + channel, old_running_mean0 * one_minus_momentum + mean0 * momentum)
        tl.store(
            running_var0_ptr + channel,
            old_running_var0 * one_minus_momentum + var0 * running_var_correction * momentum,
        )
        tl.store(running_mean1_ptr + channel, old_running_mean1 * one_minus_momentum + mean1 * momentum)
        tl.store(
            running_var1_ptr + channel,
            old_running_var1 * one_minus_momentum + var1 * running_var_correction * momentum,
        )
        tl.store(invstd0_ptr + channel, invstd0)
        tl.store(invstd1_ptr + channel, invstd1)
        tl.store(mean0_ptr + channel, mean0)
        tl.store(mean1_ptr + channel, mean1)

        out_offsets = tl.arange(0, OUT_BLOCK_)
        out_elems_per_channel: tl.constexpr = n_batches * out_height * out_width
        out_mask = out_offsets < out_elems_per_channel
        out_n = out_offsets // (out_height * out_width)
        out_rem = out_offsets - out_n * out_height * out_width
        out_h = out_rem // out_width
        out_w = out_rem - out_h * out_width

        weight0 = tl.load(weight0_ptr + channel).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel).to(tl.float32)
        acc = tl.zeros((OUT_BLOCK_,), tl.float32)
        denom = tl.zeros((OUT_BLOCK_,), tl.float32)

        for kh in tl.static_range(0, 2):
            in_h = out_h * 2 + kh
            valid_h = in_h < height
            for kw in tl.static_range(0, 2):
                in_w = out_w * 2 + kw
                valid = out_mask & valid_h & (in_w < width)
                in_hw = in_h * width + in_w
                offsets = (out_n * channels + channel) * hw_size + in_hw
                val0 = tl.load(x0_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
                val1 = tl.load(x1_ptr + offsets, mask=valid, other=0.0).to(tl.float32)
                y0 = (val0 - mean0) * invstd0 * weight0 + bias0
                y1 = (val1 - mean1) * invstd1 * weight1 + bias1
                summed = y0 + y1
                relu = tl.where((summed > 0.0) | (summed != summed), summed, 0.0)
                acc += tl.where(valid, relu, 0.0)
                denom += tl.where(valid, 1.0, 0.0)

        pooled = acc / denom
        store_offsets = (out_n * channels + channel) * (out_height * out_width) + out_h * out_width + out_w
        tl.store(out_ptr + store_offsets, pooled, mask=out_mask)


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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects ten inputs, got {len(inputs)}")

    (
        convolution_19,
        arg108_1,
        arg109_1,
        arg110_1,
        arg111_1,
        convolution_20,
        arg114_1,
        arg115_1,
        arg116_1,
        arg117_1,
    ) = inputs

    nchw_shape = (N, CHANNELS, HEIGHT, WIDTH)
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    tensors = (
        _expect_tensor("convolution_19", convolution_19, nchw_shape, nchw_stride),
        _expect_tensor("arg108_1", arg108_1, vector_shape, vector_stride),
        _expect_tensor("arg109_1", arg109_1, vector_shape, vector_stride),
        _expect_tensor("arg110_1", arg110_1, vector_shape, vector_stride),
        _expect_tensor("arg111_1", arg111_1, vector_shape, vector_stride),
        _expect_tensor("convolution_20", convolution_20, nchw_shape, nchw_stride),
        _expect_tensor("arg114_1", arg114_1, vector_shape, vector_stride),
        _expect_tensor("arg115_1", arg115_1, vector_shape, vector_stride),
        _expect_tensor("arg116_1", arg116_1, vector_shape, vector_stride),
        _expect_tensor("arg117_1", arg117_1, vector_shape, vector_stride),
    )
    device = tensors[0].device
    if any(t.device != device for t in tensors):
        raise ValueError("all tensor inputs must be on the same device")
    return tensors


def _torch_oracle(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
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
    ) = tensors

    var0, mean0 = torch.var_mean(x0, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd0_4d = torch.rsqrt(var0 + EPS)
    var1, mean1 = torch.var_mean(x1, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd1_4d = torch.rsqrt(var1 + EPS)

    y0 = (x0 - mean0) * invstd0_4d * weight0.view(1, CHANNELS, 1, 1) + bias0.view(1, CHANNELS, 1, 1)
    y1 = (x1 - mean1) * invstd1_4d * weight1.view(1, CHANNELS, 1, 1) + bias1.view(1, CHANNELS, 1, 1)
    pooled = F.avg_pool2d(torch.relu(y0 + y1), kernel_size=2, stride=2, padding=0, ceil_mode=True)

    mean0_1d = mean0.squeeze((0, 2, 3))
    var0_1d = var0.squeeze((0, 2, 3))
    mean1_1d = mean1.squeeze((0, 2, 3))
    var1_1d = var1.squeeze((0, 2, 3))
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

    return (
        invstd0_4d.squeeze((0, 2, 3)),
        invstd1_4d.squeeze((0, 2, 3)),
        pooled,
        mean1_1d.view(1, CHANNELS, 1, 1),
        mean0_1d.view(1, CHANNELS, 1, 1),
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )


def _run_triton_oracle(tensors: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        return _torch_oracle(tensors)

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
    ) = tensors
    device = x0.device

    invstd0 = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    invstd1 = torch.empty_strided((CHANNELS,), (1,), device=device, dtype=torch.float32)
    mean0 = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=device, dtype=torch.float32)
    mean1 = torch.empty_strided((1, CHANNELS, 1, 1), (CHANNELS, 1, 1, 1), device=device, dtype=torch.float32)
    pooled = torch.empty_strided(
        (N, CHANNELS, OUT_HEIGHT, OUT_WIDTH),
        (CHANNELS * OUT_HW, OUT_HW, OUT_WIDTH, 1),
        device=device,
        dtype=torch.float32,
    )

    _dual_bn_relu_avgpool_kernel[(CHANNELS,)](
        x0,
        x1,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        invstd0,
        invstd1,
        mean0,
        mean1,
        pooled,
        channels=CHANNELS,
        hw_size=HW,
        height=HEIGHT,
        width=WIDTH,
        out_height=OUT_HEIGHT,
        out_width=OUT_WIDTH,
        n_batches=N,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        STAT_BLOCK_=STAT_BLOCK,
        OUT_BLOCK_=OUT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    return (
        invstd0,
        invstd1,
        pooled,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )


@oracle_impl(hardware="H100", shapes="(T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32))")
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
    if tensors[0].is_cuda:
        return _run_triton_oracle(tensors)
    return _torch_oracle(tensors)


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
