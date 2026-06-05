"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 training BatchNorm var_mean, running-stat copy_ updates, affine ReLU6, spatial mean, and Inductor-RNG dropout return with a channel-tiled Triton pipeline, whereas Inductor currently schedules the BN-training reduction/update and the downstream ReLU6 spatial-mean/dropout consumer as separate generic regions; Inductor cannot do this today because scheduler fusion cannot sink a normalization reduction with mutable running-stat side outputs into a following reduction and stochastic pointwise epilogue; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat epilogues while fusing immediate affine activation, spatial reduction, and dropout consumers."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
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


BATCH = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
DROPOUT_P = 0.2
DROPOUT_SCALE = 1.25
PARTIAL_BLOCK = 4096
NUM_PARTIALS = (ELEMENTS_PER_CHANNEL + PARTIAL_BLOCK - 1) // PARTIAL_BLOCK
FINAL_BLOCK = 2
OUTPUT_BLOCK_ROWS = 8
BLOCK_HW = 64

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        elements_per_channel: tl.constexpr,
        num_partials: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        partial_id = tl.program_id(1)
        offsets = partial_id * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = offsets < elements_per_channel

        n_idx = offsets // hw
        hw_idx = offsets - n_idx * hw
        x_offsets = (n_idx * channels + channel) * hw + hw_idx
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        partial_offset = channel * num_partials + partial_id
        tl.store(partial_sum_ptr + partial_offset, tl.sum(x, axis=0))
        tl.store(partial_sumsq_ptr + partial_offset, tl.sum(x * x, axis=0))

    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        invstd_ptr,
        running_mean_ptr,
        running_var_ptr,
        elements_per_channel: tl.constexpr,
        num_partials: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_PARTIALS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_PARTIALS)
        mask = offsets < num_partials
        partial_offsets = channel * num_partials + offsets

        sum_x = tl.sum(tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0), axis=0)
        sum_x2 = tl.sum(tl.load(partial_sumsq_ptr + partial_offsets, mask=mask, other=0.0), axis=0)
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
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _relu6_mean_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_SPATIAL: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = rows < total_rows
        n_offsets = rows // channels
        channel = rows - n_offsets * channels
        hw_offsets = tl.arange(0, BLOCK_SPATIAL)
        valid_hw = hw_offsets < hw
        valid = row_mask[:, None] & valid_hw[None, :]

        x_offsets = (n_offsets[:, None] * channels + channel[:, None]) * hw + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * invstd[:, None] * weight[:, None] + bias[:, None]
        y = tl.minimum(tl.maximum(y, 0.0), 6.0)
        pooled = tl.sum(tl.where(valid, y, 0.0), axis=1) * (1.0 / 49.0)
        tl.store(out_ptr + rows, pooled, mask=row_mask)

    @triton.jit
    def _dropout_kernel(
        pooled_ptr,
        seed_ptr,
        out_ptr,
        total: tl.constexpr,
        dropout_p: tl.constexpr,
        dropout_scale: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        pooled = tl.load(pooled_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        seed = tl.load(seed_ptr)
        keep = tl.rand(seed, offsets.to(tl.uint32)) > dropout_p
        out = tl.where(keep, pooled * dropout_scale, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)

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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    convolution_51, arg309_1, arg310_1, arg311_1, arg312_1, shape_param = inputs
    x = _expect_f32_tensor("convolution_51", convolution_51, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg309_1", arg309_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg310_1", arg310_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg311_1", arg311_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg312_1", arg312_1, STAT_SHAPE, STAT_STRIDE)

    if tuple(int(dim) for dim in shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_training_relu6_mean_dropout.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    seed = torch.ops.prims.inductor_seeds.default(1, device=x.device)
    partial_sum = torch.empty_strided(
        (CHANNELS, NUM_PARTIALS),
        (NUM_PARTIALS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sumsq = torch.empty_strided(
        (CHANNELS, NUM_PARTIALS),
        (NUM_PARTIALS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    pooled = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(CHANNELS, NUM_PARTIALS)](
        x,
        partial_sum,
        partial_sumsq,
        channels=CHANNELS,
        hw=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        num_partials=NUM_PARTIALS,
        BLOCK_K=PARTIAL_BLOCK,
        num_warps=8,
        num_stages=4,
    )
    _finalize_stats_kernel[(CHANNELS,)](
        partial_sum,
        partial_sumsq,
        mean,
        invstd,
        running_mean,
        running_var,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        num_partials=NUM_PARTIALS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_PARTIALS=FINAL_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _relu6_mean_kernel[(triton.cdiv(BATCH * CHANNELS, OUTPUT_BLOCK_ROWS),)](
        x,
        mean,
        invstd,
        weight,
        bias,
        pooled,
        total_rows=BATCH * CHANNELS,
        channels=CHANNELS,
        hw=HW,
        BLOCK_ROWS=OUTPUT_BLOCK_ROWS,
        BLOCK_SPATIAL=BLOCK_HW,
        num_warps=1,
        num_stages=3,
    )
    _dropout_kernel[(triton.cdiv(BATCH * CHANNELS, 256),)](
        pooled,
        seed,
        out,
        total=BATCH * CHANNELS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK=256,
        num_warps=4,
        num_stages=3,
    )
    return out, running_mean, running_var


def _has_inductor_random() -> bool:
    return "inductor_random" in REPRO_PATH.read_text()


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
    if has_stochastic_ops(REPRO_PATH) or _has_inductor_random():
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
