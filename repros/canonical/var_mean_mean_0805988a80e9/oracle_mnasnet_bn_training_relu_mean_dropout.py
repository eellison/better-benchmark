"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MnasNet training BatchNorm var_mean, mutable running-stat copy_ updates, affine ReLU, spatial mean, and Inductor-RNG dropout-shaped return with Triton reduction and epilogue kernels, whereas Inductor currently schedules the BN-training statistics/update, normalized activation, ReLU spatial mean, and stochastic dropout consumer as separate generic regions; Inductor cannot do this today because scheduler fusion cannot sink a normalization reduction with mutable running-stat side outputs into a downstream activation reduction and stochastic epilogue; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat epilogues while fusing immediate affine activation, spatial-reduction, and dropout consumers."""
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
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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


BATCH = 256
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
TOTAL_ROWS = BATCH * CHANNELS
ELEMENTS_PER_CHANNEL = BATCH * HW
EPS = 1.0e-5
MOMENTUM = 0.00029999999999996696
RUNNING_VAR_CORRECTION = 1.0000797257434426
DROPOUT_P = 0.2
DROPOUT_SCALE = 1.25
STAT_BLOCK_R = 512
STAT_BLOCK_C = 16
STAT_BLOCKS = (ELEMENTS_PER_CHANNEL + STAT_BLOCK_R - 1) // STAT_BLOCK_R
FINAL_BLOCK_B = 32
FINAL_BLOCK_C = 16
OUTPUT_BLOCK_ROWS = 32
OUTPUT_BLOCK_HW = 64
DROPOUT_BLOCK = 512

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_tiled_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        elements_per_channel: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
        c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        r_mask = r < elements_per_channel
        c_mask = c < channels

        hw = height * width
        n = r // hw
        rem = r - n * hw
        h = rem // width
        w = rem - h * width
        offsets = (
            n[:, None] * x_stride_n
            + c[None, :] * x_stride_c
            + h[:, None] * x_stride_h
            + w[:, None] * x_stride_w
        )
        mask = r_mask[:, None] & c_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        sums = tl.sum(x, axis=0)
        sumsq = tl.sum(x * x, axis=0)
        out_offsets = tl.program_id(0) * channels + c
        tl.store(partial_sum_ptr + out_offsets, sums, mask=c_mask)
        tl.store(partial_sumsq_ptr + out_offsets, sumsq, mask=c_mask)

    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        invstd_ptr,
        running_mean_ptr,
        running_var_ptr,
        channels: tl.constexpr,
        num_stat_blocks: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        b = tl.arange(0, BLOCK_B)
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (b[:, None] < num_stat_blocks) & (c[None, :] < channels)
        offsets = b[:, None] * channels + c[None, :]

        sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sumsq = tl.load(partial_sumsq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = elements_per_channel + 0.0
        mean = tl.sum(sums, axis=0) / total
        var = tl.sum(sumsq, axis=0) / total - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        c_mask = c < channels
        old_running_mean = tl.load(running_mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(running_mean_ptr + c, old_running_mean * (1.0 - momentum) + mean * momentum, mask=c_mask)
        tl.store(
            running_var_ptr + c,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask,
        )
        tl.store(mean_ptr + c, mean, mask=c_mask)
        tl.store(invstd_ptr + c, invstd, mask=c_mask)

    @triton.jit
    def _relu_spatial_mean_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        pooled_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.arange(0, BLOCK_HW)
        row_mask = rows < total_rows
        hw_mask = hw_offsets < hw
        n = rows // channels
        c = rows - n * channels
        offsets = (n[:, None] * channels + c[:, None]) * hw + hw_offsets[None, :]
        mask = row_mask[:, None] & hw_mask[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=row_mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * invstd[:, None]
        y = y * weight[:, None] + bias[:, None]
        y = tl.maximum(y, 0.0)
        pooled = tl.sum(tl.where(mask, y, 0.0), axis=1) * (1.0 / 49.0)
        tl.store(pooled_ptr + rows, pooled, mask=row_mask)

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
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_51, arg309_1, arg310_1, arg311_1, arg312_1 = inputs
    x = _expect_f32_tensor("convolution_51", convolution_51, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg309_1", arg309_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg310_1", arg310_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg311_1", arg311_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg312_1", arg312_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope, including running-stat mutations."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mnasnet_bn_training_relu_mean_dropout.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    seed = torch.ops.prims.inductor_seeds.default(1, device=x.device)
    partial_sum = torch.empty_strided(
        (STAT_BLOCKS, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sumsq = torch.empty_like(partial_sum)
    mean = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    pooled = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)
    out = torch.empty_strided(OUTPUT_SHAPE, OUTPUT_STRIDE, device=x.device, dtype=torch.float32)

    _partial_stats_tiled_kernel[(STAT_BLOCKS, triton.cdiv(CHANNELS, STAT_BLOCK_C))](
        x,
        partial_sum,
        partial_sumsq,
        channels=CHANNELS,
        height=HEIGHT,
        width=WIDTH,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        BLOCK_R=STAT_BLOCK_R,
        BLOCK_C=STAT_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(CHANNELS, FINAL_BLOCK_C),)](
        partial_sum,
        partial_sumsq,
        mean,
        invstd,
        running_mean,
        running_var,
        channels=CHANNELS,
        num_stat_blocks=STAT_BLOCKS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_B=FINAL_BLOCK_B,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=1,
        num_stages=3,
    )
    _relu_spatial_mean_kernel[(triton.cdiv(TOTAL_ROWS, OUTPUT_BLOCK_ROWS),)](
        x,
        mean,
        invstd,
        weight,
        bias,
        pooled,
        total_rows=TOTAL_ROWS,
        channels=CHANNELS,
        hw=HW,
        BLOCK_ROWS=OUTPUT_BLOCK_ROWS,
        BLOCK_HW=OUTPUT_BLOCK_HW,
        num_warps=1,
        num_stages=3,
    )
    _dropout_kernel[(triton.cdiv(TOTAL_ROWS, DROPOUT_BLOCK),)](
        pooled,
        seed,
        out,
        total=TOTAL_ROWS,
        dropout_p=DROPOUT_P,
        dropout_scale=DROPOUT_SCALE,
        BLOCK=DROPOUT_BLOCK,
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
