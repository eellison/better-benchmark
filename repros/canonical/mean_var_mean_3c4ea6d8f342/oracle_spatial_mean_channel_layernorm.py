"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete residual add, 7x7 per-channel spatial mean, 640-wide channel variance/mean, affine layernorm, and final contiguous [128,640] view with Triton kernels that only stage the small [batch,channel] spatial means, whereas Inductor is already faster on the required harness for this full scope; Inductor cannot materially improve this repro through a narrower norm-template fusion change because the remaining work is dominated by mandatory spatial and channel reductions plus output traffic that the compiler already handles efficiently; the fix is BANDWIDTH_BOUND: record this full-scope prototype as not a true floor and do not open a scheduler-fusion action from this measurement."""
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
BATCH = 128
CHANNELS = 640
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-6
SPATIAL_BLOCK_C = 32
FINAL_BLOCK_C = 1024
CONTIGUOUS_MEAN_BLOCK_X = 16
SUPPORTED_INPUT_STRIDES = {
    (CHANNELS * HW, HW, WIDTH, 1),
    (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS),
}
CONTIGUOUS_INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
OUTPUT_STRIDE = (CHANNELS, 1)

if triton is not None:

    @triton.jit
    def _contiguous_spatial_mean_kernel(
        x_ptr,
        residual_ptr,
        out_ptr,
        xnumel: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        x_offsets = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)[:, None]
        hw_offsets = tl.arange(0, BLOCK_HW)[None, :]
        x_mask = x_offsets < xnumel
        hw_mask = hw_offsets < hw
        offsets = x_offsets * hw + hw_offsets
        summed = (
            tl.load(x_ptr + offsets, mask=x_mask & hw_mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + offsets, mask=x_mask & hw_mask, other=0.0).to(tl.float32)
        )
        spatial_mean = tl.sum(tl.where(hw_mask, summed, 0.0), axis=1) / hw
        tl.store(out_ptr + x_offsets, spatial_mean, mask=x_mask)

    @triton.jit
    def _spatial_mean_partials_kernel(
        x_ptr,
        residual_ptr,
        means_ptr,
        x_s0: tl.constexpr,
        x_s1: tl.constexpr,
        x_s2: tl.constexpr,
        x_s3: tl.constexpr,
        residual_s0: tl.constexpr,
        residual_s1: tl.constexpr,
        residual_s2: tl.constexpr,
        residual_s3: tl.constexpr,
        channels: tl.constexpr,
        height: tl.constexpr,
        width: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_block = tl.program_id(1)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        c_mask = c_offsets < channels
        hw_mask = hw_offsets < hw
        h_offsets = hw_offsets // width
        w_offsets = hw_offsets - h_offsets * width

        x_offsets = (
            batch * x_s0
            + c_offsets[:, None] * x_s1
            + h_offsets[None, :] * x_s2
            + w_offsets[None, :] * x_s3
        )
        residual_offsets = (
            batch * residual_s0
            + c_offsets[:, None] * residual_s1
            + h_offsets[None, :] * residual_s2
            + w_offsets[None, :] * residual_s3
        )
        mask = c_mask[:, None] & hw_mask[None, :]
        summed = (
            tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
            + tl.load(residual_ptr + residual_offsets, mask=mask, other=0.0).to(tl.float32)
        )
        spatial_mean = tl.sum(tl.where(hw_mask[None, :], summed, 0.0), axis=1) / hw

        mean_offsets = batch * channels + c_offsets
        tl.store(means_ptr + mean_offsets, spatial_mean, mask=c_mask)

    @triton.jit
    def _final_affine_kernel(
        means_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.arange(0, BLOCK_C)
        c_mask = c_offsets < channels
        mean_offsets = batch * channels + c_offsets
        spatial_mean = tl.load(means_ptr + mean_offsets, mask=c_mask, other=0.0).to(tl.float32)
        valid_mean = tl.where(c_mask, spatial_mean, 0.0)
        channel_mean = tl.sum(valid_mean, axis=0) / channels
        centered = spatial_mean - channel_mean
        channel_var = tl.maximum(tl.sum(tl.where(c_mask, centered * centered, 0.0), axis=0) / channels, 0.0)
        invstd = tl.rsqrt(channel_var + eps)

        weight = tl.load(weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + mean_offsets, centered * invstd * weight + bias, mask=c_mask)


def _expect_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...],
    *,
    strides: set[tuple[int, ...]] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if strides is not None:
        actual_stride = tuple(value.stride())
        if isinstance(strides, set):
            ok = actual_stride in strides
        else:
            ok = actual_stride == strides
        if not ok:
            raise ValueError(f"{name} has stride {actual_stride}, expected {strides}")
    return value


def _expect_shape(name: str, value: Any, expected: tuple[int, ...]) -> None:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a shape list/tuple, got {type(value)!r}")
    actual = tuple(int(dim) for dim in value)
    if actual != expected:
        raise ValueError(f"{name} is {actual}, expected {expected}")


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, residual, weight, bias, output_shape = inputs
    x = _expect_tensor(
        "convolution_45",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        strides=SUPPORTED_INPUT_STRIDES,
    )
    residual = _expect_tensor(
        "add_85",
        residual,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        strides=SUPPORTED_INPUT_STRIDES,
    )
    weight = _expect_tensor("arg157_1", weight, (CHANNELS,), strides=(1,))
    bias = _expect_tensor("arg158_1", bias, (CHANNELS,), strides=(1,))
    _expect_shape("_shape_param_0", output_shape, (BATCH, CHANNELS))

    if not (x.device == residual.device == weight.device == bias.device):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x, residual, weight, bias, (BATCH, CHANNELS)


def oracle_forward(inputs):
    """Run the full Repro.forward computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_spatial_mean_channel_layernorm.py")

    x, residual, weight, bias, output_shape = _validate_inputs(inputs)
    num_c_blocks = triton.cdiv(CHANNELS, SPATIAL_BLOCK_C)
    output = torch.empty_strided(
        output_shape,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    if tuple(x.stride()) == CONTIGUOUS_INPUT_STRIDE and tuple(residual.stride()) == CONTIGUOUS_INPUT_STRIDE:
        _contiguous_spatial_mean_kernel[(triton.cdiv(BATCH * CHANNELS, CONTIGUOUS_MEAN_BLOCK_X),)](
            x,
            residual,
            output,
            xnumel=BATCH * CHANNELS,
            hw=HW,
            BLOCK_X=CONTIGUOUS_MEAN_BLOCK_X,
            BLOCK_HW=triton.next_power_of_2(HW),
            num_warps=1,
            num_stages=3,
        )
    else:
        _spatial_mean_partials_kernel[(BATCH, num_c_blocks)](
            x,
            residual,
            output,
            x_s0=x.stride(0),
            x_s1=x.stride(1),
            x_s2=x.stride(2),
            x_s3=x.stride(3),
            residual_s0=residual.stride(0),
            residual_s1=residual.stride(1),
            residual_s2=residual.stride(2),
            residual_s3=residual.stride(3),
            channels=CHANNELS,
            height=HEIGHT,
            width=WIDTH,
            hw=HW,
            BLOCK_C=SPATIAL_BLOCK_C,
            BLOCK_HW=triton.next_power_of_2(HW),
            num_warps=4,
            num_stages=3,
        )
    _final_affine_kernel[(BATCH,)](
        output,
        weight,
        bias,
        output,
        channels=CHANNELS,
        eps=EPS,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    return output


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
