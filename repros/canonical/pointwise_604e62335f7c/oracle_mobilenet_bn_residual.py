"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full MobileNetV2 BN-inference residual epilogue in one row-tiled Triton kernel, including fp32 mean/variance normalization, fp32 gamma/beta affine, the required fp16 rounding point, residual fp16 add, and final fp16 [128,160,7,7] output, whereas tuned Inductor already measures at the same CUDAGraph replay floor for this one-output pointwise epilogue; Inductor cannot materially improve this local graph because the remaining cost is dominated by the required convolution and residual reads, output write, fp32 element math, fp16 cast/add boundary, and launch replay overhead; the fix is BANDWIDTH_BOUND: record this as a full-scope at-floor artifact rather than pursuing a scheduler change for this repro."""
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


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---

if triton is not None:

    @triton.jit
    def _bn_residual_kernel(
        mean_ptr,
        convolution_ptr,
        var_ptr,
        gamma_ptr,
        beta_ptr,
        residual_ptr,
        output_ptr,
        rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.arange(0, BLOCK_HW)
        row_mask = row_offsets < rows
        hw_mask = hw_offsets < hw_size

        channel_offsets = row_offsets % channels
        mean = tl.load(mean_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        beta = tl.load(beta_ptr + channel_offsets, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.rsqrt(var + eps)

        offsets = row_offsets[:, None] * hw_size + hw_offsets[None, :]
        mask = row_mask[:, None] & hw_mask[None, :]
        convolution = tl.load(convolution_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

        bn_fp32 = (convolution - mean[:, None]) * invstd[:, None] * gamma[:, None] + beta[:, None]
        bn_fp16 = bn_fp32.to(tl.float16)
        output = residual + bn_fp16
        tl.store(output_ptr + offsets, output, mask=mask)


BATCH = 128
EPS = 1.0e-5


def _require_f16_tensor(
    name: str,
    value: Any,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, convolution, var, gamma, beta, residual = inputs
    convolution_t = _require_f16_tensor("convolution_47", convolution)
    if convolution_t.ndim != 4:
        raise ValueError(f"convolution_47 must be rank 4, got rank {convolution_t.ndim}")
    batch, channels, height, width = convolution_t.shape
    if batch != BATCH:
        raise ValueError(f"convolution_47 batch is {batch}, expected {BATCH}")

    channel_shape = (channels,)
    image_shape = (batch, channels, height, width)
    mean_t = _require_f16_tensor("arg237_1", mean, channel_shape)
    var_t = _require_f16_tensor("arg238_1", var, channel_shape)
    gamma_t = _require_f16_tensor("arg239_1", gamma, channel_shape)
    beta_t = _require_f16_tensor("arg240_1", beta, channel_shape)
    residual_t = _require_f16_tensor("add_98", residual, image_shape)

    if any(t.device != convolution_t.device for t in (mean_t, var_t, gamma_t, beta_t, residual_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return mean_t, convolution_t, var_t, gamma_t, beta_t, residual_t


def _kernel_config(hw_size: int) -> tuple[int, int, int]:
    block_hw = triton.next_power_of_2(hw_size)
    if hw_size <= 64:
        return 8, block_hw, 8
    if hw_size <= 256:
        return 4, block_hw, 4
    return 1, block_hw, 8


@oracle_impl(hardware="H100", shapes="(T([160], f16), T([128, 160, 7, 7], f16), T([160], f16), T([160], f16), T([160], f16), T([128, 160, 7, 7], f16))")
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
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_residual.py")

    mean, convolution, var, gamma, beta, residual = _validate_inputs(inputs)
    batch, channels, height, width = convolution.shape
    hw_size = height * width
    rows = batch * channels
    block_rows, block_hw, num_warps = _kernel_config(hw_size)

    output = torch.empty_like(convolution)
    _bn_residual_kernel[(triton.cdiv(rows, block_rows),)](
        mean,
        convolution,
        var,
        gamma,
        beta,
        residual,
        output,
        rows=rows,
        channels=channels,
        hw_size=hw_size,
        eps=EPS,
        BLOCK_ROWS=block_rows,
        BLOCK_HW=block_hw,
        num_warps=num_warps,
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
