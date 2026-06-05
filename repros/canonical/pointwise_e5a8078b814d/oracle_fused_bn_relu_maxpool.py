"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete BatchNorm-affine, ReLU, and 3x3 stride-2 max-pool stencil into one Triton kernel that writes only the final channels-last pooled tensor, whereas Inductor materializes the full normalized/affine and ReLU tensors before the pooling stencil; Inductor cannot do this today because its scheduler does not fuse producer pointwise work through prims._low_memory_max_pool_with_offsets when only the pooled values escape; the fix is SCHEDULER_FUSION: teach the max-pool lowering to absorb elementwise producers and elide the unused offsets output."""
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


EPS = 0.001


def _channels_last_nchw_stride(
    batches: int,
    channels: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    del batches
    return (channels * height * width, 1, channels * width, channels)


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")
    mean, x, var, weight, bias = inputs
    tensors = (mean, x, var, weight, bias)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all inputs must be tensors")
    if x.ndim != 4:
        raise ValueError(f"expected NCHW input, got shape={tuple(x.shape)}")
    if x.dtype != torch.float32 or any(tensor.dtype != torch.float32 for tensor in (mean, var, weight, bias)):
        raise ValueError("this oracle expects f32 tensors")
    if not x.is_cuda or any(not tensor.is_cuda for tensor in (mean, var, weight, bias)):
        raise RuntimeError("CUDA tensors are required for this Triton oracle")

    batches, channels, height, width = (int(dim) for dim in x.shape)
    if height < 3 or width < 3:
        raise ValueError(f"max-pool input is too small: H={height}, W={width}")
    expected_param_shape = (channels,)
    if any(tuple(tensor.shape) != expected_param_shape for tensor in (mean, var, weight, bias)):
        raise ValueError(
            f"expected BN parameter shape {expected_param_shape}, got "
            f"mean={tuple(mean.shape)} var={tuple(var.shape)} "
            f"weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
        )
    expected_stride = _channels_last_nchw_stride(batches, channels, height, width)
    if tuple(x.stride()) != expected_stride:
        raise ValueError(f"expected channels-last NCHW input stride {expected_stride}, got {tuple(x.stride())}")
    if any(not tensor.is_contiguous() for tensor in (mean, var, weight, bias)):
        raise ValueError("expected contiguous 1D BN parameter tensors")
    return mean, x, var, weight, bias


# --- Oracle kernel(s) ---
# One program computes a tile of output columns and channels for one (N, Hout).
# The flattened physical layout is channels-last NCHW, so each output tile maps
# to contiguous stores and contiguous 3x3 stencil loads.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_OW": 8, "BLOCK_C": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_OW": 16, "BLOCK_C": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_OW": 8, "BLOCK_C": 128}, num_warps=8, num_stages=3),
        ],
        key=["CHANNELS", "OUT_HEIGHT", "OUT_WIDTH"],
    )
    @triton.jit
    def _fused_bn_relu_maxpool_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        CHANNELS: tl.constexpr,
        IN_HEIGHT: tl.constexpr,
        IN_WIDTH: tl.constexpr,
        OUT_HEIGHT: tl.constexpr,
        OUT_WIDTH: tl.constexpr,
        BLOCK_OW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ow_blocks = tl.cdiv(OUT_WIDTH, BLOCK_OW)
        n_c_blocks = tl.cdiv(CHANNELS, BLOCK_C)

        c_block = pid % n_c_blocks
        pid = pid // n_c_blocks
        ow_block = pid % n_ow_blocks
        pid = pid // n_ow_blocks
        out_h = pid % OUT_HEIGHT
        batch = pid // OUT_HEIGHT

        ow = ow_block * BLOCK_OW + tl.arange(0, BLOCK_OW)
        channel = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = (ow[:, None] < OUT_WIDTH) & (channel[None, :] < CHANNELS)

        mean = tl.load(mean_ptr + channel, mask=channel < CHANNELS, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=channel < CHANNELS, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=channel < CHANNELS, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=channel < CHANNELS, other=0.0).to(tl.float32)
        inv_std = 1.0 / tl.sqrt(var + 0.001)

        in_h0 = out_h * 2
        in_w0 = ow * 2
        best = tl.full((BLOCK_OW, BLOCK_C), -float("inf"), tl.float32)

        for kh in tl.static_range(3):
            in_h = in_h0 + kh
            for kw in tl.static_range(3):
                in_w = in_w0 + kw
                x_offsets = ((batch * IN_HEIGHT + in_h) * IN_WIDTH + in_w[:, None]) * CHANNELS + channel[None, :]
                value = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
                value = (value - mean[None, :]) * inv_std[None, :]
                value = value * weight[None, :] + bias[None, :]
                value = tl.where(value < 0.0, 0.0, value)
                best_is_nan = best != best
                value_is_nan = value != value
                take_value = value_is_nan | ((value > best) & ~best_is_nan)
                best = tl.where(best_is_nan, best, tl.where(take_value, value, best))

        out_offsets = ((batch * OUT_HEIGHT + out_h) * OUT_WIDTH + ow[:, None]) * CHANNELS + channel[None, :]
        tl.store(out_ptr + out_offsets, best, mask=mask)


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
        raise RuntimeError("Triton is required for this oracle")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    batches, channels, in_height, in_width = (int(dim) for dim in x.shape)
    out_height = (in_height - 3) // 2 + 1
    out_width = (in_width - 3) // 2 + 1

    out = torch.empty_strided(
        (batches, channels, out_height, out_width),
        _channels_last_nchw_stride(batches, channels, out_height, out_width),
        device=x.device,
        dtype=x.dtype,
    )
    grid = lambda meta: (
        batches
        * out_height
        * triton.cdiv(out_width, meta["BLOCK_OW"])
        * triton.cdiv(channels, meta["BLOCK_C"]),
    )
    _fused_bn_relu_maxpool_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        out,
        CHANNELS=channels,
        IN_HEIGHT=in_height,
        IN_WIDTH=in_width,
        OUT_HEIGHT=out_height,
        OUT_WIDTH=out_width,
    )
    return out


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
