"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the full DCGAN BatchNorm-inference affine plus leaky-ReLU pointwise scope with a channel/spatial Triton tile that hoists channel-only mean, variance, weight, bias, and sqrt/reciprocal work once per channel tile row before broadcasting across spatial elements, whereas Inductor emits one flat fused pointwise kernel that repeats the broadcast arithmetic in the output-element loop; Inductor does not generate this broadcast-invariant channel tile today because the pointwise scheduler treats the NCHW result as one flat iteration domain, but CUDAGraph measurement shows the hoist does not buy a material win because the required convolution read and output write dominate; the fix is BANDWIDTH_BOUND: record this repro as at floor unless broader pointwise bandwidth/codegen work moves both implementations together."""
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
CLASSIFICATION = "BANDWIDTH_BOUND"
EPS = 1.0e-5

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
if triton is not None:

    @triton.jit
    def _channel_affine_leaky_relu_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        eps: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)

        valid_c = c_offsets < channels
        valid_hw = hw_offsets < hw
        offsets = batch * channels * hw + c_offsets[:, None] * hw + hw_offsets[None, :]
        mask = valid_c[:, None] & valid_hw[None, :]

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=valid_c, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c_offsets, mask=valid_c, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=valid_c, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=valid_c, other=0.0).to(tl.float32)

        scale = (1.0 / tl.sqrt_rn(var + eps)) * weight
        affine = (x - mean[:, None]) * scale[:, None] + bias[:, None]
        out = tl.where(affine > 0.0, affine, affine * 0.2)
        tl.store(out_ptr + offsets, out, mask=mask)


def _require_f32_cuda_tensor(name: str, value: Any) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.dtype is not torch.float32:
        raise TypeError(f"{name} must be torch.float32, got {value.dtype}")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    mean = _require_f32_cuda_tensor("arg13_1", inputs[0])
    x = _require_f32_cuda_tensor("convolution_3", inputs[1])
    var = _require_f32_cuda_tensor("arg14_1", inputs[2])
    weight = _require_f32_cuda_tensor("arg15_1", inputs[3])
    bias = _require_f32_cuda_tensor("arg16_1", inputs[4])

    if x.ndim != 4:
        raise ValueError(f"convolution_3 must be rank 4, got rank {x.ndim}")
    batch, channels, height, width = x.shape
    if tuple(mean.shape) != (channels,):
        raise ValueError(f"arg13_1 shape must be {(channels,)}, got {tuple(mean.shape)}")
    for name, tensor in (("arg14_1", var), ("arg15_1", weight), ("arg16_1", bias)):
        if tuple(tensor.shape) != (channels,):
            raise ValueError(f"{name} shape must be {(channels,)}, got {tuple(tensor.shape)}")
        if tensor.device != x.device:
            raise ValueError(f"{name} must be on the same device as convolution_3")

    expected_stride = (channels * height * width, height * width, width, 1)
    if tuple(x.stride()) != expected_stride or x.storage_offset() != 0:
        raise ValueError(
            "convolution_3 must be contiguous NCHW with storage_offset=0, "
            f"got stride={tuple(x.stride())} storage_offset={x.storage_offset()}"
        )
    return mean, x, var, weight, bias


def _tile_shape(channels: int, hw: int) -> tuple[int, int]:
    if hw <= 16:
        return min(256, channels), 16
    if hw <= 64:
        return min(16, channels), 64
    return min(4, channels), 256


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
    batch, channels, height, width = x.shape
    hw = height * width
    output = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x.device,
        dtype=torch.float32,
    )
    block_c, block_hw = _tile_shape(channels, hw)
    grid = (batch, triton.cdiv(channels, block_c), triton.cdiv(hw, block_hw))
    _channel_affine_leaky_relu_kernel[grid](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        channels=channels,
        hw=hw,
        BLOCK_C=block_c,
        BLOCK_HW=block_hw,
        eps=EPS,
        num_warps=4,
        num_stages=1,
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
