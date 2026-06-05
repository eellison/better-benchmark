"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full fp32 BN-inference affine for `convolution_37`, rounds that result to fp16 before ReLU, and writes every segment directly into the final cat destination, whereas Inductor schedules the BN/ReLU producer and cat materialization as generic pointwise/copy work with an intermediate boundary; Inductor cannot do this today because its scheduler does not virtualize a computed producer into one segment of the cat destination while hoisting channel-invariant BN scalars; the fix is SCHEDULER_FUSION: sink BN-inference pointwise producers into cat materialization so the final concat can be virtualized with the computed last segment."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
        ],
        key=["BN_CHANNELS", "RELU32_CHANNELS", "PREFIX_CHANNELS", "OUT_CHANNELS", "HW"],
    )
    @triton.jit
    def _copy_cat_prefix_kernel(
        relu32_ptr,
        relu33_ptr,
        relu34_ptr,
        relu35_ptr,
        relu36_ptr,
        out_ptr,
        BN_CHANNELS: tl.constexpr,
        RELU32_CHANNELS: tl.constexpr,
        PREFIX_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c_out = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        hw_mask = hw < HW
        out_mask = (c_out[:, None] < PREFIX_CHANNELS) & hw_mask[None, :]
        out_offsets = n * OUT_CHANNELS * HW + c_out[:, None] * HW + hw[None, :]

        relu33_start = RELU32_CHANNELS
        relu34_start = relu33_start + BN_CHANNELS
        relu35_start = relu34_start + BN_CHANNELS
        relu36_start = relu35_start + BN_CHANNELS

        vals = tl.zeros((BLOCK_C, BLOCK_HW), tl.float32)

        is_relu32 = c_out < RELU32_CHANNELS
        relu32_c = tl.where(is_relu32, c_out, 0)
        relu32_offsets = n * RELU32_CHANNELS * HW + relu32_c[:, None] * HW + hw[None, :]
        vals += tl.load(
            relu32_ptr + relu32_offsets,
            mask=is_relu32[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        is_relu33 = (c_out >= relu33_start) & (c_out < relu34_start)
        relu33_c = tl.where(is_relu33, c_out - relu33_start, 0)
        relu33_offsets = n * BN_CHANNELS * HW + relu33_c[:, None] * HW + hw[None, :]
        vals += tl.load(
            relu33_ptr + relu33_offsets,
            mask=is_relu33[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        is_relu34 = (c_out >= relu34_start) & (c_out < relu35_start)
        relu34_c = tl.where(is_relu34, c_out - relu34_start, 0)
        relu34_offsets = n * BN_CHANNELS * HW + relu34_c[:, None] * HW + hw[None, :]
        vals += tl.load(
            relu34_ptr + relu34_offsets,
            mask=is_relu34[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        is_relu35 = (c_out >= relu35_start) & (c_out < relu36_start)
        relu35_c = tl.where(is_relu35, c_out - relu35_start, 0)
        relu35_offsets = n * BN_CHANNELS * HW + relu35_c[:, None] * HW + hw[None, :]
        vals += tl.load(
            relu35_ptr + relu35_offsets,
            mask=is_relu35[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        is_relu36 = (c_out >= relu36_start) & (c_out < PREFIX_CHANNELS)
        relu36_c = tl.where(is_relu36, c_out - relu36_start, 0)
        relu36_offsets = n * BN_CHANNELS * HW + relu36_c[:, None] * HW + hw[None, :]
        vals += tl.load(
            relu36_ptr + relu36_offsets,
            mask=is_relu36[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        tl.store(out_ptr + out_offsets, vals, mask=out_mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_C": 8, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 64}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_C": 32, "BLOCK_HW": 64}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_C": 16, "BLOCK_HW": 128}, num_warps=4, num_stages=3),
        ],
        key=["BN_CHANNELS", "RELU32_CHANNELS", "OUT_CHANNELS", "HW"],
    )
    @triton.jit
    def _bn_relu_tail_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BN_CHANNELS: tl.constexpr,
        RELU32_CHANNELS: tl.constexpr,
        OUT_CHANNELS: tl.constexpr,
        HW: tl.constexpr,
        EPS: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n = tl.program_id(0)
        c_block = tl.program_id(1)
        hw_block = tl.program_id(2)

        c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
        mask = (c[:, None] < BN_CHANNELS) & (hw[None, :] < HW)

        conv_offsets = n * BN_CHANNELS * HW + c[:, None] * HW + hw[None, :]
        conv = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=c < BN_CHANNELS, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + c, mask=c < BN_CHANNELS, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c < BN_CHANNELS, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=c < BN_CHANNELS, other=0.0).to(tl.float32)

        affine = (conv - mean[:, None]) * tl.rsqrt(var[:, None] + EPS)
        affine = affine * weight[:, None] + bias[:, None]
        affine_fp16 = affine.to(tl.float16)
        relu = tl.where(
            affine_fp16 != affine_fp16,
            affine_fp16,
            tl.maximum(affine_fp16, 0.0),
        )

        out_c = RELU32_CHANNELS + 4 * BN_CHANNELS + c
        out_offsets = n * OUT_CHANNELS * HW + out_c[:, None] * HW + hw[None, :]
        tl.store(out_ptr + out_offsets, relu, mask=mask)


def _validate_inputs(inputs):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu_cat.py")
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects 10 inputs, got {len(inputs)}")

    mean, conv, var, weight, bias, relu32, relu33, relu34, relu35, relu36 = inputs
    tensors = (mean, conv, var, weight, bias, relu32, relu33, relu34, relu35, relu36)
    names = (
        "mean",
        "convolution_37",
        "var",
        "weight",
        "bias",
        "relu_32",
        "relu_33",
        "relu_34",
        "relu_35",
        "relu_36",
    )
    for name, tensor in zip(names, tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if not tensor.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor")
        if tensor.dtype != torch.float16:
            raise ValueError(f"{name} must have dtype torch.float16, got {tensor.dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if mean.ndim != 1:
        raise ValueError(f"mean must be rank 1, got shape {tuple(mean.shape)}")
    bn_channels = mean.shape[0]
    for name, tensor in (("var", var), ("weight", weight), ("bias", bias)):
        if tuple(tensor.shape) != (bn_channels,):
            raise ValueError(f"{name} must have shape [{bn_channels}], got {tuple(tensor.shape)}")

    if conv.ndim != 4:
        raise ValueError(f"convolution_37 must be rank 4, got shape {tuple(conv.shape)}")
    batch, conv_channels, height, width = conv.shape
    if conv_channels != bn_channels:
        raise ValueError(f"convolution_37 channel count {conv_channels} != {bn_channels}")
    for name, tensor in (("relu_33", relu33), ("relu_34", relu34), ("relu_35", relu35), ("relu_36", relu36)):
        if tuple(tensor.shape) != (batch, bn_channels, height, width):
            raise ValueError(
                f"{name} must have shape {(batch, bn_channels, height, width)}, got {tuple(tensor.shape)}"
            )
    if relu32.ndim != 4 or relu32.shape[0] != batch or relu32.shape[2:] != (height, width):
        raise ValueError(
            f"relu_32 must have shape [batch, channels, {height}, {width}], got {tuple(relu32.shape)}"
        )

    return (
        mean,
        conv,
        var,
        weight,
        bias,
        relu32,
        relu33,
        relu34,
        relu35,
        relu36,
        batch,
        bn_channels,
        relu32.shape[1],
        height,
        width,
    )


def oracle_forward(inputs):
    """Run the full Repro.forward scope: BN-inference affine/ReLU plus cat."""
    (
        mean,
        conv,
        var,
        weight,
        bias,
        relu32,
        relu33,
        relu34,
        relu35,
        relu36,
        batch,
        bn_channels,
        relu32_channels,
        height,
        width,
    ) = _validate_inputs(inputs)

    hw = height * width
    out_channels = relu32_channels + 5 * bn_channels
    prefix_channels = relu32_channels + 4 * bn_channels
    out = torch.empty_strided(
        (batch, out_channels, height, width),
        (out_channels * hw, hw, width, 1),
        device=conv.device,
        dtype=torch.float16,
    )
    prefix_grid = lambda meta: (
        batch,
        triton.cdiv(prefix_channels, meta["BLOCK_C"]),
        triton.cdiv(hw, meta["BLOCK_HW"]),
    )
    _copy_cat_prefix_kernel[prefix_grid](
        relu32,
        relu33,
        relu34,
        relu35,
        relu36,
        out,
        BN_CHANNELS=bn_channels,
        RELU32_CHANNELS=relu32_channels,
        PREFIX_CHANNELS=prefix_channels,
        OUT_CHANNELS=out_channels,
        HW=hw,
    )
    tail_grid = lambda meta: (
        batch,
        triton.cdiv(bn_channels, meta["BLOCK_C"]),
        triton.cdiv(hw, meta["BLOCK_HW"]),
    )
    _bn_relu_tail_kernel[tail_grid](
        mean,
        conv,
        var,
        weight,
        bias,
        out,
        BN_CHANNELS=bn_channels,
        RELU32_CHANNELS=relu32_channels,
        OUT_CHANNELS=out_channels,
        HW=hw,
        EPS=1.0e-5,
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
