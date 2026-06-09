"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full ShuffleNet inference BN-affine, explicit fp16 ReLU, virtual cat, channel-shuffle clone layout, final view, and both returned split views by writing one shared contiguous fp16[512,464,7,7] backing tensor directly, whereas Inductor currently lowers the BN/ReLU producer and the cat/view/permute/clone/view/split layout indexing as generic scheduled work with avoidable intermediate layout traffic; Inductor cannot do this today because its scheduler does not keep a channel cat virtual across a fixed channel-shuffle clone and split-return boundary while fusing the pointwise BN epilogue into the same output layout; the fix is SCHEDULER_FUSION: teach scheduler/codegen to represent fixed channel-shuffle cats as a direct multi-source output layout and sink affine pointwise producers into the final split backing allocation."""
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
# Replace this section with your optimized Triton kernel(s).
#
# Recommended pattern: use @triton.autotune so the kernel auto-selects
# the best config for each shape encountered via --all-shapes.

if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_N": 1024}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_N": 2048}, num_warps=8, num_stages=3),
        ],
        key=["TOTAL", "CHANNELS", "HW", "CONV_STRIDE_N", "SPLIT_STRIDE_N"],
    )
    @triton.jit
    def oracle_kernel(
        mean_ptr,
        conv_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        split_input_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        CHANNELS: tl.constexpr,
        HW: tl.constexpr,
        CONV_STRIDE_N: tl.constexpr,
        SPLIT_STRIDE_N: tl.constexpr,
        OUT_STRIDE_N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < TOTAL
        safe_offsets = tl.where(mask, offsets, 0)

        spatial = safe_offsets % HW
        channel = (safe_offsets // HW) % CHANNELS
        batch = safe_offsets // (CHANNELS * HW)

        conv_offsets = batch * CONV_STRIDE_N + channel * HW + spatial
        split_offsets = batch * SPLIT_STRIDE_N + channel * HW + spatial

        x = tl.load(conv_ptr + conv_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * (1.0 / tl.sqrt(var + 1.0e-5))
        y = y * weight + bias
        y = y.to(tl.float16).to(tl.float32)
        y = tl.where(y < 0.0, 0.0, y)

        cat_side = tl.load(split_input_ptr + split_offsets, mask=mask, other=0.0)
        out_base = batch * OUT_STRIDE_N + channel * (2 * HW) + spatial
        tl.store(out_ptr + out_base, cat_side, mask=mask)
        tl.store(out_ptr + out_base + HW, y, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([232], f16), T([512, 232, 7, 7], f16), T([232], f16), T([232], f16), T([232], f16), T([512, 232, 7, 7], f16, stride=(22736, 49, 7, 1)), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))")
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
    if len(inputs) != 8:
        raise ValueError(f"expected 8 inputs, got {len(inputs)}")

    (
        arg257_1,
        convolution_51,
        arg258_1,
        arg259_1,
        arg260_1,
        getitem_24,
        _shape_param_0,
        _shape_param_1,
    ) = inputs

    if not convolution_51.is_cuda:
        y = (convolution_51 - arg257_1.float()[None, :, None, None])
        y = y * torch.reciprocal(torch.sqrt(arg258_1.float()[None, :, None, None] + 1.0e-5))
        y = y * arg259_1[None, :, None, None] + arg260_1[None, :, None, None]
        relu = torch.relu(y.to(torch.float16))
        cat = torch.cat([getitem_24, relu], 1)
        shuffled = cat.view(_shape_param_0).permute(0, 2, 1, 3, 4).clone().view(_shape_param_1)
        split_tensor = torch.split(shuffled, 232, 1)
        return (split_tensor[1], split_tensor[0])

    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    tensor_inputs = (arg257_1, convolution_51, arg258_1, arg259_1, arg260_1, getitem_24)
    if any(tensor.dtype != torch.float16 for tensor in tensor_inputs):
        raise ValueError("oracle_shufflenet_bn_channel_shuffle_split.py expects fp16 tensor inputs")
    if any(tensor.device != convolution_51.device for tensor in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same device")
    if convolution_51.ndim != 4 or getitem_24.ndim != 4:
        raise ValueError("oracle expects 4D NCHW activation tensors")

    n_batches, channels, height, width = convolution_51.shape
    if tuple(getitem_24.shape) != (n_batches, channels, height, width):
        raise ValueError("split input shape must match convolution_51 shape")
    if any(tensor.numel() != channels for tensor in (arg257_1, arg258_1, arg259_1, arg260_1)):
        raise ValueError("BN parameter lengths must match the channel dimension")
    if list(_shape_param_0) != [n_batches, 2, channels, height, width]:
        raise ValueError("first shape parameter does not match channel-shuffle view")
    if list(_shape_param_1) != [n_batches, 2 * channels, height, width]:
        raise ValueError("second shape parameter does not match final view")

    hw = height * width
    out = torch.empty(
        (n_batches, 2 * channels, height, width),
        device=convolution_51.device,
        dtype=torch.float16,
    )
    total = n_batches * channels * hw

    grid = lambda meta: (triton.cdiv(total, meta["BLOCK_N"]),)
    oracle_kernel[grid](
        arg257_1,
        convolution_51,
        arg258_1,
        arg259_1,
        arg260_1,
        getitem_24,
        out,
        TOTAL=total,
        CHANNELS=channels,
        HW=hw,
        CONV_STRIDE_N=convolution_51.stride(0),
        SPLIT_STRIDE_N=getitem_24.stride(0),
        OUT_STRIDE_N=out.stride(0),
    )

    return (out[:, channels:, :, :], out[:, :channels, :, :])


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
