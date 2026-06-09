"""Gap diagnosis (classification: BANDWIDTH_BOUND): this artifact computes the full MobileNetV2 BN-inference affine over fp16 [128,1280,7,7], clamps to ReLU6 bounds in fp32, rounds to fp16, spatially averages each 7x7 tile, and returns the final contiguous fp16 [128,1280] view by folding BN into per-channel scale/shift values reused across a batch tile, whereas tuned Inductor's generated full-scope reduction is already faster than this oracle for the same output contract; Inductor cannot be assigned a local missing optimization from this artifact because the measured oracle is not a floor and the compiled kernel already wins despite recomputing the small per-channel affine expression inside the fused reduction; the fix is BANDWIDTH_BOUND: treat this as a non-floor diagnostic artifact and leave the row closed only as not_true_floor unless a faster full-scope oracle proves a real gap."""
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


CHANNELS = 1280
BATCH = 128
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
BLOCK_BATCH = 16
BLOCK_CHANNELS = 4
BLOCK_HW = 64

if triton is not None:

    @triton.jit
    def _bn_relu6_spatial_mean_kernel(
        mean_ptr,
        x_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        BLOCK_BATCH_: tl.constexpr,
        BLOCK_CHANNELS_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        eps: tl.constexpr,
    ):
        batch_offsets = tl.program_id(0) * BLOCK_BATCH_ + tl.arange(0, BLOCK_BATCH_)
        channel_offsets = tl.program_id(1) * BLOCK_CHANNELS_ + tl.arange(0, BLOCK_CHANNELS_)
        hw_offsets = tl.arange(0, BLOCK_HW_)

        valid_channels = channel_offsets < 1280
        valid_batch_channel = (batch_offsets[:, None] < 128) & valid_channels[None, :]
        valid_hw = hw_offsets < 49

        mean = tl.load(mean_ptr + channel_offsets, mask=valid_channels, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel_offsets, mask=valid_channels, other=1.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel_offsets, mask=valid_channels, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel_offsets, mask=valid_channels, other=0.0).to(tl.float32)
        scale = (1.0 / tl.sqrt(var + eps)) * weight
        shift = bias - mean * scale

        x_offsets = (
            (batch_offsets[:, None, None] * 1280 + channel_offsets[None, :, None]) * 49
            + hw_offsets[None, None, :]
        )
        valid = valid_batch_channel[:, :, None] & valid_hw[None, None, :]
        x = tl.load(x_ptr + x_offsets, mask=valid, other=0.0).to(tl.float32)

        affine = x * scale[None, :, None] + shift[None, :, None]
        relu6 = tl.where(affine < 0.0, 0.0, affine)
        relu6 = tl.where(relu6 > 6.0, 6.0, relu6)
        relu6_h = relu6.to(tl.float16)
        reduced = tl.sum(tl.where(valid, relu6_h.to(tl.float32), 0.0), axis=2) * (1.0 / 49.0)

        out_offsets = batch_offsets[:, None] * 1280 + channel_offsets[None, :]
        tl.store(out_ptr + out_offsets, reduced, mask=valid_batch_channel)


def _require_f16_tensor(
    name,
    value,
    shape,
    stride,
):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")
    if value.dtype != torch.float16:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float16")
    if not value.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs):
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    mean, x, var, weight, bias, view_shape = inputs
    mean_t = _require_f16_tensor("arg257_1", mean, (CHANNELS,), (1,))
    x_t = _require_f16_tensor(
        "convolution_51",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    var_t = _require_f16_tensor("arg258_1", var, (CHANNELS,), (1,))
    weight_t = _require_f16_tensor("arg259_1", weight, (CHANNELS,), (1,))
    bias_t = _require_f16_tensor("arg260_1", bias, (CHANNELS,), (1,))

    if list(view_shape) != [BATCH, CHANNELS]:
        raise ValueError(f"unexpected output view shape parameter: {view_shape!r}")
    if any(t.device != x_t.device for t in (mean_t, var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return mean_t, x_t, var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([1280], f16), T([128, 1280, 7, 7], f16), T([1280], f16), T([1280], f16), T([1280], f16), S([128, 1280]))")
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
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_relu6_spatial_mean.py")

    mean, x, var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (BATCH, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float16,
    )
    _bn_relu6_spatial_mean_kernel[(triton.cdiv(BATCH, BLOCK_BATCH), triton.cdiv(CHANNELS, BLOCK_CHANNELS))](
        mean,
        x,
        var,
        weight,
        bias,
        output,
        BLOCK_BATCH_=BLOCK_BATCH,
        BLOCK_CHANNELS_=BLOCK_CHANNELS,
        BLOCK_HW_=BLOCK_HW,
        eps=EPS,
        num_warps=1,
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
