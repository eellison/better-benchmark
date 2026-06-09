"""Gap diagnosis: this oracle computes the full virtual channel-cat, BN-inference affine in fp32, fp16 cast, and ReLU in one Triton pass while reading directly from the four source tensors, whereas Inductor materializes the concatenated activation before the downstream pointwise BN/ReLU schedule; Inductor cannot do this today because its scheduler does not model fixed channel concatenation as a virtual multi-source layout that can be inlined into a fusible pointwise consumer; the fix is SCHEDULER_FUSION: teach cat-to-pointwise fusion to lower concat reads as channel-range source selection instead of forcing a dense cat intermediate."""
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
# Do not add oracle-local sys.path or REPO_ROOT import hacks.
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


EPS = 1.0e-5
BLOCK_SIZE = 1024


def _check_tensor(
    name: str,
    value: object,
    *,
    ndim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape={tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype={dtype}, got {value.dtype}")
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got stride={tuple(value.stride())}")
    return value


def _validate_inputs(
    inputs: list[object] | tuple[object, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_cat_bn_relu.py")
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    x0 = _check_tensor("avg_pool2d_2", inputs[0], ndim=4, dtype=torch.float16)
    x1 = _check_tensor("convolution_89", inputs[1], ndim=4, dtype=torch.float16)
    x2 = _check_tensor("convolution_91", inputs[2], ndim=4, dtype=torch.float16)
    x3 = _check_tensor("convolution_93", inputs[3], ndim=4, dtype=torch.float16)
    mean = _check_tensor("arg471_1", inputs[4], ndim=1, dtype=torch.float16)
    var = _check_tensor("arg472_1", inputs[5], ndim=1, dtype=torch.float16)
    weight = _check_tensor("arg473_1", inputs[6], ndim=1, dtype=torch.float16)
    bias = _check_tensor("arg474_1", inputs[7], ndim=1, dtype=torch.float16)

    batch, c0, height, width = x0.shape
    expected_tail = (batch, 32, height, width)
    for name, tensor in (
        ("convolution_89", x1),
        ("convolution_91", x2),
        ("convolution_93", x3),
    ):
        if tuple(tensor.shape) != expected_tail:
            raise ValueError(f"{name} has shape={tuple(tensor.shape)}, expected={expected_tail}")

    channels = c0 + 3 * x1.shape[1]
    expected_stat_shape = (channels,)
    for name, tensor in (
        ("arg471_1", mean),
        ("arg472_1", var),
        ("arg473_1", weight),
        ("arg474_1", bias),
    ):
        if tuple(tensor.shape) != expected_stat_shape:
            raise ValueError(f"{name} has shape={tuple(tensor.shape)}, expected={expected_stat_shape}")

    device = x0.device
    if any(t.device != device for t in (x1, x2, x3, mean, var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x0, x1, x2, x3, mean, var, weight, bias

if triton is not None:

    @triton.jit
    def _cat_bn_relu_kernel(
        x0_ptr,
        x1_ptr,
        x2_ptr,
        x3_ptr,
        mean_ptr,
        var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        TOTAL: tl.constexpr,
        C0: tl.constexpr,
        TAIL_C: tl.constexpr,
        CHANNELS: tl.constexpr,
        HW: tl.constexpr,
        EPSILON: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < TOTAL

        channel_hw = CHANNELS * HW
        batch = offsets // channel_hw
        rem = offsets - batch * channel_hw
        channel = rem // HW
        spatial = rem - channel * HW

        is_x0 = channel < C0
        is_x1 = (channel >= C0) & (channel < C0 + TAIL_C)
        is_x2 = (channel >= C0 + TAIL_C) & (channel < C0 + 2 * TAIL_C)
        is_x3 = channel >= C0 + 2 * TAIL_C

        x0_offsets = batch * C0 * HW + channel * HW + spatial
        x1_channel = channel - C0
        x2_channel = channel - C0 - TAIL_C
        x3_channel = channel - C0 - 2 * TAIL_C
        tail_plane = TAIL_C * HW
        x1_offsets = batch * tail_plane + x1_channel * HW + spatial
        x2_offsets = batch * tail_plane + x2_channel * HW + spatial
        x3_offsets = batch * tail_plane + x3_channel * HW + spatial

        x0 = tl.load(x0_ptr + x0_offsets, mask=mask & is_x0, other=0.0)
        x1 = tl.load(x1_ptr + x1_offsets, mask=mask & is_x1, other=0.0)
        x2 = tl.load(x2_ptr + x2_offsets, mask=mask & is_x2, other=0.0)
        x3 = tl.load(x3_ptr + x3_offsets, mask=mask & is_x3, other=0.0)
        x = (x0 + x1 + x2 + x3).to(tl.float32)

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * (1.0 / tl.sqrt(var + EPSILON))
        y = y * weight + bias
        y_f16 = y.to(tl.float16)
        # ReLU after the fp16 cast, preserving NaNs like aten.relu.
        relu = tl.where(y_f16 < 0.0, 0.0, y_f16)
        tl.store(out_ptr + offsets, relu, mask=mask)


@oracle_impl(hardware="H100", shapes="(T([64, 512, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([64, 32, 7, 7], f16), T([608], f16), T([608], f16), T([608], f16), T([608], f16))")
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
    x0, x1, x2, x3, mean, var, weight, bias = _validate_inputs(inputs)
    batch, c0, height, width = x0.shape
    tail_c = x1.shape[1]
    channels = c0 + 3 * tail_c
    hw = height * width
    total = batch * channels * hw

    output = torch.empty(
        (batch, channels, height, width),
        device=x0.device,
        dtype=torch.float16,
    )
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    _cat_bn_relu_kernel[grid](
        x0,
        x1,
        x2,
        x3,
        mean,
        var,
        weight,
        bias,
        output,
        TOTAL=total,
        C0=c0,
        TAIL_C=tail_c,
        CHANNELS=channels,
        HW=hw,
        EPSILON=EPS,
        BLOCK=BLOCK_SIZE,
        num_warps=4,
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
            # All timing must go through bench_oracle(). Direct do_bench or
            # compiled(*inputs) timing includes dispatch overhead and can invent
            # fake gaps for fast kernels.
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
