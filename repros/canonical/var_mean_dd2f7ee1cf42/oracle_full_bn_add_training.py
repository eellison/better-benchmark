"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Visformer training-BatchNorm scope from the two activation inputs, including the fused add producer, population per-channel var_mean over N/H/W, invstd output, affine normalized output, centered output, and both in-place running-stat copy_ returns in one channel-tiled Triton schedule, whereas Inductor currently lowers this shape through the generic norm-template schedule that does not pick the tighter full-scope add-plus-dual-output epilogue plan; Inductor cannot do this today because the training-normalization scheduler does not preserve cheap elementwise producers and multiple materialized post-stat consumers while selecting a compact per-channel reduction configuration for this fixed C=768, N*H*W=6272 case; the fix is SCHEDULER_FUSION: extend the BN-training norm template to fuse structured add producers and sibling affine/centered epilogues with mutable running-stat side outputs under the same channel reduction schedule."""
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


N = 128
CHANNELS = 768
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_M = 8192
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    @triton.jit
    def _bn_add_train_full_kernel(
        lhs_ptr,
        rhs_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        invstd_out_ptr,
        affine_out_ptr,
        centered_out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        lhs = tl.load(lhs_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        rhs = tl.load(rhs_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        x = lhs + rhs
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
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
        tl.store(invstd_out_ptr + channel, invstd)

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        centered = x - mean
        affine = centered * invstd
        affine = affine * weight + bias
        tl.store(centered_out_ptr + x_offsets, centered, mask=mask)
        tl.store(affine_out_ptr + x_offsets, affine, mask=mask)


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
    if value.dtype != torch.float32:
        raise TypeError(f"{name} has dtype {value.dtype}, expected torch.float32")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    add_169, convolution_54, arg193_1, arg194_1, arg195_1, arg196_1 = inputs
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    lhs = _expect_f32_tensor("add_169", add_169, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)
    rhs = _expect_f32_tensor("convolution_54", convolution_54, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)
    running_mean = _expect_f32_tensor("arg193_1", arg193_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg194_1", arg194_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg195_1", arg195_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg196_1", arg196_1, (CHANNELS,), (1,))

    device = lhs.device
    if any(t.device != device for t in (rhs, running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return lhs, rhs, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    lhs, rhs, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    x = lhs + rhs
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd_4d = torch.rsqrt(var + EPS)
    centered = x - mean
    affine = centered * invstd_4d
    affine = affine * weight[None, :, None, None] + bias[None, :, None, None]
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    invstd_1d = invstd_4d.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return invstd_1d, affine, centered, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([128, 768, 7, 7], f32), T([128, 768, 7, 7], f32), T([768], f32), T([768], f32), T([768], f32), T([768], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat mutations."""
    lhs, rhs, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not lhs.is_cuda:
        return _torch_reference(inputs)

    invstd = torch.empty_strided((CHANNELS,), (1,), device=lhs.device, dtype=torch.float32)
    affine = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=lhs.device,
        dtype=torch.float32,
    )
    centered = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=lhs.device,
        dtype=torch.float32,
    )

    _bn_add_train_full_kernel[(CHANNELS,)](
        lhs,
        rhs,
        running_mean,
        running_var,
        weight,
        bias,
        invstd,
        affine,
        centered,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_M,
        num_warps=4,
        num_stages=2,
    )
    return invstd, affine, centered, running_mean, running_var


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
