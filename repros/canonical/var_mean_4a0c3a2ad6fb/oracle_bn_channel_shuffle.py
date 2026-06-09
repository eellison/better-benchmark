"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet training-BatchNorm plus skip-branch channel-shuffle scope, including the channelwise population var_mean, running mean/variance copy_ side effects, affine ReLU epilogue, cat/view/permute/clone/view layout transform, and full shuffled output tensor by writing the skip branch and BN branch directly into the final contiguous layout, whereas Inductor lowers the decomposed BN, mutable running-stat updates, concat, permutation clone, and final view as generic reduction and layout work; Inductor cannot do this today because its scheduler does not preserve a fixed channel-shuffle consumer across a training-normalization producer with side-effect outputs and an already-strided sibling input; the fix is SCHEDULER_FUSION: teach the BN-training/layout scheduler to fuse running-stat epilogues with direct channel-shuffle stores into the final output layout."""
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


N = 512
C = 232
H = 7
W = 7
HW = H * W
K = N * HW
OUT_C = C * 2
EPS = 1.0e-5
MOMENTUM = 0.1
OLD_WEIGHT = 0.9
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK = 32768
POINTWISE_BLOCK = 256

if triton is not None:

    @triton.jit
    def _bn_stats_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        k_size: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        old_weight: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < k_size
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size

        x_offsets = n_idx * channels * hw_size + channel * hw_size + hw_idx
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) / k_size
        centered = x - mean
        var = tl.sum(centered * centered, axis=0) / k_size
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * old_weight + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * old_weight + var * running_var_correction * momentum,
        )

    @triton.jit
    def _shuffle_output_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        skip_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        hw_idx = offsets % hw_size
        channel = (offsets // hw_size) % channels
        n_idx = offsets // (channels * hw_size)

        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd * weight + bias
        y = tl.where(y != y, y, tl.maximum(y, 0.0))

        skip_offsets = n_idx * out_channels * hw_size + channel * hw_size + hw_idx
        skip = tl.load(skip_ptr + skip_offsets, mask=mask, other=0.0).to(tl.float32)

        out_base = n_idx * out_channels * hw_size + channel * 2 * hw_size + hw_idx
        tl.store(out_ptr + out_base, skip, mask=mask)
        tl.store(out_ptr + out_base + hw_size, y, mask=mask)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(dim) for dim in value)
    except TypeError as exc:
        raise TypeError(f"expected shape parameter, got {value!r}") from exc


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
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 8:
        raise ValueError(f"{REPRO_ID} expects 8 inputs, got {len(inputs)}")

    (
        convolution_54,
        arg327_1,
        arg328_1,
        arg329_1,
        arg330_1,
        getitem_130,
        shape0,
        shape1,
    ) = inputs

    activation_stride = (C * HW, HW, W, 1)
    shuffled_half_stride = (OUT_C * HW, HW, W, 1)
    x = _expect_f32_tensor("convolution_54", convolution_54, (N, C, H, W), activation_stride)
    running_mean = _expect_f32_tensor("arg327_1", arg327_1, (C,), (1,))
    running_var = _expect_f32_tensor("arg328_1", arg328_1, (C,), (1,))
    weight = _expect_f32_tensor("arg329_1", arg329_1, (C,), (1,))
    bias = _expect_f32_tensor("arg330_1", arg330_1, (C,), (1,))
    skip = _expect_f32_tensor("getitem_130", getitem_130, (N, C, H, W), shuffled_half_stride)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, skip)):
        raise ValueError("all tensor inputs must be on the same device")

    if _shape_tuple(shape0) != (N, 2, C, H, W):
        raise ValueError(f"unexpected first view shape parameter: {shape0!r}")
    if _shape_tuple(shape1) != (N, OUT_C, H, W):
        raise ValueError(f"unexpected second view shape parameter: {shape1!r}")

    return x, running_mean, running_var, weight, bias, skip


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    (
        x,
        running_mean,
        running_var,
        weight,
        bias,
        skip,
    ) = _validate_inputs(inputs)
    shape0 = _shape_tuple(inputs[6])
    shape1 = _shape_tuple(inputs[7])

    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = torch.relu((x - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    shuffled = (
        torch.cat([skip, y], dim=1)
        .view(shape0)
        .permute(0, 2, 1, 3, 4)
        .clone(memory_format=torch.contiguous_format)
        .view(shape1)
    )
    running_mean.copy_(running_mean * OLD_WEIGHT + mean_1d * MOMENTUM)
    running_var.copy_(running_var * OLD_WEIGHT + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return shuffled, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([512, 232, 7, 7], f32), T([232], f32), T([232], f32), T([232], f32), T([232], f32), T([512, 232, 7, 7], f32, stride=(22736, 49, 7, 1)), S([512, 2, 232, 7, 7]), S([512, 464, 7, 7]))")
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
    x, running_mean, running_var, weight, bias, skip = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    shuffled = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=x.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((C,), (1,), device=x.device, dtype=torch.float32)

    _bn_stats_kernel[(C,)](
        x,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=C,
        hw_size=HW,
        k_size=K,
        eps=EPS,
        momentum=MOMENTUM,
        old_weight=OLD_WEIGHT,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=STAT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _shuffle_output_kernel[(triton.cdiv(N * C * HW, POINTWISE_BLOCK),)](
        x,
        weight,
        bias,
        skip,
        mean,
        invstd,
        shuffled,
        total=N * C * HW,
        channels=C,
        out_channels=OUT_C,
        hw_size=HW,
        BLOCK=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return shuffled, running_mean, running_var


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
