"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileViT training-BatchNorm over channels-last f32[128,640,8,8], both running-stat copy_ side effects, affine SiLU, as_strided/reshape-compatible spatial mean, and the three returned outputs without materializing the normalized activation, whereas Inductor currently schedules the BN var_mean/update region and the downstream SiLU plus spatial mean consumer through generic reduction/pointwise regions with avoidable intermediate activation traffic; Inductor cannot do this today because its scheduler cannot fuse a training-BatchNorm reduction template with a following activation and small spatial reduction while also preserving mutable running-stat side outputs on a channels-last logical-NCHW input; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to expose mean/invstd/running-stat results and sink affine SiLU plus fixed 8x8 spatial mean into the channel-tiled reduction schedule."""
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

from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True

N = 128
CHANNELS = 640
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804
STATS_BLOCK = 8192
FUSED_BLOCK_N = 128
FUSED_BLOCK_HW = 64
POOL_BLOCK_N = 16
POOL_BLOCK_HW = 64


def get_inputs() -> tuple[Any, ...]:
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _fused_bn_silu_spatial_mean_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N)
        hw_offsets = tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < hw_size
        x_offsets = n_offsets[:, None] * channels * hw_size + hw_offsets[None, :] * channels + channel

        x = tl.load(x_ptr + x_offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum2 = tl.sum(x * x, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x2 = tl.sum(row_sum2, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + 1.0e-5)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        silu = tl.where(hw_mask[None, :], silu, 0.0)
        pooled = tl.sum(silu, axis=1) / hw_size
        tl.store(out_ptr + n_offsets * channels + channel, pooled)

    @triton.jit
    def _stats_update_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = n_idx * channels * hw_size + hw_idx * channels + channel

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + 1.0e-5)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_running_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(stats_ptr + channel, mean)
        tl.store(stats_ptr + channels + channel, invstd)

    @triton.jit
    def _bn_silu_spatial_mean_from_stats_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        batch: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        hw_offsets = tl.arange(0, BLOCK_HW)
        n_mask = n_offsets < batch
        hw_mask = hw_offsets < hw_size
        x_offsets = n_offsets[:, None] * channels * hw_size + hw_offsets[None, :] * channels + channel

        x = tl.load(x_ptr + x_offsets, mask=n_mask[:, None] & hw_mask[None, :], other=0.0).to(tl.float32)
        mean = tl.load(stats_ptr + channel).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)

        y = (x - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        silu = tl.where(hw_mask[None, :], silu, 0.0)
        pooled = tl.sum(silu, axis=1) / hw_size
        tl.store(out_ptr + n_offsets * channels + channel, pooled, mask=n_mask)


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
    if not value.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor for this Triton oracle")
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    convolution_34, primals_307, primals_308, primals_309, primals_310, shape_param = inputs
    x = _expect_f32_tensor(
        "convolution_34",
        convolution_34,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS),
    )
    running_mean = _expect_f32_tensor("primals_307", primals_307, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("primals_308", primals_308, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("primals_309", primals_309, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("primals_310", primals_310, (CHANNELS,), (1,))

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected reshape shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 640, 8, 8], f32, stride=(40960, 1, 5120, 640)), T([640], f32), T([640], f32), T([640], f32), T([640], f32), S([128, 640]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_training_silu_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _fused_bn_silu_spatial_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=FUSED_BLOCK_N,
        BLOCK_HW=FUSED_BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return output, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(item.clone() if isinstance(item, torch.Tensor) else item for item in inputs)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _run_check(
    instance: torch.nn.Module,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
) -> bool:
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)

    with torch.no_grad():
        expected = instance(*eager_inputs)
        actual = oracle_forward(oracle_inputs)
        torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(expected_list) != len(actual_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    for i, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        shape_ok = expected_tensor.shape == actual_tensor.shape
        dtype_ok = expected_tensor.dtype == actual_tensor.dtype
        stride_ok = expected_tensor.stride() == actual_tensor.stride()
        if not shape_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            all_pass = False
            continue
        if not dtype_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
            all_pass = False
            continue
        if not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual_tensor.stride()} "
                f"eager={expected_tensor.stride()}"
            )
            all_pass = False
            continue

        max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
        ok = torch.allclose(expected_tensor.float(), actual_tensor.float(), atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(ok)

    return all_pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Accepted for template compatibility")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
    args = parser.parse_args()

    if args.show_hw:
        import json

        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    if has_stochastic_ops(REPRO_PATH):
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(
                        f"WARNING: oracle is slower than compile for "
                        f"{result['repro_id']} (ratio={result['ratio']:.3f}x)"
                    )
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")
            true_floor = result["status"] == "GOOD"
            print(f"classification: {CLASSIFICATION}")
            print(f"true_floor: {'yes' if true_floor else 'no'} ({result['status']})")
            print(f"actionable: {'yes' if ACTIONABLE and true_floor else 'no'}")


if __name__ == "__main__":
    main()
