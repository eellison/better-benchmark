"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete EfficientNet training-BatchNorm scope for f32 `[128,1280,7,7]`, including per-channel `var_mean(correction=0)`, in-place running mean/variance `copy_` side effects, affine SiLU, and the returned spatial mean, in a channel-specialized Triton kernel without materializing the normalized activation, whereas tuned Inductor already lowers the same full scope into a faster practical memory-traffic and launch envelope for this fixed shape; Inductor cannot materially improve this local region through a scheduler-fusion rewrite because the remaining work is dominated by required activation reads for channel statistics, affine/exp math, spatial mean stores, and running-stat writes rather than an uncovered surrounding-op fusion gap; the fix is BANDWIDTH_BOUND: do not create a gap-specific work item from this oracle because compile is already faster than the full-scope Triton candidate."""
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

N = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_N = 128
BLOCK_HW = 64


def get_inputs() -> tuple[Any, ...]:
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_silu_spatial_mean_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        hw_mask = hw_offsets < hw_size
        offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        x_for_stats = tl.where(hw_mask[None, :], x, 0.0)
        row_sum = tl.sum(x_for_stats, axis=1)
        row_sum2 = tl.sum(x_for_stats * x_for_stats, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x2 = tl.sum(row_sum2, axis=0)
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

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        silu = tl.where(hw_mask[None, :], silu, 0.0)
        pooled = tl.sum(silu, axis=1) / hw_size

        tl.store(out_ptr + n_offsets * channels + channel, pooled)


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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_training_silu_spatial_mean.py")
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    x, running_mean, running_var, weight, bias, shape_param = inputs
    x = _expect_f32_tensor(
        "convolution_80",
        x,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg355_1", running_mean, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg356_1", running_var, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg357_1", weight, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg358_1", bias, (CHANNELS,), (1,))

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")
    if any(t.device != x.device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")

    return x, running_mean, running_var, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _bn_silu_spatial_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N_=BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    return output, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(item.clone() if isinstance(item, torch.Tensor) else item for item in inputs)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
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

    alias_input_indices = (None, 1, 2)
    all_pass = True
    for i, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        metadata_ok = (
            expected_tensor.shape == actual_tensor.shape
            and expected_tensor.dtype == actual_tensor.dtype
            and expected_tensor.stride() == actual_tensor.stride()
            and expected_tensor.device == actual_tensor.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(actual_tensor.shape)} stride={actual_tensor.stride()} dtype={actual_tensor.dtype}) "
                f"eager=(shape={list(expected_tensor.shape)} stride={expected_tensor.stride()} dtype={expected_tensor.dtype})"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected_tensor.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual_tensor.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias "
                    f"oracle={actual_alias} eager={expected_alias}"
                )
                all_pass = False
                continue

        if expected_tensor.is_floating_point():
            max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
            ok = torch.allclose(expected_tensor.float(), actual_tensor.float(), atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(expected_tensor, actual_tensor)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(exact, shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype})"
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
    parser.add_argument(
        "--no-skip-stochastic",
        action="store_true",
        help="Disable auto-detection and skipping of stochastic outputs",
    )
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


if __name__ == "__main__":
    main()
