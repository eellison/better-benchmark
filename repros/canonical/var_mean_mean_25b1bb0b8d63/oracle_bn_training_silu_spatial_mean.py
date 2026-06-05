"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet training-BatchNorm, running-stat copy_ side effects, affine SiLU, as_strided/reshape-compatible 7x7 spatial mean, and three returned outputs with Triton kernels that keep only per-channel stats between the normalization update and the coalesced spatial-pooling epilogue, whereas Inductor currently schedules the var_mean/update normalization region and downstream SiLU plus spatial mean consumer as generic producer/consumer regions with avoidable activation traffic; Inductor cannot do this today because its scheduler cannot fuse or pipeline a mutable training-BatchNorm reduction template with a following activation and small spatial reduction while preserving the running-stat copy_ return aliases; the fix is SCHEDULER_FUSION: teach the BN-training scheduler to expose mean/invstd/running-stat side outputs and sink affine SiLU plus fixed 7x7 spatial pooling into a channel-tiled lowering."""
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

CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True

N = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_HW = 64
FINAL_STATS_C = 64


def get_inputs() -> tuple[Any, ...]:
    """Load inputs from the repro's make_inputs."""
    return tuple(_harness_get_inputs(REPRO_DIR))


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 16, "RBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 32, "RBLOCK": 128}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 64, "RBLOCK": 128}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 128, "RBLOCK": 128}, num_warps=8, num_stages=3),
        ],
        key=["total_chw"],
    )
    @triton.jit
    def _batch_partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sum2_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        batch: tl.constexpr,
        total_chw: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        rindex = tl.arange(0, RBLOCK)[None, :]
        xmask = xindex < total_chw
        rmask = rindex < batch
        channel = xindex % channels
        hw = xindex // channels
        offsets = rindex * channels * hw_size + hw * channels + channel

        x = tl.load(x_ptr + offsets, mask=xmask & rmask, other=0.0).to(tl.float32)
        sum_x = tl.sum(x, axis=1)[:, None]
        sum_x2 = tl.sum(x * x, axis=1)[:, None]
        tl.store(partial_sum_ptr + xindex, sum_x, mask=xmask)
        tl.store(partial_sum2_ptr + xindex, sum_x2, mask=xmask)

    @triton.jit
    def _final_stats_update_kernel(
        partial_sum_ptr,
        partial_sum2_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        c_offsets = tl.program_id(0) * BLOCK_C_ + tl.arange(0, BLOCK_C_)[:, None]
        hw_offsets = tl.arange(0, BLOCK_HW_)[None, :]
        c_mask = c_offsets < channels
        hw_mask = hw_offsets < hw_size
        partial_offsets = hw_offsets * channels + c_offsets

        partial_sum = tl.load(partial_sum_ptr + partial_offsets, mask=c_mask & hw_mask, other=0.0).to(tl.float32)
        partial_sum2 = tl.load(partial_sum2_ptr + partial_offsets, mask=c_mask & hw_mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(partial_sum, axis=1)[:, None]
        sum_x2 = tl.sum(partial_sum2, axis=1)[:, None]
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(running_mean_ptr + c_offsets, old_running_mean * (1.0 - momentum) + mean * momentum, mask=c_mask)
        tl.store(
            running_var_ptr + c_offsets,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask,
        )
        tl.store(stats_ptr + c_offsets, mean, mask=c_mask)
        tl.store(stats_ptr + channels + c_offsets, invstd, mask=c_mask)

    @triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 16, "RBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 32, "RBLOCK": 64}, num_warps=4, num_stages=3),
            triton.Config({"XBLOCK": 64, "RBLOCK": 64}, num_warps=8, num_stages=3),
            triton.Config({"XBLOCK": 128, "RBLOCK": 64}, num_warps=8, num_stages=3),
        ],
        key=["total_out"],
    )
    @triton.jit
    def _flat_output_from_stats_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        total_out: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
    ):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
        hw_offsets = tl.arange(0, RBLOCK)[None, :]
        xmask = xindex < total_out
        hw_mask = hw_offsets < hw_size
        channel = xindex % channels
        n_idx = xindex // channels
        offsets = n_idx * channels * hw_size + hw_offsets * channels + channel

        x = tl.load(x_ptr + offsets, mask=xmask & hw_mask, other=0.0).to(tl.float32)
        mean = tl.load(stats_ptr + channel, mask=xmask, other=0.0).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel, mask=xmask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=xmask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=xmask, other=0.0).to(tl.float32)

        y = (x - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)
        pooled = tl.sum(tl.where(hw_mask, silu, 0.0), axis=1)[:, None] / hw_size
        tl.store(out_ptr + xindex, pooled, mask=xmask)


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

    convolution_80, primals_356, primals_357, primals_358, primals_359, shape_param = inputs
    x = _expect_f32_tensor(
        "convolution_80",
        convolution_80,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS),
    )
    running_mean = _expect_f32_tensor("primals_356", primals_356, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("primals_357", primals_357, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("primals_358", primals_358, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("primals_359", primals_359, (CHANNELS,), (1,))

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected reshape shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


def oracle_forward(inputs: list[Any] | tuple[Any, ...]):
    """Run the full Repro.forward scope, including running-stat mutations."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_training_silu_spatial_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sum = torch.empty_strided(
        (HW, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    partial_sum2 = torch.empty_strided(
        (HW, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _batch_partial_stats_kernel[lambda meta: (triton.cdiv(CHANNELS * HW, meta["XBLOCK"]),)](
        x,
        partial_sum,
        partial_sum2,
        channels=CHANNELS,
        hw_size=HW,
        batch=N,
        total_chw=CHANNELS * HW,
    )
    _final_stats_update_kernel[(triton.cdiv(CHANNELS, FINAL_STATS_C),)](
        partial_sum,
        partial_sum2,
        running_mean,
        running_var,
        stats,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_C_=FINAL_STATS_C,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    _flat_output_from_stats_kernel[lambda meta: (triton.cdiv(N * CHANNELS, meta["XBLOCK"]),)](
        x,
        weight,
        bias,
        stats,
        output,
        channels=CHANNELS,
        hw_size=HW,
        total_out=N * CHANNELS,
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

    all_pass = True
    alias_input_indices = (None, 1, 2)
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

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected_tensor.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual_tensor.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias oracle={actual_alias} "
                    f"eager={expected_alias}"
                )
                all_pass = False
                continue

        max_diff = (expected_tensor.float() - actual_tensor.float()).abs().max().item()
        ok = torch.allclose(expected_tensor.float(), actual_tensor.float(), atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"stride={expected_tensor.stride()} max_diff={max_diff:.2e})"
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")
            print(f"actionable: {'yes' if ACTIONABLE and result['status'] == 'GOOD' else 'no'}")


if __name__ == "__main__":
    main()
