"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full LearningToPaint training-BatchNorm residual ReLU average-pool scope by reducing channel statistics, updating the running-stat copy_ outputs in place, and fusing affine residual ReLU directly into the returned spatial average without materializing the normalized activation, whereas Inductor currently schedules the var_mean/running-stat update, broadcast normalization, residual ReLU, and avg_pool2d/view as separate producer and consumer regions with full activation traffic; Inductor cannot do this today because its scheduler cannot fuse a normalization reduction template with mutable copy_ side effects into a downstream spatial reduction consumer; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat side outputs while sinking affine residual activation and spatial-pool consumers into the same fused schedule."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile importable without Triton.
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

BATCH = 1024
CHANNELS = 512
HEIGHT = 4
WIDTH = 4
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000610388817677

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
POOLED_SHAPE = (BATCH, CHANNELS)
POOLED_STRIDE = (CHANNELS, 1)
MEAN_VIEW_SHAPE = (1, CHANNELS, 1, 1)
MEAN_VIEW_STRIDE = (CHANNELS, 1, 1, 1)

STATS_BLOCK_R = 512
STATS_BLOCK_C = 16
POOL_BLOCK_C = 64


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sq_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        r_block = tl.program_id(0)
        c_block = tl.program_id(1) * BLOCK_C
        r = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
        c = c_block + tl.arange(0, BLOCK_C)
        r_mask = r < elements_per_channel
        c_mask = c < channels

        n = r // hw_size
        hw = r - n * hw_size
        offsets = (n[:, None] * channels + c[None, :]) * hw_size + hw[:, None]
        mask = r_mask[:, None] & c_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        partial_sum = tl.sum(x, axis=0)
        partial_sq = tl.sum(x * x, axis=0)
        out_offsets = r_block * channels + c
        tl.store(partial_sum_ptr + out_offsets, partial_sum, mask=c_mask)
        tl.store(partial_sq_ptr + out_offsets, partial_sq, mask=c_mask)

    @triton.jit
    def _finalize_stats_kernel(
        partial_sum_ptr,
        partial_sq_ptr,
        running_mean_ptr,
        running_var_ptr,
        invstd_out_ptr,
        mean_view_out_ptr,
        channels: tl.constexpr,
        n_blocks: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_block = tl.program_id(0) * BLOCK_C
        b = tl.arange(0, BLOCK_B)
        c = c_block + tl.arange(0, BLOCK_C)
        mask = (b[:, None] < n_blocks) & (c[None, :] < channels)
        offsets = b[:, None] * channels + c[None, :]

        sums = tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sqs = tl.load(partial_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        total = elements_per_channel + 0.0
        mean = tl.sum(sums, axis=0) / total
        mean_sq = tl.sum(sqs, axis=0) / total
        var = tl.maximum(mean_sq - mean * mean, 0.0)
        invstd = tl.rsqrt(var + eps)

        c_mask = c < channels
        old_mean = tl.load(running_mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        old_var = tl.load(running_var_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(invstd_out_ptr + c, invstd, mask=c_mask)
        tl.store(mean_view_out_ptr + c, mean, mask=c_mask)
        tl.store(running_mean_ptr + c, old_mean * (1.0 - momentum) + mean * momentum, mask=c_mask)
        tl.store(
            running_var_ptr + c,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask,
        )

    @triton.jit
    def _bn_residual_relu_pool_kernel(
        x_ptr,
        residual_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_HW: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_block = tl.program_id(1) * BLOCK_C
        hw = tl.arange(0, BLOCK_HW)
        c = c_block + tl.arange(0, BLOCK_C)
        hw_mask = hw < hw_size
        c_mask = c < channels

        offsets = (batch * channels + c[None, :]) * hw_size + hw[:, None]
        mask = hw_mask[:, None] & c_mask[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

        y = (x - mean[None, :]) * invstd[None, :]
        y = y * weight[None, :] + bias[None, :] + residual
        y = tl.maximum(y, 0.0)
        pooled = tl.sum(tl.where(mask, y, 0.0), axis=0) * 0.0625
        tl.store(out_ptr + batch * channels + c, pooled, mask=c_mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    convolution_20, arg123_1, arg124_1, arg125_1, arg126_1, relu_14, shape_param = inputs
    x = _expect_f32_tensor("convolution_20", convolution_20, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg123_1", arg123_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg124_1", arg124_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg125_1", arg125_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg126_1", arg126_1, STAT_SHAPE, STAT_STRIDE)
    residual = _expect_f32_tensor("relu_14", relu_14, INPUT_SHAPE, INPUT_STRIDE)

    if tuple(int(dim) for dim in shape_param) not in (POOLED_SHAPE, (BATCH, -1)):
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, residual)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias, residual


@oracle_impl(hardware="H100", shapes="(T([1024, 512, 4, 4], f32), T([512], f32), T([512], f32), T([512], f32), T([512], f32), T([1024, 512, 4, 4], f32), S([1024, -1]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_residual_relu_avgpool.py")

    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    n_stat_blocks = triton.cdiv(ELEMENTS_PER_CHANNEL, STATS_BLOCK_R)

    partial_sum = torch.empty((n_stat_blocks, CHANNELS), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty_like(partial_sum)
    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    pooled = torch.empty_strided(POOLED_SHAPE, POOLED_STRIDE, device=x.device, dtype=torch.float32)
    mean_view = torch.empty_strided(MEAN_VIEW_SHAPE, MEAN_VIEW_STRIDE, device=x.device, dtype=torch.float32)

    _partial_stats_kernel[(n_stat_blocks, triton.cdiv(CHANNELS, STATS_BLOCK_C))](
        x,
        partial_sum,
        partial_sq,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        BLOCK_R=STATS_BLOCK_R,
        BLOCK_C=STATS_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _finalize_stats_kernel[(triton.cdiv(CHANNELS, STATS_BLOCK_C),)](
        partial_sum,
        partial_sq,
        running_mean,
        running_var,
        invstd,
        mean_view,
        channels=CHANNELS,
        n_blocks=n_stat_blocks,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_B=triton.next_power_of_2(n_stat_blocks),
        BLOCK_C=STATS_BLOCK_C,
        num_warps=1,
        num_stages=3,
    )
    _bn_residual_relu_pool_kernel[(BATCH, triton.cdiv(CHANNELS, POOL_BLOCK_C))](
        x,
        residual,
        weight,
        bias,
        mean_view,
        invstd,
        pooled,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_HW=HW,
        BLOCK_C=POOL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )
    return invstd, pooled, mean_view, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> list[Any]:
    return [item.detach().clone() if isinstance(item, torch.Tensor) else item for item in inputs]


def _normalize_outputs(outputs: Any) -> list[torch.Tensor]:
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in outputs:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_fresh_inputs(
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

    all_ok = True
    for idx, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        shape_ok = expected_tensor.shape == actual_tensor.shape
        dtype_ok = expected_tensor.dtype == actual_tensor.dtype
        stride_ok = expected_tensor.stride() == actual_tensor.stride()
        if not shape_ok:
            print(
                f"  output {idx}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            all_ok = False
            continue
        if not dtype_ok:
            print(
                f"  output {idx}: WARNING dtype mismatch oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
        expected_f32 = expected_tensor.float()
        actual_f32 = actual_tensor.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        values_ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        ok = bool(values_ok and stride_ok)
        print(
            f"  output {idx}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"max_diff={max_diff:.2e} expected_stride={expected_tensor.stride()} "
            f"oracle_stride={actual_tensor.stride()})"
        )
        all_ok = all_ok and ok
    return all_ok


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
        help="Accepted for template CLI compatibility; this repro has no stochastic outputs.",
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
        ok = _check_oracle_fresh_inputs(instance, inputs, atol=args.atol, rtol=args.rtol)
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
