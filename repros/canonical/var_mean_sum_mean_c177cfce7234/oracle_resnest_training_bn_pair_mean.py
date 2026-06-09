"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ResNeSt training-BatchNorm, in-place running-stat updates, affine ReLU, radix-2 channel-pair sum, and spatial mean by sinking the BN epilogue directly into the final pairwise reduction, whereas Inductor currently lowers the multi-output var_mean and mutable copy_ side effects separately from the downstream view/sum/mean consumer and cannot avoid generic intermediate normalized-activation work; Inductor cannot do this today because its scheduler does not fuse a training normalization reduction with aliasing running-stat outputs through a reshape-mediated sibling channel reduction; the fix is SCHEDULER_FUSION: extend normalization scheduling so BN stats, mutable running-stat returns, and downstream fixed channel-pair/spatial reductions can be emitted as one fused reduction plan."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


N = 32
CHANNELS = 1024
HALF_CHANNELS = 512
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
STAT_BLOCK_N = 32
STAT_BLOCK_HW = 256
PAIR_BLOCK_C = 4
PAIR_BLOCK_HW = 256

X_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
X_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUT_SHAPE = (N, HALF_CHANNELS, 1, 1)
OUT_STRIDE = (HALF_CHANNELS, 1, 1, 1)
EXPECTED_VIEW_SHAPE = [N, 2, HALF_CHANNELS, HEIGHT, WIDTH]
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True


if triton is not None:

    @triton.jit
    def _stats_update_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        stats_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N)
        hw_offsets = tl.arange(0, BLOCK_HW)
        hw_valid = hw_offsets < hw_size
        offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        vals = tl.load(x_ptr + offsets, mask=hw_valid[None, :], other=0.0).to(tl.float32)
        row_sum = tl.sum(vals, axis=1)
        row_sum_sq = tl.sum(vals * vals, axis=1)
        total_sum = tl.sum(row_sum, axis=0)
        total_sum_sq = tl.sum(row_sum_sq, axis=0)

        mean = total_sum / elements_per_channel
        var = total_sum_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(stats_ptr + channel, mean)
        tl.store(stats_ptr + channels + channel, invstd)

    @triton.jit
    def _bn_relu_pair_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        half_channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        batch = tl.program_id(0)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)
        c_valid = c_offsets < half_channels
        hw_valid = hw_offsets < hw_size
        valid = c_valid[:, None] & hw_valid[None, :]

        c1_offsets = c_offsets + half_channels
        base = batch * channels * hw_size
        x0_offsets = base + c_offsets[:, None] * hw_size + hw_offsets[None, :]
        x1_offsets = base + c1_offsets[:, None] * hw_size + hw_offsets[None, :]

        x0 = tl.load(x_ptr + x0_offsets, mask=valid, other=0.0).to(tl.float32)
        x1 = tl.load(x_ptr + x1_offsets, mask=valid, other=0.0).to(tl.float32)

        mean0 = tl.load(stats_ptr + c_offsets, mask=c_valid, other=0.0).to(tl.float32)
        mean1 = tl.load(stats_ptr + c1_offsets, mask=c_valid, other=0.0).to(tl.float32)
        inv0 = tl.load(stats_ptr + channels + c_offsets, mask=c_valid, other=0.0).to(tl.float32)
        inv1 = tl.load(stats_ptr + channels + c1_offsets, mask=c_valid, other=0.0).to(tl.float32)
        weight0 = tl.load(weight_ptr + c_offsets, mask=c_valid, other=0.0).to(tl.float32)
        weight1 = tl.load(weight_ptr + c1_offsets, mask=c_valid, other=0.0).to(tl.float32)
        bias0 = tl.load(bias_ptr + c_offsets, mask=c_valid, other=0.0).to(tl.float32)
        bias1 = tl.load(bias_ptr + c1_offsets, mask=c_valid, other=0.0).to(tl.float32)

        y0 = ((x0 - mean0[:, None]) * inv0[:, None]) * weight0[:, None] + bias0[:, None]
        y1 = ((x1 - mean1[:, None]) * inv1[:, None]) * weight1[:, None] + bias1[:, None]
        y0 = tl.where(y0 != y0, y0, tl.maximum(y0, 0.0))
        y1 = tl.where(y1 != y1, y1, tl.maximum(y1, 0.0))
        pair = y0 + y1
        reduced = tl.sum(tl.where(valid, pair, 0.0), axis=1) * (1.0 / 196.0)

        tl.store(out_ptr + batch * half_channels + c_offsets, reduced, mask=c_valid)


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


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_22, arg126_1, arg127_1, arg128_1, arg129_1, shape_param = inputs
    x = _expect_f32_tensor("convolution_22", convolution_22, X_SHAPE, X_STRIDE)
    running_mean = _expect_f32_tensor("arg126_1", arg126_1, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("arg127_1", arg127_1, VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("arg128_1", arg128_1, VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("arg129_1", arg129_1, VECTOR_SHAPE, VECTOR_STRIDE)

    if list(shape_param) != EXPECTED_VIEW_SHAPE:
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = torch.relu((x - mean) * invstd * weight[None, :, None, None] + bias[None, :, None, None])
    paired = y.view(N, 2, HALF_CHANNELS, HEIGHT, WIDTH).sum(dim=1)
    out = paired.mean(dim=(2, 3), keepdim=True)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return out, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([32, 1024, 14, 14], f32), T([1024], f32), T([1024], f32), T([1024], f32), T([1024], f32), S([32, 2, 512, 14, 14]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat side effects."""
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if not x.is_cuda:
        return _torch_reference(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for the CUDA oracle")

    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _stats_update_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        stats,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=STAT_BLOCK_N,
        BLOCK_HW=STAT_BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    _bn_relu_pair_spatial_mean_kernel[(N, triton.cdiv(HALF_CHANNELS, PAIR_BLOCK_C))](
        x,
        weight,
        bias,
        stats,
        out,
        channels=CHANNELS,
        half_channels=HALF_CHANNELS,
        hw_size=HW,
        BLOCK_C=PAIR_BLOCK_C,
        BLOCK_HW=PAIR_BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return out, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
        elif isinstance(item, list):
            cloned.append(list(item))
        else:
            cloned.append(item)
    return tuple(cloned)


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
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
            torch.cuda.synchronize()

    expected_list = _normalize_outputs(expected)
    actual_list = _normalize_outputs(actual)
    if len(actual_list) != len(expected_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(expected_list)}"
        )
        return False

    all_pass = True
    alias_input_indices = (None, 1, 2)
    for i, (expected_tensor, actual_tensor) in enumerate(zip(expected_list, actual_list)):
        metadata_ok = True
        if expected_tensor.shape != actual_tensor.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            metadata_ok = False
        if expected_tensor.dtype != actual_tensor.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
            metadata_ok = False
        if expected_tensor.stride() != actual_tensor.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual_tensor.stride()} "
                f"eager={expected_tensor.stride()}"
            )
            metadata_ok = False

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected_tensor.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual_tensor.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias "
                    f"oracle={actual_alias} eager={expected_alias}"
                )
                metadata_ok = False

        if not metadata_ok:
            all_pass = False
            continue

        if expected_tensor.is_floating_point():
            expected_f32 = expected_tensor.float()
            actual_f32 = actual_tensor.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            nan_ok = torch.equal(expected_nan, actual_nan)
            finite = ~expected_nan & ~actual_nan
            if finite.any():
                finite_expected = expected_f32[finite]
                finite_actual = actual_f32[finite]
                max_diff = (finite_expected - finite_actual).abs().max().item()
                values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
            else:
                max_diff = 0.0
                values_ok = True
            ok = nan_ok and values_ok
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
                f"stride={expected_tensor.stride()} max_finite_diff={max_diff:.2e} "
                f"nan_count={int(expected_nan.sum().item())})"
            )
            if not nan_ok:
                print(
                    f"  output {i}: NaN mask mismatch "
                    f"(eager={int(expected_nan.sum().item())}, oracle={int(actual_nan.sum().item())})"
                )
            all_pass = all_pass and ok
        else:
            ok = torch.equal(expected_tensor, actual_tensor)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(exact, shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype})"
            )
            all_pass = all_pass and ok

    alias_ok = (
        isinstance(actual, tuple)
        and len(actual) == 3
        and actual[1] is oracle_inputs[1]
        and actual[2] is oracle_inputs[2]
    )
    print(f"  running-stat aliases: {'PASS' if alias_ok else 'FAIL'}")
    return all_pass and alias_ok


# --- CLI entry point ---
def main() -> None:
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
        ok = _run_check(
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
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
