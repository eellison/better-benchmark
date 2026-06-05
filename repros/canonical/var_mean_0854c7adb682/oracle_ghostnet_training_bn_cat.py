"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full GhostNet training-BatchNorm scope, including per-channel var_mean over [512,7,7], running mean/variance copy_ side effects, affine ReLU, and the channel cat with relu_35 into the exact [512,960,7,7] layout, whereas Inductor currently schedules the BN statistics, running-stat updates, normalization epilogue, and cat copy through separate generic reduction, pointwise, and cat kernels; Inductor cannot do this today because its scheduler/codegen lacks a norm-template plan that fuses training-BN reductions and mutable running-stat returns with a following layout-sensitive channel concatenation; the fix is SCHEDULER_FUSION: add a guarded training-BatchNorm plus channel-cat fusion schedule that emits the final cat layout directly."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile useful without Triton.
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


N = 512
CHANNELS = 480
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (N, CHANNELS * 2, HEIGHT, WIDTH)
OUTPUT_STRIDE = (CHANNELS * 2 * HW, HW, WIDTH, 1)

REDUCTION_X_BLOCK = 1
REDUCTION_R_BLOCK = 2048
COPY_BLOCK = 1024


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_update_relu_second_half_kernel(
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
        X_BLOCK: tl.constexpr,
        R_BLOCK: tl.constexpr,
    ):
        channels_offsets = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)[:, None]
        r_offsets = tl.arange(0, R_BLOCK)[None, :]
        c_mask = channels_offsets < channels

        sum_x = tl.zeros((X_BLOCK,), tl.float32)
        sum_x_sq = tl.zeros((X_BLOCK,), tl.float32)
        for r_start in tl.range(0, elements_per_channel, R_BLOCK):
            r_index = r_start + r_offsets
            r_mask = r_index < elements_per_channel
            hw = r_index % hw_size
            n = r_index // hw_size
            offsets = n * channels * hw_size + channels_offsets * hw_size + hw
            x = tl.load(x_ptr + offsets, mask=r_mask & c_mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=1)
            sum_x_sq += tl.sum(x * x, axis=1)

        scale = 1.0 / elements_per_channel
        mean = sum_x * scale
        var = sum_x_sq * scale - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        channels_1d = tl.program_id(0) * X_BLOCK + tl.arange(0, X_BLOCK)
        c_mask_1d = channels_1d < channels
        old_running_mean = tl.load(running_mean_ptr + channels_1d, mask=c_mask_1d, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channels_1d, mask=c_mask_1d, other=0.0).to(tl.float32)
        tl.store(
            running_mean_ptr + channels_1d,
            old_running_mean * (1.0 - momentum) + mean * momentum,
            mask=c_mask_1d,
        )
        tl.store(
            running_var_ptr + channels_1d,
            old_running_var * (1.0 - momentum) + var * running_var_correction * momentum,
            mask=c_mask_1d,
        )

        weight = tl.load(weight_ptr + channels_1d, mask=c_mask_1d, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels_1d, mask=c_mask_1d, other=0.0).to(tl.float32)
        mean_b = mean[:, None]
        invstd_b = invstd[:, None]
        weight_b = weight[:, None]
        bias_b = bias[:, None]
        for r_start in tl.range(0, elements_per_channel, R_BLOCK):
            r_index = r_start + r_offsets
            r_mask = r_index < elements_per_channel
            hw = r_index % hw_size
            n = r_index // hw_size
            input_offsets = n * channels * hw_size + channels_offsets * hw_size + hw
            out_offsets = n * (channels * 2) * hw_size + (channels + channels_offsets) * hw_size + hw
            x = tl.load(x_ptr + input_offsets, mask=r_mask & c_mask, other=0.0).to(tl.float32)
            affine = (x - mean_b) * invstd_b * weight_b + bias_b
            relu = tl.where(affine != affine, affine, tl.maximum(affine, 0.0))
            tl.store(out_ptr + out_offsets, relu, mask=r_mask & c_mask)

    @triton.jit
    def _copy_cat_first_half_kernel(
        relu_input_ptr,
        out_ptr,
        numel: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        inner = offsets % (channels * hw_size)
        n = offsets // (channels * hw_size)
        values = tl.load(relu_input_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + n * (channels * 2) * hw_size + inner, values, mask=mask)


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
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects six inputs, got {len(inputs)}")

    convolution_84, arg459_1, arg460_1, arg461_1, arg462_1, relu_35 = inputs
    x = _expect_f32_tensor("convolution_84", convolution_84, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg459_1", arg459_1, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("arg460_1", arg460_1, VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("arg461_1", arg461_1, VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("arg462_1", arg462_1, VECTOR_SHAPE, VECTOR_STRIDE)
    relu_input = _expect_f32_tensor("relu_35", relu_35, INPUT_SHAPE, INPUT_STRIDE)

    if any(t.device != x.device for t in (running_mean, running_var, weight, bias, relu_input)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias, relu_input


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full training-BatchNorm update, affine ReLU, and cat scope."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_ghostnet_training_bn_cat.py")

    x, running_mean, running_var, weight, bias, relu_input = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _bn_update_relu_second_half_kernel[(triton.cdiv(CHANNELS, REDUCTION_X_BLOCK),)](
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
        X_BLOCK=REDUCTION_X_BLOCK,
        R_BLOCK=REDUCTION_R_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _copy_cat_first_half_kernel[(triton.cdiv(N * CHANNELS * HW, COPY_BLOCK),)](
        relu_input,
        output,
        numel=N * CHANNELS * HW,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK=COPY_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return output, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(item.detach().clone() if isinstance(item, torch.Tensor) else item for item in inputs)


def _normalize_outputs(out: Any) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        result: list[torch.Tensor] = []
        for item in out:
            result.extend(_normalize_outputs(item))
        return result
    return []


def _check_oracle_cloned_inputs(
    oracle_fn,
    instance,
    inputs: list[Any] | tuple[Any, ...],
    *,
    atol: float,
    rtol: float,
    skip_stochastic: bool = True,
) -> bool:
    del skip_stochastic
    eager_inputs = _clone_inputs(inputs)
    oracle_inputs = _clone_inputs(inputs)

    with torch.no_grad():
        eager = instance(*eager_inputs)
        oracle_out = oracle_fn(oracle_inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    oracle_list = _normalize_outputs(oracle_out)
    if len(oracle_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(oracle_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    all_pass = True
    alias_input_indices = (None, 1, 2)
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} "
                f"eager={expected.dtype}"
            )
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected.data_ptr() == eager_inputs[alias_index].data_ptr()
            actual_alias = actual.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != actual_alias or not actual_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias oracle={actual_alias} "
                    f"eager={expected_alias}"
                )
                all_pass = False
                continue

        if expected.is_floating_point():
            expected_f32 = expected.float()
            actual_f32 = actual.float()
            expected_nan = torch.isnan(expected_f32)
            actual_nan = torch.isnan(actual_f32)
            nan_mask_ok = torch.equal(expected_nan, actual_nan)
            finite = ~expected_nan & ~actual_nan
            if finite.any():
                finite_expected = expected_f32[finite]
                finite_actual = actual_f32[finite]
                max_diff = (finite_expected - finite_actual).abs().max().item()
                values_ok = torch.allclose(finite_expected, finite_actual, atol=atol, rtol=rtol)
            else:
                max_diff = 0.0
                values_ok = True
            ok = nan_mask_ok and values_ok
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()} max_finite_diff={max_diff:.2e} "
                f"nan_mask={'PASS' if nan_mask_ok else 'FAIL'})"
            )
        else:
            ok = torch.equal(expected, actual)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
        all_pass = all_pass and bool(ok)

    return all_pass


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
        ok = _check_oracle_cloned_inputs(
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
