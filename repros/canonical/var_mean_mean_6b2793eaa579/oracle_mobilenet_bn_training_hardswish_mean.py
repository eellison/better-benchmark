"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 training-BatchNorm var_mean, running-stat copy_ side effects, affine hard-swish, and final 7x7 spatial mean with a low-register channel-statistics pass and a direct hard-swish pooling epilogue that returns only the required `[256,960]` pooled tensor plus the two mutated running-stat aliases, whereas Inductor currently emits separate generic reduction regions for the BN-training stats/update and the hard-swish spatial-mean consumer; Inductor cannot do this today because scheduler fusion cannot keep mutable training-normalization side outputs and a following activation-fed reduction consumer in one reusable normalization template; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat epilogues while sinking affine hard-swish and small spatial-mean consumers into the same channel-tiled lowering."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


BATCH = 256
CHANNELS = 960
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
EPS = 1.0e-3
MOMENTUM = 0.01
RUNNING_VAR_CORRECTION = 1.0000797257434426
STAT_C_BLOCK = 1
STAT_R_BLOCK = 1024
OUT_ROW_BLOCK = 8
BLOCK_HW = 64

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _stats_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        mean_ptr,
        invstd_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        channel = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        r_offsets = tl.arange(0, BLOCK_R)
        channel_mask = channel < channels
        sum_x = tl.zeros((BLOCK_C,), tl.float32)
        sum_x2 = tl.zeros((BLOCK_C,), tl.float32)

        for r_start in tl.range(0, elements_per_channel, BLOCK_R):
            r = r_start + r_offsets
            r_mask = r < elements_per_channel
            n = r // hw_size
            hw = r - n * hw_size
            x = tl.load(
                x_ptr + n[None, :] * channels * hw_size + channel[:, None] * hw_size + hw[None, :],
                mask=channel_mask[:, None] & r_mask[None, :],
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            sum_x += tl.sum(x, axis=1)
            sum_x2 += tl.sum(x * x, axis=1)

        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel, mask=channel_mask, other=0.0).to(tl.float32)
        tl.store(mean_ptr + channel, mean, mask=channel_mask)
        tl.store(invstd_ptr + channel, invstd, mask=channel_mask)
        tl.store(
            running_mean_ptr + channel,
            old_running_mean * (1.0 - momentum) + mean * momentum,
            mask=channel_mask,
        )
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum)
            + var * running_var_correction * momentum,
            mask=channel_mask,
        )

    @triton.jit
    def _hardswish_spatial_mean_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        row_mask = rows < total_rows
        hw_mask = hw_offsets < hw_size
        channel = rows - (rows // channels) * channels

        x = tl.load(
            x_ptr + rows[:, None] * hw_size + hw_offsets[None, :],
            mask=row_mask[:, None] & hw_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=row_mask, other=0.0).to(tl.float32)

        y = (x - mean[:, None]) * invstd[:, None]
        y = y * weight[:, None] + bias[:, None]
        relu6 = tl.minimum(tl.maximum(y + 3.0, 0.0), 6.0)
        y = y * relu6 * (1.0 / 6.0)
        y = tl.where(row_mask[:, None] & hw_mask[None, :], y, 0.0)
        pooled = tl.sum(y, axis=1) * (1.0 / 49.0)
        tl.store(out_ptr + rows, pooled, mask=row_mask)


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

    convolution_61, arg305_1, arg306_1, arg307_1, arg308_1, shape_param = inputs
    x = _expect_f32_tensor("convolution_61", convolution_61, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg305_1", arg305_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg306_1", arg306_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg307_1", arg307_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg308_1", arg308_1, STAT_SHAPE, STAT_STRIDE)

    if tuple(int(dim) for dim in shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope, including running-stat mutations."""
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_training_hardswish_mean.py")

    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided(STAT_SHAPE, STAT_STRIDE, device=x.device, dtype=torch.float32)

    total_rows = BATCH * CHANNELS
    _stats_kernel[(triton.cdiv(CHANNELS, STAT_C_BLOCK),)](
        x,
        running_mean,
        running_var,
        mean,
        invstd,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_C=STAT_C_BLOCK,
        BLOCK_R=STAT_R_BLOCK,
        num_warps=4,
        num_stages=1,
    )
    _hardswish_spatial_mean_kernel[(triton.cdiv(total_rows, OUT_ROW_BLOCK),)](
        x,
        mean,
        invstd,
        weight,
        bias,
        output,
        total_rows=total_rows,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_ROWS=OUT_ROW_BLOCK,
        BLOCK_HW_=BLOCK_HW,
        num_warps=1,
        num_stages=1,
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
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
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
        if expected_tensor.shape != actual_tensor.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual_tensor.shape)} "
                f"eager={list(expected_tensor.shape)}"
            )
            all_pass = False
            continue
        if expected_tensor.dtype != actual_tensor.dtype:
            print(
                f"  output {i}: SCOPE_MISMATCH dtype oracle={actual_tensor.dtype} "
                f"eager={expected_tensor.dtype}"
            )
            all_pass = False
            continue
        if expected_tensor.stride() != actual_tensor.stride():
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

        expected_f32 = expected_tensor.float()
        actual_f32 = actual_tensor.float()
        expected_nan = torch.isnan(expected_f32)
        actual_nan = torch.isnan(actual_f32)
        finite = ~expected_nan & ~actual_nan
        if finite.any():
            max_diff = (expected_f32[finite] - actual_f32[finite]).abs().max().item()
            values_ok = torch.allclose(
                expected_f32[finite],
                actual_f32[finite],
                atol=atol,
                rtol=rtol,
            )
        else:
            max_diff = 0.0
            values_ok = True
        nan_ok = torch.equal(expected_nan, actual_nan)
        ok = bool(values_ok and nan_ok)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected_tensor.shape)} dtype={expected_tensor.dtype} "
            f"stride={expected_tensor.stride()} max_finite_diff={max_diff:.2e})"
        )
        all_pass = all_pass and ok

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
