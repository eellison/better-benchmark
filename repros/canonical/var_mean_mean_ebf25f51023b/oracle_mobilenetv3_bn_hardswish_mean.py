"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 training-BatchNorm var_mean, mutable running-stat copy_ updates, affine hard-swish, and 7x7 spatial mean with a dedicated two-kernel Triton schedule that carries only per-channel mean/invstd between the stats update and spatial-mean epilogue, whereas Inductor currently lowers the same scope through generic var_mean and persistent spatial-reduction templates; Inductor cannot do this today because its scheduler cannot select a full-scope BN-training hard-swish spatial-mean template while preserving copy_ side-effect returns; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat epilogues and sinks affine hard-swish plus fixed 7x7 spatial mean into the specialized lowering."""
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
    oracle_impl,
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
EPS = 0.001
MOMENTUM = 0.01
RUNNING_VAR_CORRECTION = 1.0000797257434426
BLOCK_HW = 64
TOTAL_ROWS = BATCH * CHANNELS
STAT_BLOCK_R = 16384
OUTPUT_BLOCK_ROWS = 32

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS, 1, 1)
OUTPUT_STRIDE = (CHANNELS, 1, 1, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _channel_stats_kernel(
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
        BLOCK_R_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        r_offsets = tl.arange(0, BLOCK_R_)
        sum_x = tl.full((), 0.0, tl.float32)
        sum_x2 = tl.full((), 0.0, tl.float32)
        for r_base in tl.range(0, elements_per_channel, BLOCK_R_):
            r = r_base + r_offsets
            r_mask = r < elements_per_channel
            hw = r % hw_size
            n = r // hw_size
            x_offsets = n * (channels * hw_size) + channel * hw_size + hw
            x = tl.load(x_ptr + x_offsets, mask=r_mask, other=0.0).to(tl.float32)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x2 / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(
            running_mean_ptr + channel,
            old_running_mean * (1.0 - momentum) + mean * momentum,
        )
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum)
            + var * running_var_correction * momentum,
        )
        tl.store(stats_ptr + channel, mean)
        tl.store(stats_ptr + channels + channel, invstd)

    @triton.jit
    def _hardswish_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        total_rows: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        hw_mask = hw_offsets < hw_size
        channels_idx = rows - (rows // channels) * channels
        offsets = rows[:, None] * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        mean = tl.load(stats_ptr + channels_idx).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channels_idx).to(tl.float32)
        weight = tl.load(weight_ptr + channels_idx).to(tl.float32)
        bias = tl.load(bias_ptr + channels_idx).to(tl.float32)

        affine = (x - mean[:, None]) * invstd[:, None]
        affine = affine * weight[:, None] + bias[:, None]
        clamp = tl.minimum(tl.maximum(affine + 3.0, 0.0), 6.0)
        hardswish = affine * clamp * 0.16666666666666666
        pooled = tl.sum(tl.where(hw_mask[None, :], hardswish, 0.0), axis=1) * (1.0 / 49.0)
        tl.store(out_ptr + rows, pooled)


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
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    convolution_57, arg289_1, arg290_1, arg291_1, arg292_1 = inputs
    x = _expect_f32_tensor("convolution_57", convolution_57, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg289_1", arg289_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg290_1", arg290_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg291_1", arg291_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg292_1", arg292_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([256, 960, 7, 7], f32), T([960], f32), T([960], f32), T([960], f32), T([960], f32))")
def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope, including running-stat mutations."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mobilenetv3_bn_hardswish_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )
    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _channel_stats_kernel[(CHANNELS,)](
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
        BLOCK_R_=STAT_BLOCK_R,
        num_warps=8,
        num_stages=1,
    )
    _hardswish_spatial_mean_kernel[(triton.cdiv(TOTAL_ROWS, OUTPUT_BLOCK_ROWS),)](
        x,
        weight,
        bias,
        stats,
        output,
        total_rows=TOTAL_ROWS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_ROWS=OUTPUT_BLOCK_ROWS,
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
    alias_input_indices: tuple[int | None, ...] = (None, 1, 2)
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

        alias_idx = alias_input_indices[i]
        if alias_idx is not None and actual_tensor.data_ptr() != oracle_inputs[alias_idx].data_ptr():
            print(f"  output {i}: SCOPE_MISMATCH expected alias of input {alias_idx}")
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
            f"max_finite_diff={max_diff:.2e})"
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
