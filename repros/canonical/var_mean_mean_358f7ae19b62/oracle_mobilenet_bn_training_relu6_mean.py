"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV2 training-BatchNorm var_mean, running-stat copy_ side effects, affine ReLU6, spatial mean, and three returned outputs in one channel-specialized Triton kernel, whereas Inductor currently schedules the BN-training reduction/update and the downstream ReLU6 spatial-mean consumer as separate generic regions; Inductor cannot do this today because scheduler fusion cannot sink a normalization reduction with mutable running-stat side outputs into a following reduction consumer; the fix is SCHEDULER_FUSION: add a BN-training fusion schedule that exposes mean/invstd/running-stat epilogues while fusing immediate affine activation and spatial reduction consumers."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


BATCH = 128
CHANNELS = 1280
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_N = 128
BLOCK_HW = 64

INPUT_SHAPE = (BATCH, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (BATCH, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)
CLASSIFICATION = "SCHEDULER_FUSION"
ACTIONABLE = True


if triton is not None:

    @triton.jit
    def _bn_training_relu6_spatial_mean_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        batch: tl.constexpr,
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
        n_mask = n_offsets < batch
        hw_mask = hw_offsets < hw_size
        mask = n_mask[:, None] & hw_mask[None, :]

        x_offsets = (
            (n_offsets[:, None] * channels + channel) * hw_size
            + hw_offsets[None, :]
        )
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum2 = tl.sum(x * x, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x2 = tl.sum(row_sum2, axis=0)
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

        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        y = tl.where(y < 0.0, 0.0, y)
        y = tl.where(y > 6.0, 6.0, y)
        y = tl.where(mask, y, 0.0)
        pooled = tl.sum(y, axis=1) * (1.0 / 49.0)
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

    convolution_51, arg309_1, arg310_1, arg311_1, arg312_1, shape_param = inputs
    x = _expect_f32_tensor("convolution_51", convolution_51, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg309_1", arg309_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg310_1", arg310_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg311_1", arg311_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg312_1", arg312_1, STAT_SHAPE, STAT_STRIDE)

    if tuple(int(dim) for dim in shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output view shape parameter: {shape_param!r}")

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 1280, 7, 7], f32), T([1280], f32), T([1280], f32), T([1280], f32), T([1280], f32), S([128, 1280]))")
def oracle_forward(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the full Repro.forward scope.

    The running mean and variance inputs are intentionally mutated in place to
    match the two aten.copy_ return values from the captured training-BN graph.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_mobilenet_bn_training_relu6_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _bn_training_relu6_spatial_mean_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        output,
        batch=BATCH,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N_=BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
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
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
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

        expected_f32 = expected.float()
        actual_f32 = actual.float()
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
            f"(shape={list(expected.shape)} dtype={expected.dtype} "
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
