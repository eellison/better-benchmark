"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete training-BatchNorm plus affine SiLU scope for f32 `[128,160,8,8]`, including per-channel `var_mean(correction=0)`, in-place running mean/variance `copy_` aliases, and the returned contiguous activation, using a stats/update kernel plus a parallel normalized epilogue, whereas tuned Inductor already lowers the decomposed var_mean/update/normalization/SiLU graph into the same practical memory-traffic and launch envelope for this fixed shape; Inductor cannot materially improve this repro through local scheduler fusion, split-K, or algebraic rewrites because the remaining cost is the required activation read, channel-stat reduction, affine reads, exp/div epilogue, output store, and running-stat updates; the fix is BANDWIDTH_BOUND: record this as an at-floor training-BN SiLU case unless broader normalization-template or math-codegen improvements move the family."""
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


N = 128
CHANNELS = 160
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804
STATS_BLOCK_N = 128
STATS_BLOCK_HW = 64


# --- Oracle kernel(s) ---

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
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        n_offsets = tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + offsets).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum_sq = tl.sum(x * x, axis=1)
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

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 512}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=4, num_stages=3),
        ],
        key=["total"],
    )
    @triton.jit
    def _bn_silu_epilogue_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        stats_ptr,
        out_ptr,
        total: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        channel = (offsets // hw_size) % channels

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(stats_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(stats_ptr + channels + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * invstd * weight + bias
        out = y / (tl.exp(-y) + 1.0)
        tl.store(out_ptr + offsets, out, mask=mask)


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
        raise RuntimeError("Triton is required for oracle_training_bn_silu.py")
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects five inputs, got {len(inputs)}")

    x, running_mean, running_var, weight, bias = inputs
    x_t = _expect_f32_tensor(
        "convolution_33",
        x,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    running_mean_t = _expect_f32_tensor("arg300_1", running_mean, vector_shape, vector_stride)
    running_var_t = _expect_f32_tensor("arg301_1", running_var, vector_shape, vector_stride)
    weight_t = _expect_f32_tensor("arg302_1", weight, vector_shape, vector_stride)
    bias_t = _expect_f32_tensor("arg303_1", bias, vector_shape, vector_stride)

    if any(t.device != x_t.device for t in (running_mean_t, running_var_t, weight_t, bias_t)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x_t, running_mean_t, running_var_t, weight_t, bias_t


@oracle_impl(hardware="H100", shapes="(T([128, 160, 8, 8], f32), T([160], f32), T([160], f32), T([160], f32), T([160], f32))")
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
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    stats = torch.empty_strided(
        (2, CHANNELS),
        (CHANNELS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
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
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        eps=EPS,
        BLOCK_N_=STATS_BLOCK_N,
        BLOCK_HW_=STATS_BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    grid = lambda meta: (triton.cdiv(TOTAL, meta["BLOCK_SIZE"]),)
    _bn_silu_epilogue_kernel[grid](
        x,
        weight,
        bias,
        stats,
        out,
        total=TOTAL,
        channels=CHANNELS,
        hw_size=HW,
    )
    return out, running_mean, running_var


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
        actual = oracle_fn(oracle_inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(actual_list) != len(eager_list):
        print(
            f"  SCOPE_MISMATCH: oracle produces {len(actual_list)} outputs, "
            f"eager produces {len(eager_list)}"
        )
        return False

    alias_input_indices = (None, 1, 2)
    all_pass = True
    for i, (expected, observed) in enumerate(zip(eager_list, actual_list)):
        metadata_ok = (
            expected.shape == observed.shape
            and expected.dtype == observed.dtype
            and expected.stride() == observed.stride()
            and expected.device == observed.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(observed.shape)} stride={observed.stride()} dtype={observed.dtype}) "
                f"eager=(shape={list(expected.shape)} stride={expected.stride()} dtype={expected.dtype})"
            )
            all_pass = False
            continue

        alias_index = alias_input_indices[i]
        if alias_index is not None:
            expected_alias = expected.data_ptr() == eager_inputs[alias_index].data_ptr()
            observed_alias = observed.data_ptr() == oracle_inputs[alias_index].data_ptr()
            if expected_alias != observed_alias or not observed_alias:
                print(
                    f"  output {i}: SCOPE_MISMATCH alias "
                    f"oracle={observed_alias} eager={expected_alias}"
                )
                all_pass = False
                continue

        if expected.is_floating_point():
            expected_f32 = expected.float()
            observed_f32 = observed.float()
            expected_nan = torch.isnan(expected_f32)
            observed_nan = torch.isnan(observed_f32)
            nan_mask_ok = torch.equal(expected_nan, observed_nan)
            finite = ~(expected_nan | observed_nan)
            if finite.any():
                max_diff = (expected_f32[finite] - observed_f32[finite]).abs().max().item()
                values_ok = torch.allclose(
                    expected_f32[finite],
                    observed_f32[finite],
                    atol=atol,
                    rtol=rtol,
                )
            else:
                max_diff = 0.0
                values_ok = True
            ok = nan_mask_ok and values_ok
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} stride={expected.stride()} "
                f"max_finite_diff={max_diff:.2e} nan_count={int(expected_nan.sum().item())})"
            )
        else:
            ok = torch.equal(expected, observed)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={expected.dtype})")
        all_pass = all_pass and bool(ok)

    return all_pass


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
