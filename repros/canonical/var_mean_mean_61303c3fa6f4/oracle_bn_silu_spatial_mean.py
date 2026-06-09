"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete EfficientNet training-BatchNorm plus SiLU spatial-mean scope, including population per-channel `var_mean` over `[128,7,7]`, running mean/variance `copy_` updates, affine normalization, `x / (1 + exp(-x))`, and the returned `[128,1152,1,1]` mean directly in Triton, whereas tuned Inductor already lands in the same two-pass normalization plus nonlinear-reduction memory and launch envelope for this fixed shape; Inductor cannot materially improve it through a local oracle-style rewrite because the remaining work is dominated by required activation reads for statistics and output reduction, exp latency, running-stat stores, and output stores rather than avoidable intermediate traffic; the fix is BANDWIDTH_BOUND: record this as an at-floor case unless broader norm-template launch overhead or hardware math throughput improvements move the baseline."""
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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


BATCH = 128
CHANNELS = 1152
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = BATCH * HW
TOTAL_ELEMENTS = BATCH * CHANNELS * HW
EPS = 1.0e-3
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
STAT_BLOCK = 8192
MEAN_BLOCK_N = 4
MEAN_BLOCK_C = 16
MEAN_BLOCK_HW = 64
CLASSIFICATION = "BANDWIDTH_BOUND"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _stats_update_kernel(
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
        BLOCK_K: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < elements_per_channel
        batch = offsets // hw_size
        hw = offsets - batch * hw_size
        x_offsets = (batch * channels + channel) * hw_size + hw

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        total = tl.sum(vals, axis=0)
        total_sq = tl.sum(vals * vals, axis=0)
        mean = total / elements_per_channel
        var = total_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel).to(tl.float32)
        tl.store(running_mean_ptr + channel, old_mean * (1.0 - momentum) + mean * momentum)
        tl.store(
            running_var_ptr + channel,
            old_var * (1.0 - momentum) + var * running_var_correction * momentum,
        )
        tl.store(mean_ptr + channel, mean)
        tl.store(invstd_ptr + channel, invstd)

    @triton.jit
    def _affine_silu_spatial_mean_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        batch_size: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_HW: tl.constexpr,
    ):
        n_offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        c_offsets = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
        hw_offsets = tl.arange(0, BLOCK_HW)

        n = n_offsets[:, None, None]
        c = c_offsets[None, :, None]
        hw = hw_offsets[None, None, :]
        mask = (n < batch_size) & (c < channels) & (hw < hw_size)
        x_offsets = (n * channels + c) * hw_size + hw

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < channels, other=0.0).to(tl.float32)

        y = (vals - mean[None, :, None]) * invstd[None, :, None]
        y = y * weight[None, :, None] + bias[None, :, None]
        silu = y / (1.0 + tl.exp(-y))
        spatial_sum = tl.sum(tl.where(hw < hw_size, silu, 0.0), axis=2)
        out = spatial_sum / hw_size
        out_offsets = n_offsets[:, None] * channels + c_offsets[None, :]
        out_mask = (n_offsets[:, None] < batch_size) & (c_offsets[None, :] < channels)
        tl.store(out_ptr + out_offsets, out, mask=out_mask)


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


def _validate_inputs(
    inputs: list[Any] | tuple[Any, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 5:
        raise ValueError(f"{REPRO_ID} expects 5 inputs, got {len(inputs)}")

    x, running_mean, running_var, weight, bias = inputs
    x = _expect_f32_tensor(
        "convolution_76",
        x,
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg339_1", running_mean, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg340_1", running_var, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg341_1", weight, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg342_1", bias, (CHANNELS,), (1,))

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([128, 1152, 7, 7], f32), T([1152], f32), T([1152], f32), T([1152], f32), T([1152], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation."""
    if triton is None:
        raise RuntimeError("Triton is required for this oracle")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if not x.is_cuda:
        raise RuntimeError("This oracle requires CUDA tensors")

    mean = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    out = torch.empty_strided(
        (BATCH, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _stats_update_kernel[(CHANNELS,)](
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
        BLOCK_K=STAT_BLOCK,
        num_warps=8,
        num_stages=3,
    )
    _affine_silu_spatial_mean_kernel[
        (triton.cdiv(BATCH, MEAN_BLOCK_N), triton.cdiv(CHANNELS, MEAN_BLOCK_C))
    ](
        x,
        weight,
        bias,
        mean,
        invstd,
        out,
        batch_size=BATCH,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_N=MEAN_BLOCK_N,
        BLOCK_C=MEAN_BLOCK_C,
        BLOCK_HW=MEAN_BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return out, running_mean, running_var


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        if isinstance(item, torch.Tensor):
            cloned.append(item.clone())
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
        eager = instance(*eager_inputs)
        oracle_out = oracle_forward(oracle_inputs)
        if isinstance(oracle_inputs[0], torch.Tensor) and oracle_inputs[0].is_cuda:
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
    for i, (eager_tensor, oracle_tensor) in enumerate(zip(eager_list, oracle_list)):
        shape_ok = eager_tensor.shape == oracle_tensor.shape
        dtype_ok = eager_tensor.dtype == oracle_tensor.dtype
        stride_ok = eager_tensor.stride() == oracle_tensor.stride()
        if not shape_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(oracle_tensor.shape)} "
                f"eager={list(eager_tensor.shape)}"
            )
            all_pass = False
            continue
        if not dtype_ok:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={oracle_tensor.dtype} eager={eager_tensor.dtype}")
            all_pass = False
            continue
        if not stride_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={oracle_tensor.stride()} "
                f"eager={eager_tensor.stride()}"
            )
            all_pass = False
            continue

        eager_f32 = eager_tensor.float()
        oracle_f32 = oracle_tensor.float()
        max_diff = (eager_f32 - oracle_f32).abs().max().item()
        values_ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
            f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
            f"stride={eager_tensor.stride()} max_diff={max_diff:.2e})"
        )
        all_pass = all_pass and bool(values_ok)

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 3
        and oracle_out[1] is oracle_inputs[1]
        and oracle_out[2] is oracle_inputs[2]
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
