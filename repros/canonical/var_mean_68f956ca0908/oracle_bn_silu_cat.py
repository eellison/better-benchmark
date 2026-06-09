"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileViT training BatchNorm-SiLU-channel-cat scope in one per-channel Triton program, including population `var_mean` over `[128,8,8]`, `rsqrt(var + 1e-5)`, in-place running-stat `copy_` updates, affine normalization, explicit `y / (exp(-y) + 1)` SiLU, and both halves of the final `[128,320,8,8]` cat output, whereas Inductor lowers the decomposed norm-template plus SiLU and cat consumers through the generic training-normalization schedule; Inductor cannot do this today because the norm-template scheduler does not preserve this fixed-shape activation and channel-cat consumer as one compact per-channel training-BN schedule with mutable running-stat returns; the fix is SCHEDULER_FUSION: teach the training BatchNorm template to fuse affine activation and static channel-cat epilogues while still emitting the running-stat side effects."""
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


N = 128
CHANNELS = 160
OUT_CHANNELS = 320
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_OUT_ELEMENTS = N * OUT_CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001220852154804
BLOCK_M = 8192
CLASSIFICATION = "SCHEDULER_FUSION"


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _bn_silu_cat_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
        out_ptr,
        channels: tl.constexpr,
        out_channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        sum_x = tl.sum(vals, axis=0)
        sum_x2 = tl.sum(vals * vals, axis=0)
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
        y = (vals - mean) * invstd
        y = y * weight + bias
        silu = y / (tl.exp(-y) + 1.0)

        out_residual_offsets = (n_idx * out_channels + channel) * hw_size + hw_idx
        out_silu_offsets = (n_idx * out_channels + channels + channel) * hw_size + hw_idx
        residual = tl.load(residual_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_residual_offsets, residual, mask=mask)
        tl.store(out_ptr + out_silu_offsets, silu, mask=mask)


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_32, arg294_1, arg295_1, arg296_1, arg297_1, add_208 = inputs
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    x = _expect_f32_tensor("convolution_32", convolution_32, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)
    running_mean = _expect_f32_tensor("arg294_1", arg294_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg295_1", arg295_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg296_1", arg296_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg297_1", arg297_1, (CHANNELS,), (1,))
    residual = _expect_f32_tensor("add_208", add_208, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, residual)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias, residual


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd = torch.rsqrt(var + EPS)
    y = (x - mean) * invstd
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    silu = y / (torch.exp(-y) + 1.0)
    out = torch.cat([residual, silu], dim=1)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return out, running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([128, 160, 8, 8], f32), T([160], f32), T([160], f32), T([160], f32), T([160], f32), T([128, 160, 8, 8], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    The returned running mean and variance are the original input tensors after
    the in-place `copy_` updates, matching the captured side effects.
    """
    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    out = torch.empty_strided(
        (N, OUT_CHANNELS, HEIGHT, WIDTH),
        (OUT_CHANNELS * HW, HW, WIDTH, 1),
        device=x.device,
        dtype=torch.float32,
    )
    _bn_silu_cat_kernel[(CHANNELS,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        residual,
        out,
        channels=CHANNELS,
        out_channels=OUT_CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK=BLOCK_M,
        num_warps=8,
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
        metadata_ok = (
            eager_tensor.shape == oracle_tensor.shape
            and eager_tensor.dtype == oracle_tensor.dtype
            and eager_tensor.stride() == oracle_tensor.stride()
            and eager_tensor.device == oracle_tensor.device
        )
        if not metadata_ok:
            print(
                f"  output {i}: SCOPE_MISMATCH "
                f"oracle=(shape={list(oracle_tensor.shape)} stride={oracle_tensor.stride()} "
                f"dtype={oracle_tensor.dtype}) eager=(shape={list(eager_tensor.shape)} "
                f"stride={eager_tensor.stride()} dtype={eager_tensor.dtype})"
            )
            all_pass = False
            continue

        if eager_tensor.is_floating_point():
            eager_f32 = eager_tensor.float()
            oracle_f32 = oracle_tensor.float()
            max_diff = (eager_f32 - oracle_f32).abs().max().item()
            ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(eager_tensor, oracle_tensor)
            print(f"  output {i}: {'PASS' if ok else 'FAIL'} (exact, dtype={eager_tensor.dtype})")

        if not ok:
            all_pass = False
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
                        help="Accepted for template-compatible CLI; this repro is deterministic")
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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
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
