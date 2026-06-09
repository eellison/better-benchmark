"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet training-BatchNorm forward scope, including population `var_mean` over `[512,7,7]` per channel, `rsqrt(var + 1e-5)`, the affine normalized `[512,80,7,7]` output, the `[1,80,1,1]` mean side view, and both in-place running-stat `copy_` updates, by splitting the long per-channel statistics reduction into many partial programs before a small finalizer and a coalesced affine epilogue, whereas Inductor's current norm-template lowering does not expose this fixed-shape BN-training reduction as a split-K statistics schedule feeding the required side outputs; Inductor cannot do this today because the normalization scheduler lacks a guarded cooperative split-K variant for small-channel, large-`N*H*W` training BatchNorm with mutable running-stat returns; the fix is COOPERATIVE_SPLIT_K: add a BN-training stats template that splits the reduction dimension, finalizes mean/invstd/running-stat side outputs, and feeds the normalized affine store without falling back to the generic per-channel reduction plan."""
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


N = 512
CHANNELS = 80
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361
STAT_BLOCK = 1024
STAT_BLOCKS = 25
FINAL_BLOCK = 32
POINTWISE_BLOCK = 256
CLASSIFICATION = "COOPERATIVE_SPLIT_K"


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
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        channel = tl.program_id(1)
        offsets = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        part_sum = tl.sum(vals, axis=0)
        part_sum_sq = tl.sum(vals * vals, axis=0)
        out_offset = channel * stat_blocks + block_id
        tl.store(partial_ptr + out_offset, part_sum)
        tl.store(partial_ptr + channels * stat_blocks + out_offset, part_sum_sq)

    @triton.jit
    def _finalize_stats_kernel(
        partial_ptr,
        running_mean_ptr,
        running_var_ptr,
        invstd_out_ptr,
        mean_out_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_P)
        mask = offsets < stat_blocks
        base = channel * stat_blocks + offsets

        sums = tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_ptr + channels * stat_blocks + base, mask=mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(sums, axis=0)
        total_sum_sq = tl.sum(sums_sq, axis=0)
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
        tl.store(invstd_out_ptr + channel, invstd)
        tl.store(mean_out_ptr + channel, mean)

    @triton.jit
    def _normalize_affine_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        invstd_ptr,
        mean_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_elements
        channel = (offsets // hw_size) % channels

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        tl.store(out_ptr + offsets, y, mask=mask)


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

    convolution_91, arg493_1, arg494_1, arg495_1, arg496_1 = inputs
    x = _expect_f32_tensor(
        "convolution_91",
        convolution_91,
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
    )
    running_mean = _expect_f32_tensor("arg493_1", arg493_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg494_1", arg494_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg495_1", arg495_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg496_1", arg496_1, (CHANNELS,), (1,))

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd_4d = torch.rsqrt(var + EPS)
    y = (x - mean) * invstd_4d
    y = y * weight[None, :, None, None] + bias[None, :, None, None]
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    invstd_1d = invstd_4d.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(running_var * (1.0 - MOMENTUM) + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM)
    return invstd_1d, y, mean_1d[None, :, None, None], running_mean, running_var


@oracle_impl(hardware="H100", shapes="(T([512, 80, 7, 7], f32), T([80], f32), T([80], f32), T([80], f32), T([80], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward computation.

    The returned running mean and variance are the same input tensors after the
    in-place `copy_` updates, matching the captured training-BN side effects.
    """
    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    partial = torch.empty_strided(
        (2, CHANNELS, STAT_BLOCKS),
        (CHANNELS * STAT_BLOCKS, STAT_BLOCKS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    normalized = torch.empty_strided(
        (N, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, HW, WIDTH, 1),
        device=x.device,
        dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (1, CHANNELS, 1, 1),
        (CHANNELS, 1, 1, 1),
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(STAT_BLOCKS, CHANNELS)](
        x,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        BLOCK_M=STAT_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    _finalize_stats_kernel[(CHANNELS,)](
        partial,
        running_mean,
        running_var,
        invstd,
        mean,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_P=FINAL_BLOCK,
        num_warps=1,
        num_stages=3,
    )
    _normalize_affine_kernel[(triton.cdiv(TOTAL_ELEMENTS, POINTWISE_BLOCK),)](
        x,
        weight,
        bias,
        invstd,
        mean,
        normalized,
        total_elements=TOTAL_ELEMENTS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_M=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return invstd, normalized, mean, running_mean, running_var


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
            if isinstance(item, torch.Tensor):
                result.append(item)
            elif isinstance(item, (tuple, list)):
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

        if eager_tensor.is_floating_point():
            eager_f32 = eager_tensor.float()
            oracle_f32 = oracle_tensor.float()
            max_diff = (eager_f32 - oracle_f32).abs().max().item()
            values_ok = torch.allclose(eager_f32, oracle_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()} max_diff={max_diff:.2e})"
            )
        else:
            values_ok = torch.equal(eager_tensor, oracle_tensor)
            print(
                f"  output {i}: {'PASS' if values_ok else 'FAIL'} "
                f"(exact, shape={list(eager_tensor.shape)} dtype={eager_tensor.dtype} "
                f"stride={eager_tensor.stride()})"
            )
        all_pass = all_pass and bool(values_ok)

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 5
        and oracle_out[3] is oracle_inputs[1]
        and oracle_out[4] is oracle_inputs[2]
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
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
