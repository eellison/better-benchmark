"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete MobileNetV3 training-BatchNorm var_mean, running-stat copy_ side effects, affine ReLU, and final `[512,120,1,1]` spatial mean with Triton split statistics plus a fused normalize/ReLU/pool epilogue, whereas Inductor lowers the captured var_mean/update/affine/ReLU/mean graph through its generic normalization schedule; Inductor cannot do this today because norm-template canonicalization does not keep mutable training-BN side outputs and the downstream spatial reduction inside one planned full-scope schedule; the fix is SCHEDULER_FUSION: teach the BN-training template to expose running-stat side effects while fusing immediate affine activation and spatial-reduction consumers."""
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


# --- Oracle kernel(s) ---
N = 512
CHANNELS = 120
HEIGHT = 28
WIDTH = 28
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000024912370735
STAT_BLOCK_N = 8
STAT_BLOCKS = N // STAT_BLOCK_N
POOL_BLOCK_N = 8
BLOCK_HW = 1024
FINAL_BLOCK = 64
INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
STAT_SHAPE = (CHANNELS,)
STAT_STRIDE = (1,)
OUTPUT_SHAPE = (N, CHANNELS, 1, 1)
OUTPUT_STRIDE = (CHANNELS, 1, 1, 1)


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        n_size: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        channel = tl.program_id(1)
        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        n_mask = n_offsets < n_size
        hw_mask = hw_offsets < hw_size
        mask = n_mask[:, None] & hw_mask[None, :]

        x_offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        row_sum = tl.sum(x, axis=1)
        row_sum_sq = tl.sum(x * x, axis=1)
        sum_x = tl.sum(row_sum, axis=0)
        sum_x_sq = tl.sum(row_sum_sq, axis=0)

        out_offset = channel * stat_blocks + n_block
        tl.store(partial_ptr + out_offset, sum_x)
        tl.store(partial_ptr + channels * stat_blocks + out_offset, sum_x_sq)

    @triton.jit
    def _bn_relu_spatial_mean_from_partials_kernel(
        x_ptr,
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        partial_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        n_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
        BLOCK_P_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        channel = tl.program_id(1)

        p_offsets = tl.arange(0, BLOCK_P_)
        p_mask = p_offsets < stat_blocks
        p_base = channel * stat_blocks + p_offsets
        sums = tl.load(partial_ptr + p_base, mask=p_mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(
            partial_ptr + channels * stat_blocks + p_base,
            mask=p_mask,
            other=0.0,
        ).to(tl.float32)
        sum_x = tl.sum(sums, axis=0)
        sum_x_sq = tl.sum(sums_sq, axis=0)
        mean = sum_x / elements_per_channel
        var = sum_x_sq / elements_per_channel - mean * mean
        var = tl.maximum(var, 0.0)
        invstd = tl.rsqrt(var + eps)

        old_running_mean = tl.load(running_mean_ptr + channel).to(tl.float32)
        old_running_var = tl.load(running_var_ptr + channel).to(tl.float32)
        update_mask = n_block == 0
        tl.store(
            running_mean_ptr + channel,
            old_running_mean * (1.0 - momentum) + mean * momentum,
            mask=update_mask,
        )
        tl.store(
            running_var_ptr + channel,
            old_running_var * (1.0 - momentum)
            + var * running_var_correction * momentum,
            mask=update_mask,
        )

        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        n_mask = n_offsets < n_size
        hw_mask = hw_offsets < hw_size
        mask = n_mask[:, None] & hw_mask[None, :]
        x_offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + channel).to(tl.float32)
        bias = tl.load(bias_ptr + channel).to(tl.float32)
        y = (x - mean) * invstd
        y = y * weight + bias
        relu = tl.where(y != y, y, tl.maximum(y, 0.0))
        pooled = tl.sum(tl.where(mask, relu, 0.0), axis=1) / hw_size

        out_offsets = n_offsets * channels + channel
        tl.store(out_ptr + out_offsets, pooled, mask=n_mask)


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

    convolution_20, arg107_1, arg108_1, arg109_1, arg110_1 = inputs
    x = _expect_f32_tensor("convolution_20", convolution_20, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("arg107_1", arg107_1, STAT_SHAPE, STAT_STRIDE)
    running_var = _expect_f32_tensor("arg108_1", arg108_1, STAT_SHAPE, STAT_STRIDE)
    weight = _expect_f32_tensor("arg109_1", arg109_1, STAT_SHAPE, STAT_STRIDE)
    bias = _expect_f32_tensor("arg110_1", arg110_1, STAT_SHAPE, STAT_STRIDE)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return x, running_mean, running_var, weight, bias


@oracle_impl(hardware="H100", shapes="(T([512, 120, 28, 28], f32), T([120], f32), T([120], f32), T([120], f32), T([120], f32))")
def oracle_forward(inputs):
    """Run the full Repro.forward scope, including running-stat copy_ effects.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).

    Args:
        inputs: tuple of tensors/values from get_inputs(), identical to what
                Repro.forward() receives via Repro()(*inputs).

    Returns:
        Same output structure as Repro()(*inputs).
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_bn_relu_spatial_mean.py")

    x, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    partial = torch.empty_strided(
        (2, CHANNELS, STAT_BLOCKS),
        (CHANNELS * STAT_BLOCKS, STAT_BLOCKS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x.device,
        dtype=torch.float32,
    )

    _partial_stats_kernel[(STAT_BLOCKS, CHANNELS)](
        x,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        n_size=N,
        stat_blocks=STAT_BLOCKS,
        BLOCK_N_=STAT_BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
        num_warps=8,
        num_stages=3,
    )
    _bn_relu_spatial_mean_from_partials_kernel[(triton.cdiv(N, POOL_BLOCK_N), CHANNELS)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        partial,
        output,
        channels=CHANNELS,
        hw_size=HW,
        n_size=N,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N_=POOL_BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
        BLOCK_P_=FINAL_BLOCK,
        num_warps=8,
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
    for i, (expected, actual) in enumerate(zip(eager_list, oracle_list)):
        if expected.shape != actual.shape:
            print(
                f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} "
                f"eager={list(expected.shape)}"
            )
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(f"  output {i}: WARNING dtype mismatch oracle={actual.dtype} eager={expected.dtype}")
        if expected.stride() != actual.stride():
            print(
                f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} "
                f"eager={expected.stride()}"
            )
            all_pass = False
            continue

        expected_f32 = expected.float()
        actual_f32 = actual.float()
        max_diff = (expected_f32 - actual_f32).abs().max().item()
        ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
        print(
            f"  output {i}: {'PASS' if ok else 'FAIL'} "
            f"(shape={list(expected.shape)} dtype={expected.dtype} max_diff={max_diff:.2e})"
        )
        if not ok:
            all_pass = False

    alias_ok = (
        isinstance(oracle_out, tuple)
        and len(oracle_out) == 3
        and oracle_out[1] is oracle_inputs[1]
        and oracle_out[2] is oracle_inputs[2]
    )
    if not alias_ok:
        print("  SCOPE_MISMATCH: running-stat outputs must alias input tensors")
    return all_pass and alias_ok


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
