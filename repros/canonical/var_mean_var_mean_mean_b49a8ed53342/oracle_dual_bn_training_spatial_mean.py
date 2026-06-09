"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full two-branch training-BatchNorm RepVGG block, including both per-channel var_mean reductions over channels-last `[128, 1408, 7, 7]` inputs, four running-stat copy_ side effects, affine branch sum, ReLU, and final spatial mean, using a paired statistics kernel plus a fused normalize/update/pool Triton kernel that preserves returned input aliases, whereas Inductor lowers the captured norm-template graph through separate generic reduction and consumer schedules; Inductor cannot do this today because normalization template fusion does not plan sibling mutable BN-training reductions together with a downstream post-add spatial reduction over channels-last activations; the fix is SCHEDULER_FUSION: teach the scheduler/norm template to fuse paired BN-training statistics, running-stat side effects, and immediate activation/pooling consumers into one full-scope channel schedule."""
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


CLASSIFICATION = "SCHEDULER_FUSION"

N = 128
CHANNELS = 1408
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
OUTPUT_SHAPE = (N, CHANNELS)
OUTPUT_STRIDE = (CHANNELS, 1)

STAT_BLOCK_M = 512
STAT_BLOCK_C = 8
STAT_BLOCKS = (ELEMENTS_PER_CHANNEL + STAT_BLOCK_M - 1) // STAT_BLOCK_M
EPILOGUE_BLOCK_N = 8
EPILOGUE_BLOCK_C = 32
PARTIAL_BLOCK = 16


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_partial_stats_kernel(
        x0_ptr,
        x1_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_block = tl.program_id(0)
        m_block = tl.program_id(1)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = (m_offsets[:, None] < elements_per_channel) & (
            c_offsets[None, :] < channels
        )

        flat = m_offsets[:, None] * channels + c_offsets[None, :]
        x0 = tl.load(x0_ptr + flat, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + flat, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(x0, axis=0)
        sum1 = tl.sum(x1, axis=0)
        sumsq0 = tl.sum(x0 * x0, axis=0)
        sumsq1 = tl.sum(x1 * x1, axis=0)

        out = m_block * channels + c_offsets
        stat_stride = stat_blocks * channels
        c_mask = c_offsets < channels
        tl.store(partial_ptr + out, sum0, mask=c_mask)
        tl.store(partial_ptr + stat_stride + out, sum1, mask=c_mask)
        tl.store(partial_ptr + 2 * stat_stride + out, sumsq0, mask=c_mask)
        tl.store(partial_ptr + 3 * stat_stride + out, sumsq1, mask=c_mask)

    @triton.jit
    def _dual_bn_update_relu_spatial_mean_kernel(
        x0_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        weight0_ptr,
        bias0_ptr,
        x1_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        weight1_ptr,
        bias1_ptr,
        partial_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        stat_blocks: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        c_block = tl.program_id(1)
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        c_offsets = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        n_mask = n_offsets < 128
        c_mask = c_offsets < channels

        p_offsets = tl.arange(0, BLOCK_P)
        p_mask = p_offsets < stat_blocks
        p_base = p_offsets[:, None] * channels + c_offsets[None, :]
        stat_stride = stat_blocks * channels

        partial_mask = p_mask[:, None] & c_mask[None, :]
        sum0_parts = tl.load(partial_ptr + p_base, mask=partial_mask, other=0.0).to(tl.float32)
        sum1_parts = tl.load(
            partial_ptr + stat_stride + p_base,
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)
        sumsq0_parts = tl.load(
            partial_ptr + 2 * stat_stride + p_base,
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)
        sumsq1_parts = tl.load(
            partial_ptr + 3 * stat_stride + p_base,
            mask=partial_mask,
            other=0.0,
        ).to(tl.float32)

        scale = 1.0 / elements_per_channel
        mean0 = tl.sum(sum0_parts, axis=0) * scale
        mean1 = tl.sum(sum1_parts, axis=0) * scale
        var0 = tl.sum(sumsq0_parts, axis=0) * scale - mean0 * mean0
        var1 = tl.sum(sumsq1_parts, axis=0) * scale - mean1 * mean1
        var0 = tl.maximum(var0, 0.0)
        var1 = tl.maximum(var1, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)

        update_mask = (n_block == 0) & c_mask
        old_mean0 = tl.load(running_mean0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        old_var0 = tl.load(running_var0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        old_mean1 = tl.load(running_mean1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        old_var1 = tl.load(running_var1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        tl.store(
            running_mean0_ptr + c_offsets,
            old_mean0 * (1.0 - momentum) + mean0 * momentum,
            mask=update_mask,
        )
        tl.store(
            running_var0_ptr + c_offsets,
            old_var0 * (1.0 - momentum) + var0 * running_var_correction * momentum,
            mask=update_mask,
        )
        tl.store(
            running_mean1_ptr + c_offsets,
            old_mean1 * (1.0 - momentum) + mean1 * momentum,
            mask=update_mask,
        )
        tl.store(
            running_var1_ptr + c_offsets,
            old_var1 * (1.0 - momentum) + var1 * running_var_correction * momentum,
            mask=update_mask,
        )

        weight0 = tl.load(weight0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
        value_mask = n_mask[:, None] & c_mask[None, :]
        for hw in tl.static_range(0, 49):
            flat = (n_offsets[:, None] * hw_size + hw) * channels + c_offsets[None, :]
            x0 = tl.load(x0_ptr + flat, mask=value_mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptr + flat, mask=value_mask, other=0.0).to(tl.float32)
            y0 = (x0 - mean0[None, :]) * invstd0[None, :] * weight0[None, :] + bias0[None, :]
            y1 = (x1 - mean1[None, :]) * invstd1[None, :] * weight1[None, :] + bias1[None, :]
            summed = y0 + y1
            relu = tl.where(summed != summed, summed, tl.maximum(summed, 0.0))
            acc += tl.where(value_mask, relu, 0.0)

        out_offsets = n_offsets[:, None] * channels + c_offsets[None, :]
        tl.store(out_ptr + out_offsets, acc * (1.0 / 49.0), mask=value_mask)


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
    if len(inputs) != 11:
        raise ValueError(f"{REPRO_ID} expects eleven inputs, got {len(inputs)}")

    (
        convolution_42,
        primals_341,
        primals_342,
        primals_343,
        primals_344,
        convolution_43,
        primals_347,
        primals_348,
        primals_349,
        primals_350,
        shape_param,
    ) = inputs

    x0 = _expect_f32_tensor("convolution_42", convolution_42, INPUT_SHAPE, INPUT_STRIDE)
    running_mean0 = _expect_f32_tensor("primals_341", primals_341, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var0 = _expect_f32_tensor("primals_342", primals_342, VECTOR_SHAPE, VECTOR_STRIDE)
    weight0 = _expect_f32_tensor("primals_343", primals_343, VECTOR_SHAPE, VECTOR_STRIDE)
    bias0 = _expect_f32_tensor("primals_344", primals_344, VECTOR_SHAPE, VECTOR_STRIDE)
    x1 = _expect_f32_tensor("convolution_43", convolution_43, INPUT_SHAPE, INPUT_STRIDE)
    running_mean1 = _expect_f32_tensor("primals_347", primals_347, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var1 = _expect_f32_tensor("primals_348", primals_348, VECTOR_SHAPE, VECTOR_STRIDE)
    weight1 = _expect_f32_tensor("primals_349", primals_349, VECTOR_SHAPE, VECTOR_STRIDE)
    bias1 = _expect_f32_tensor("primals_350", primals_350, VECTOR_SHAPE, VECTOR_STRIDE)

    if tuple(shape_param) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape parameter: {shape_param!r}")

    device = x0.device
    tensor_inputs = (
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
    )
    if any(t.device != device for t in tensor_inputs):
        raise ValueError("all tensor inputs must be on the same CUDA device")
    return (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
    )


@oracle_impl(hardware="H100", shapes="(T([128, 1408, 7, 7], f32, stride=(68992, 1, 9856, 1408)), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), T([128, 1408, 7, 7], f32, stride=(68992, 1, 9856, 1408)), T([1408], f32), T([1408], f32), T([1408], f32), T([1408], f32), S([128, 1408]))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat copy_ effects."""
    if triton is None:
        raise RuntimeError("Triton is required for oracle_dual_bn_training_spatial_mean.py")

    (
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
    ) = _validate_inputs(inputs)

    partial = torch.empty_strided(
        (4, STAT_BLOCKS, CHANNELS),
        (STAT_BLOCKS * CHANNELS, CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    output = torch.empty_strided(
        OUTPUT_SHAPE,
        OUTPUT_STRIDE,
        device=x0.device,
        dtype=torch.float32,
    )

    _dual_partial_stats_kernel[(triton.cdiv(CHANNELS, STAT_BLOCK_C), STAT_BLOCKS)](
        x0,
        x1,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        BLOCK_M=STAT_BLOCK_M,
        BLOCK_C=STAT_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _dual_bn_update_relu_spatial_mean_kernel[
        (triton.cdiv(N, EPILOGUE_BLOCK_N), triton.cdiv(CHANNELS, EPILOGUE_BLOCK_C))
    ](
        x0,
        running_mean0,
        running_var0,
        weight0,
        bias0,
        x1,
        running_mean1,
        running_var1,
        weight1,
        bias1,
        partial,
        output,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        stat_blocks=STAT_BLOCKS,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_N=EPILOGUE_BLOCK_N,
        BLOCK_C=EPILOGUE_BLOCK_C,
        BLOCK_P=PARTIAL_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return output, running_mean0, running_var0, running_mean1, running_var1


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
    alias_input_indices = (None, 1, 2, 6, 7)
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
            max_diff = (expected_f32 - actual_f32).abs().max().item()
            ok = torch.allclose(expected_f32, actual_f32, atol=atol, rtol=rtol)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(shape={list(expected.shape)} dtype={expected.dtype} max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(expected, actual)
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
