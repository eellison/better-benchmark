"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG dual training-BatchNorm scope, including both population var/mean reductions, four running-stat copy_ side effects, affine branch sum, ReLU, and final 7x7 spatial mean in one channel-specialized Triton kernel while preserving the returned running-stat aliases, whereas Inductor lowers the two BN-training/stat branches and downstream activation/spatial mean through generic normalization and reduction scheduling; Inductor cannot do this today because normalization template canonicalization does not fuse sibling training-BN reductions, mutable running-stat returns, and the shared post-add spatial reduction into one full-scope channel schedule; the fix is SCHEDULER_FUSION: teach the scheduler/norm template to keep paired per-channel BN statistics, running-stat updates, and the post-sum spatial mean in one fused schedule."""
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
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    get_shape_key,
    has_stochastic_ops,
)

N = 128
CHANNELS = 1408
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871
BLOCK_N = 128
BLOCK_HW = 64
SPLIT_K = 8
BLOCK_M = 1024
OUTPUT_BLOCK_N = 8
BLOCK_SPLITS = 8


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_bn_stats_partial_kernel(
        x0_ptr,
        x1_ptr,
        partials_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        split_k: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split = tl.program_id(1)
        split_size = tl.cdiv(elements_per_channel, split_k)
        m_offsets = split * split_size + tl.arange(0, BLOCK_M_)
        mask = (m_offsets < elements_per_channel) & (m_offsets < (split + 1) * split_size)
        n_offsets = m_offsets // hw_size
        hw_offsets = m_offsets - n_offsets * hw_size
        input_offsets = (n_offsets * channels + channel) * hw_size + hw_offsets

        x0 = tl.load(x0_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(x0, axis=0)
        sum1 = tl.sum(x1, axis=0)
        sum_sq0 = tl.sum(x0 * x0, axis=0)
        sum_sq1 = tl.sum(x1 * x1, axis=0)

        base = split * channels + channel
        plane_stride = split_k * channels
        tl.store(partials_ptr + base, sum0)
        tl.store(partials_ptr + plane_stride + base, sum_sq0)
        tl.store(partials_ptr + 2 * plane_stride + base, sum1)
        tl.store(partials_ptr + 3 * plane_stride + base, sum_sq1)

    @triton.jit
    def _dual_bn_finalize_kernel(
        partials_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        stats_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        split_k: tl.constexpr,
        BLOCK_SPLITS_: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split_offsets = tl.arange(0, BLOCK_SPLITS_)
        mask = split_offsets < split_k
        partial_offsets = split_offsets * channels + channel
        plane_stride = split_k * channels
        sum0 = tl.sum(tl.load(partials_ptr + partial_offsets, mask=mask, other=0.0), axis=0)
        sum_sq0 = tl.sum(tl.load(partials_ptr + plane_stride + partial_offsets, mask=mask, other=0.0), axis=0)
        sum1 = tl.sum(tl.load(partials_ptr + 2 * plane_stride + partial_offsets, mask=mask, other=0.0), axis=0)
        sum_sq1 = tl.sum(tl.load(partials_ptr + 3 * plane_stride + partial_offsets, mask=mask, other=0.0), axis=0)

        mean0 = sum0 / elements_per_channel
        mean1 = sum1 / elements_per_channel
        var0 = tl.maximum(sum_sq0 / elements_per_channel - mean0 * mean0, 0.0)
        var1 = tl.maximum(sum_sq1 / elements_per_channel - mean1 * mean1, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)

        old_running_mean0 = tl.load(running_mean0_ptr + channel).to(tl.float32)
        old_running_var0 = tl.load(running_var0_ptr + channel).to(tl.float32)
        old_running_mean1 = tl.load(running_mean1_ptr + channel).to(tl.float32)
        old_running_var1 = tl.load(running_var1_ptr + channel).to(tl.float32)
        tl.store(running_mean0_ptr + channel, old_running_mean0 * (1.0 - momentum) + mean0 * momentum)
        tl.store(
            running_var0_ptr + channel,
            old_running_var0 * (1.0 - momentum) + var0 * running_var_correction * momentum,
        )
        tl.store(running_mean1_ptr + channel, old_running_mean1 * (1.0 - momentum) + mean1 * momentum)
        tl.store(
            running_var1_ptr + channel,
            old_running_var1 * (1.0 - momentum) + var1 * running_var_correction * momentum,
        )

        tl.store(stats_ptr + channel, mean0)
        tl.store(stats_ptr + channels + channel, invstd0)
        tl.store(stats_ptr + 2 * channels + channel, mean1)
        tl.store(stats_ptr + 3 * channels + channel, invstd1)

    @triton.jit
    def _dual_bn_relu_spatial_mean_output_kernel(
        x0_ptr,
        weight0_ptr,
        bias0_ptr,
        x1_ptr,
        weight1_ptr,
        bias1_ptr,
        stats_ptr,
        out_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_N_: tl.constexpr,
        BLOCK_HW_: tl.constexpr,
    ):
        n_block = tl.program_id(0)
        channel = tl.program_id(1)
        n_offsets = n_block * BLOCK_N_ + tl.arange(0, BLOCK_N_)
        hw_offsets = tl.arange(0, BLOCK_HW_)
        hw_mask = hw_offsets < hw_size
        offsets = (n_offsets[:, None] * channels + channel) * hw_size + hw_offsets[None, :]

        x0 = tl.load(x0_ptr + offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=hw_mask[None, :], other=0.0).to(tl.float32)
        mean0 = tl.load(stats_ptr + channel).to(tl.float32)
        invstd0 = tl.load(stats_ptr + channels + channel).to(tl.float32)
        mean1 = tl.load(stats_ptr + 2 * channels + channel).to(tl.float32)
        invstd1 = tl.load(stats_ptr + 3 * channels + channel).to(tl.float32)
        weight0 = tl.load(weight0_ptr + channel).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel).to(tl.float32)

        y0 = (x0 - mean0) * invstd0 * weight0 + bias0
        y1 = (x1 - mean1) * invstd1 * weight1 + bias1
        relu = tl.maximum(y0 + y1, 0.0)
        relu = tl.where(hw_mask[None, :], relu, 0.0)
        pooled = tl.sum(relu, axis=1) / hw_size
        tl.store(out_ptr + n_offsets * channels + channel, pooled)


def _expect_tensor(
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
        arg340_1,
        arg341_1,
        arg342_1,
        arg343_1,
        convolution_43,
        arg346_1,
        arg347_1,
        arg348_1,
        arg349_1,
        shape_param,
    ) = inputs

    nchw_shape = (N, CHANNELS, HEIGHT, WIDTH)
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    vector_shape = (CHANNELS,)
    vector_stride = (1,)
    x0 = _expect_tensor("convolution_42", convolution_42, nchw_shape, nchw_stride)
    running_mean0 = _expect_tensor("arg340_1", arg340_1, vector_shape, vector_stride)
    running_var0 = _expect_tensor("arg341_1", arg341_1, vector_shape, vector_stride)
    weight0 = _expect_tensor("arg342_1", arg342_1, vector_shape, vector_stride)
    bias0 = _expect_tensor("arg343_1", arg343_1, vector_shape, vector_stride)
    x1 = _expect_tensor("convolution_43", convolution_43, nchw_shape, nchw_stride)
    running_mean1 = _expect_tensor("arg346_1", arg346_1, vector_shape, vector_stride)
    running_var1 = _expect_tensor("arg347_1", arg347_1, vector_shape, vector_stride)
    weight1 = _expect_tensor("arg348_1", arg348_1, vector_shape, vector_stride)
    bias1 = _expect_tensor("arg349_1", arg349_1, vector_shape, vector_stride)

    if tuple(shape_param) != (N, CHANNELS):
        raise ValueError(f"unexpected view shape parameter: {shape_param!r}")

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


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope.

    The four running-stat tensors are mutated in place and returned, matching
    the aliasing behavior of the captured aten.copy_ nodes.
    """
    if triton is None:
        raise RuntimeError("Triton is required for oracle_repvgg_dual_bn_relu_spatial_mean.py")

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
    view_default = torch.empty_strided(
        (N, CHANNELS),
        (CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    partials = torch.empty((4, SPLIT_K, CHANNELS), device=x0.device, dtype=torch.float32)
    stats = torch.empty((4, CHANNELS), device=x0.device, dtype=torch.float32)

    _dual_bn_stats_partial_kernel[(CHANNELS, SPLIT_K)](
        x0,
        x1,
        partials,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        split_k=SPLIT_K,
        BLOCK_M_=BLOCK_M,
        num_warps=8,
        num_stages=3,
    )
    _dual_bn_finalize_kernel[(CHANNELS,)](
        partials,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        stats,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        split_k=SPLIT_K,
        BLOCK_SPLITS_=BLOCK_SPLITS,
        num_warps=1,
        num_stages=3,
    )
    _dual_bn_relu_spatial_mean_output_kernel[(triton.cdiv(N, OUTPUT_BLOCK_N), CHANNELS)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        stats,
        view_default,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_N_=OUTPUT_BLOCK_N,
        BLOCK_HW_=BLOCK_HW,
        num_warps=4,
        num_stages=3,
    )
    return view_default, running_mean0, running_var0, running_mean1, running_var1


def _clone_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, ...]:
    cloned: list[Any] = []
    for item in inputs:
        cloned.append(item.clone() if isinstance(item, torch.Tensor) else item)
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
            print(f"  output {i}: SCOPE_MISMATCH shape oracle={list(actual.shape)} eager={list(expected.shape)}")
            all_pass = False
            continue
        if expected.dtype != actual.dtype:
            print(f"  output {i}: SCOPE_MISMATCH dtype oracle={actual.dtype} eager={expected.dtype}")
            all_pass = False
            continue
        if expected.stride() != actual.stride():
            print(f"  output {i}: SCOPE_MISMATCH stride oracle={actual.stride()} eager={expected.stride()}")
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
            max_diff = (expected.float() - actual.float()).abs().max().item()
            ok = torch.allclose(expected.float(), actual.float(), atol=atol, rtol=rtol)
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
