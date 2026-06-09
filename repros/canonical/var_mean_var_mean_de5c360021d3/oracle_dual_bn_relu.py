"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete RepVGG dual training-BatchNorm block with paired per-channel population var_mean reductions, four in-place running-stat copy_ updates, both affine normalized branches, their sum, ReLU, and all five side outputs using shared shape-specialized Triton statistics and epilogue kernels, whereas Inductor lowers the sibling BN-training branches through generic norm-template schedules before the post-add activation; Inductor cannot do this today because norm-template canonicalization does not co-schedule paired mutable BN reductions and the shared downstream affine/add/ReLU consumer as one full-scope plan; the fix is SCHEDULER_FUSION: teach the normalization scheduler to group sibling BN-training templates, preserve running-stat aliases, and sink the common epilogue into the fused schedule."""
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

N = 128
CHANNELS = 384
HEIGHT = 14
WIDTH = 14
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0000398612827361

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, HW, WIDTH, 1)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
MEAN_SHAPE = (1, CHANNELS, 1, 1)
MEAN_STRIDE = (CHANNELS, 1, 1, 1)
OUTPUT_STRIDE = INPUT_STRIDE

SPLIT_K = 8
STAT_BLOCK_M = 4096
FINAL_BLOCK_SPLITS = 8
EPILOGUE_BLOCK_M = 256


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None:

    @triton.jit
    def _dual_bn_partial_stats_kernel(
        x0_ptr,
        x1_ptr,
        partial_ptr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        elements_per_channel: tl.constexpr,
        split_k: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split = tl.program_id(1)
        split_size = tl.cdiv(elements_per_channel, split_k)
        m_offsets = split * split_size + tl.arange(0, BLOCK_M)
        mask = (m_offsets < elements_per_channel) & (
            m_offsets < (split + 1) * split_size
        )
        n_offsets = m_offsets // hw_size
        hw_offsets = m_offsets - n_offsets * hw_size
        input_offsets = (n_offsets * channels + channel) * hw_size + hw_offsets

        x0 = tl.load(x0_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        sum0 = tl.sum(x0, axis=0)
        sum1 = tl.sum(x1, axis=0)
        sumsq0 = tl.sum(x0 * x0, axis=0)
        sumsq1 = tl.sum(x1 * x1, axis=0)

        base = split * channels + channel
        plane_stride = split_k * channels
        tl.store(partial_ptr + base, sum0)
        tl.store(partial_ptr + plane_stride + base, sumsq0)
        tl.store(partial_ptr + 2 * plane_stride + base, sum1)
        tl.store(partial_ptr + 3 * plane_stride + base, sumsq1)

    @triton.jit
    def _dual_bn_finalize_kernel(
        partial_ptr,
        running_mean0_ptr,
        running_var0_ptr,
        running_mean1_ptr,
        running_var1_ptr,
        invstd0_ptr,
        invstd1_ptr,
        mean0_ptr,
        mean1_ptr,
        channels: tl.constexpr,
        elements_per_channel: tl.constexpr,
        split_k: tl.constexpr,
        eps: tl.constexpr,
        momentum: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_SPLITS: tl.constexpr,
    ):
        channel = tl.program_id(0)
        split_offsets = tl.arange(0, BLOCK_SPLITS)
        mask = split_offsets < split_k
        partial_offsets = split_offsets * channels + channel
        plane_stride = split_k * channels

        sum0 = tl.sum(tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0), axis=0)
        sumsq0 = tl.sum(
            tl.load(partial_ptr + plane_stride + partial_offsets, mask=mask, other=0.0),
            axis=0,
        )
        sum1 = tl.sum(
            tl.load(partial_ptr + 2 * plane_stride + partial_offsets, mask=mask, other=0.0),
            axis=0,
        )
        sumsq1 = tl.sum(
            tl.load(partial_ptr + 3 * plane_stride + partial_offsets, mask=mask, other=0.0),
            axis=0,
        )

        inv_count = 1.0 / elements_per_channel
        mean0 = sum0 * inv_count
        mean1 = sum1 * inv_count
        var0 = tl.maximum(sumsq0 * inv_count - mean0 * mean0, 0.0)
        var1 = tl.maximum(sumsq1 * inv_count - mean1 * mean1, 0.0)
        invstd0 = tl.rsqrt(var0 + eps)
        invstd1 = tl.rsqrt(var1 + eps)

        old_mean0 = tl.load(running_mean0_ptr + channel).to(tl.float32)
        old_var0 = tl.load(running_var0_ptr + channel).to(tl.float32)
        old_mean1 = tl.load(running_mean1_ptr + channel).to(tl.float32)
        old_var1 = tl.load(running_var1_ptr + channel).to(tl.float32)
        tl.store(running_mean0_ptr + channel, old_mean0 * (1.0 - momentum) + mean0 * momentum)
        tl.store(
            running_var0_ptr + channel,
            old_var0 * (1.0 - momentum) + var0 * running_var_correction * momentum,
        )
        tl.store(running_mean1_ptr + channel, old_mean1 * (1.0 - momentum) + mean1 * momentum)
        tl.store(
            running_var1_ptr + channel,
            old_var1 * (1.0 - momentum) + var1 * running_var_correction * momentum,
        )

        tl.store(invstd0_ptr + channel, invstd0)
        tl.store(invstd1_ptr + channel, invstd1)
        tl.store(mean0_ptr + channel, mean0)
        tl.store(mean1_ptr + channel, mean1)

    @triton.jit
    def _dual_bn_affine_relu_kernel(
        x0_ptr,
        weight0_ptr,
        bias0_ptr,
        x1_ptr,
        weight1_ptr,
        bias1_ptr,
        invstd0_ptr,
        invstd1_ptr,
        mean0_ptr,
        mean1_ptr,
        out_ptr,
        total_elements: tl.constexpr,
        channels: tl.constexpr,
        hw_size: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask = offsets < total_elements
        channel = (offsets // hw_size) % channels

        x0 = tl.load(x0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean0 = tl.load(mean0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        mean1 = tl.load(mean1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd0 = tl.load(invstd0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight0 = tl.load(weight0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias0 = tl.load(bias0_ptr + channel, mask=mask, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + channel, mask=mask, other=0.0).to(tl.float32)

        y0 = (x0 - mean0) * invstd0 * weight0 + bias0
        y1 = (x1 - mean1) * invstd1 * weight1 + bias1
        summed = y0 + y1
        relu = tl.where(summed != summed, summed, tl.maximum(summed, 0.0))
        tl.store(out_ptr + offsets, relu, mask=mask)


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
    if len(inputs) != 10:
        raise ValueError(f"{REPRO_ID} expects ten inputs, got {len(inputs)}")

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
    ) = inputs

    x0 = _expect_f32_tensor("convolution_14", x0, INPUT_SHAPE, INPUT_STRIDE)
    running_mean0 = _expect_f32_tensor("arg107_1", running_mean0, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var0 = _expect_f32_tensor("arg108_1", running_var0, VECTOR_SHAPE, VECTOR_STRIDE)
    weight0 = _expect_f32_tensor("arg109_1", weight0, VECTOR_SHAPE, VECTOR_STRIDE)
    bias0 = _expect_f32_tensor("arg110_1", bias0, VECTOR_SHAPE, VECTOR_STRIDE)
    x1 = _expect_f32_tensor("convolution_15", x1, INPUT_SHAPE, INPUT_STRIDE)
    running_mean1 = _expect_f32_tensor("arg113_1", running_mean1, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var1 = _expect_f32_tensor("arg114_1", running_var1, VECTOR_SHAPE, VECTOR_STRIDE)
    weight1 = _expect_f32_tensor("arg115_1", weight1, VECTOR_SHAPE, VECTOR_STRIDE)
    bias1 = _expect_f32_tensor("arg116_1", bias1, VECTOR_SHAPE, VECTOR_STRIDE)

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


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
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
    var0, mean0 = torch.ops.aten.var_mean.correction(
        x0, [0, 2, 3], correction=0, keepdim=True
    )
    var1, mean1 = torch.ops.aten.var_mean.correction(
        x1, [0, 2, 3], correction=0, keepdim=True
    )
    invstd0_4d = torch.rsqrt(var0 + EPS)
    invstd1_4d = torch.rsqrt(var1 + EPS)
    y0 = (x0 - mean0) * invstd0_4d * weight0[:, None, None] + bias0[:, None, None]
    y1 = (x1 - mean1) * invstd1_4d * weight1[:, None, None] + bias1[:, None, None]
    out = torch.relu(y0 + y1)
    mean0_1d = mean0.squeeze((0, 2, 3))
    mean1_1d = mean1.squeeze((0, 2, 3))
    var0_1d = var0.squeeze((0, 2, 3))
    var1_1d = var1.squeeze((0, 2, 3))
    running_mean0.copy_(running_mean0 * (1.0 - MOMENTUM) + mean0_1d * MOMENTUM)
    running_var0.copy_(
        running_var0 * (1.0 - MOMENTUM)
        + var0_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    running_mean1.copy_(running_mean1 * (1.0 - MOMENTUM) + mean1_1d * MOMENTUM)
    running_var1.copy_(
        running_var1 * (1.0 - MOMENTUM)
        + var1_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return (
        invstd0_4d.squeeze((0, 2, 3)),
        invstd1_4d.squeeze((0, 2, 3)),
        out,
        mean1_1d[None, :, None, None],
        mean0_1d[None, :, None, None],
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )


@oracle_impl(hardware="H100", shapes="(T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32), T([128, 384, 14, 14], f32), T([384], f32), T([384], f32), T([384], f32), T([384], f32))")
def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat copy_ effects."""
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
    if triton is None:
        return _torch_reference(inputs)

    partial = torch.empty_strided(
        (4, SPLIT_K, CHANNELS),
        (SPLIT_K * CHANNELS, CHANNELS, 1),
        device=x0.device,
        dtype=torch.float32,
    )
    invstd0 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x0.device, dtype=torch.float32)
    invstd1 = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x0.device, dtype=torch.float32)
    mean0 = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x0.device, dtype=torch.float32)
    mean1 = torch.empty_strided(MEAN_SHAPE, MEAN_STRIDE, device=x0.device, dtype=torch.float32)
    out = torch.empty_strided(INPUT_SHAPE, OUTPUT_STRIDE, device=x0.device, dtype=torch.float32)

    _dual_bn_partial_stats_kernel[(CHANNELS, SPLIT_K)](
        x0,
        x1,
        partial,
        channels=CHANNELS,
        hw_size=HW,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        split_k=SPLIT_K,
        BLOCK_M=STAT_BLOCK_M,
        num_warps=8,
        num_stages=3,
    )
    _dual_bn_finalize_kernel[(CHANNELS,)](
        partial,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
        invstd0,
        invstd1,
        mean0,
        mean1,
        channels=CHANNELS,
        elements_per_channel=ELEMENTS_PER_CHANNEL,
        split_k=SPLIT_K,
        eps=EPS,
        momentum=MOMENTUM,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_SPLITS=FINAL_BLOCK_SPLITS,
        num_warps=1,
        num_stages=3,
    )
    _dual_bn_affine_relu_kernel[(triton.cdiv(TOTAL_ELEMENTS, EPILOGUE_BLOCK_M),)](
        x0,
        weight0,
        bias0,
        x1,
        weight1,
        bias1,
        invstd0,
        invstd1,
        mean0,
        mean1,
        out,
        total_elements=TOTAL_ELEMENTS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_M=EPILOGUE_BLOCK_M,
        num_warps=4,
        num_stages=3,
    )
    return (
        invstd0,
        invstd1,
        out,
        mean1,
        mean0,
        running_mean0,
        running_var0,
        running_mean1,
        running_var1,
    )


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
    alias_input_indices = (None, None, None, None, None, 1, 2, 6, 7)
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
