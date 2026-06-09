"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete training-BatchNorm residual-ReLU scope, including Welford population `var_mean` over `[1024,8,8]` per channel, `libdevice.rsqrt(var + 1e-5)`, returned squeezed invstd and unsqueezed mean views, affine multiply/add, residual add, ReLU, and both in-place running-stat `copy_` updates, by splitting the long per-channel statistics reduction into exact Welford partials before a finalizer and coalesced activation epilogue, whereas Inductor already measures at the same floor for this full scope once the oracle preserves the generated Welford variance formulation; Inductor cannot materially improve this local repro with a cooperative split-K template because the exact statistics reduction, running-stat mutations, affine/residual reads, ReLU, and output traffic dominate; the fix is BANDWIDTH_BOUND: record this as an at-floor BN-training case unless broader normalization-template or memory-traffic work moves both implementations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._inductor.runtime import triton_helpers
    from torch._inductor.runtime.triton_helpers import libdevice
except ImportError:
    triton = None
    tl = None
    triton_helpers = None
    libdevice = None

REPRO_DIR = Path(__file__).resolve().parent
REPRO_ID = REPRO_DIR.name
REPRO_PATH = REPRO_DIR / "repro.py"

from oracle_harness import (
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N = 1024
CHANNELS = 256
HEIGHT = 8
WIDTH = 8
HW = HEIGHT * WIDTH
ELEMENTS_PER_CHANNEL = N * HW
TOTAL_ELEMENTS = N * CHANNELS * HW
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_DECAY = 0.9
RUNNING_VAR_CORRECTION = 1.0000152590218967
STAT_BLOCK = 1024
STAT_BLOCKS = 64
FINAL_BLOCK = 64
POINTWISE_BLOCK = 256
CLASSIFICATION = "BANDWIDTH_BOUND"


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
        offsets = block_id * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
        mask = offsets < elements_per_channel
        n_idx = offsets // hw_size
        hw_idx = offsets - n_idx * hw_size
        x_offsets = (n_idx * channels + channel) * hw_size + hw_idx

        vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        reduce_mask = mask
        mean_acc = tl.zeros([1, BLOCK_M], tl.float32)
        m2_acc = tl.zeros([1, BLOCK_M], tl.float32)
        weight_acc = tl.zeros([1, BLOCK_M], tl.float32)
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            vals,
            mean_acc,
            m2_acc,
            weight_acc,
            True,
        )
        mean_acc = tl.where(reduce_mask, mean_next, mean_acc)
        m2_acc = tl.where(reduce_mask, m2_next, m2_acc)
        weight_acc = tl.where(reduce_mask, weight_next, weight_acc)
        part_mean, part_m2, part_weight = triton_helpers.welford(
            mean_acc,
            m2_acc,
            weight_acc,
            1,
        )
        out_offset = channel * stat_blocks + block_id
        scalar = tl.arange(0, 1)
        tl.store(partial_ptr + out_offset + scalar, part_mean)
        tl.store(partial_ptr + channels * stat_blocks + out_offset + scalar, part_m2)
        tl.store(partial_ptr + 2 * channels * stat_blocks + out_offset + scalar, part_weight)

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
        decay: tl.constexpr,
        running_var_correction: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        channel = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_P)[None, :]
        mask = offsets < stat_blocks
        base = channel * stat_blocks + offsets

        means = tl.load(partial_ptr + base, mask=mask, other=0.0).to(tl.float32)
        m2s = tl.load(partial_ptr + channels * stat_blocks + base, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(partial_ptr + 2 * channels * stat_blocks + base, mask=mask, other=0.0).to(tl.float32)
        mean_inputs = tl.where(mask, means, 0.0)
        m2_inputs = tl.where(mask, m2s, 0.0)
        weight_inputs = tl.where(mask, weights, 0.0)
        mean, m2, _weight = triton_helpers.welford(
            mean_inputs,
            m2_inputs,
            weight_inputs,
            1,
        )
        var = m2 / elements_per_channel
        invstd = libdevice.rsqrt(var + eps)

        scalar = tl.arange(0, 1)
        old_mean = tl.load(running_mean_ptr + channel + scalar).to(tl.float32)
        old_var = tl.load(running_var_ptr + channel + scalar).to(tl.float32)
        new_mean = mean * momentum + old_mean * decay
        corrected_var = var * running_var_correction
        new_var = corrected_var * momentum + old_var * decay
        tl.store(running_mean_ptr + channel + scalar, new_mean)
        tl.store(running_var_ptr + channel + scalar, new_var)
        tl.store(invstd_out_ptr + channel + scalar, invstd)
        tl.store(mean_out_ptr + channel + scalar, mean)

    @triton.jit
    def _affine_residual_relu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        residual_ptr,
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
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        normalized = (x - mean) * invstd
        affine = normalized * weight
        biased = affine + bias
        with_residual = biased + residual
        relu = tl.where(with_residual != with_residual, with_residual, tl.maximum(with_residual, 0.0))
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
    if tuple(value.stride()) != stride:
        raise ValueError(f"{name} has stride {tuple(value.stride())}, expected {stride}")
    return value


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    if len(inputs) != 6:
        raise ValueError(f"{REPRO_ID} expects 6 inputs, got {len(inputs)}")

    convolution_15, arg93_1, arg94_1, arg95_1, arg96_1, relu_10 = inputs
    nchw_stride = (CHANNELS * HW, HW, WIDTH, 1)
    x = _expect_f32_tensor("convolution_15", convolution_15, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)
    running_mean = _expect_f32_tensor("arg93_1", arg93_1, (CHANNELS,), (1,))
    running_var = _expect_f32_tensor("arg94_1", arg94_1, (CHANNELS,), (1,))
    weight = _expect_f32_tensor("arg95_1", arg95_1, (CHANNELS,), (1,))
    bias = _expect_f32_tensor("arg96_1", arg96_1, (CHANNELS,), (1,))
    residual = _expect_f32_tensor("relu_10", relu_10, (N, CHANNELS, HEIGHT, WIDTH), nchw_stride)

    device = x.device
    if any(t.device != device for t in (running_mean, running_var, weight, bias, residual)):
        raise ValueError("all tensor inputs must be on the same device")
    return x, running_mean, running_var, weight, bias, residual


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    var, mean = torch.ops.aten.var_mean.correction(x, [0, 2, 3], correction=0, keepdim=True)
    invstd_4d = torch.ops.aten.rsqrt.default(torch.ops.aten.add.Tensor(var, EPS))
    normalized = torch.ops.aten.mul.Tensor(torch.ops.aten.sub.Tensor(x, mean), invstd_4d)
    mean_1d = torch.ops.aten.squeeze.dims(mean, [0, 2, 3])
    invstd_1d = torch.ops.aten.squeeze.dims(invstd_4d, [0, 2, 3])
    new_running_mean = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(mean_1d, MOMENTUM),
        torch.ops.aten.mul.Tensor(running_mean, RUNNING_VAR_DECAY),
    )
    var_1d = torch.ops.aten.squeeze.dims(var, [0, 2, 3])
    corrected_var = torch.ops.aten.mul.Tensor(var_1d, RUNNING_VAR_CORRECTION)
    new_running_var = torch.ops.aten.add.Tensor(
        torch.ops.aten.mul.Tensor(corrected_var, MOMENTUM),
        torch.ops.aten.mul.Tensor(running_var, RUNNING_VAR_DECAY),
    )
    scaled = torch.ops.aten.mul.Tensor(normalized, weight[:, None, None])
    biased = torch.ops.aten.add.Tensor(scaled, bias[:, None, None])
    with_residual = torch.ops.aten.add.Tensor(biased, residual)
    relu = torch.ops.aten.relu.default(with_residual)
    mean_out = mean_1d[None, :, None, None]
    running_mean.copy_(new_running_mean)
    running_var.copy_(new_running_var)
    return invstd_1d, relu, mean_out, running_mean, running_var


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full captured computation and return all five repro outputs."""
    x, running_mean, running_var, weight, bias, residual = _validate_inputs(inputs)
    if triton is None or not x.is_cuda:
        return _torch_reference(inputs)

    partial = torch.empty_strided(
        (3, CHANNELS, STAT_BLOCKS),
        (CHANNELS * STAT_BLOCKS, STAT_BLOCKS, 1),
        device=x.device,
        dtype=torch.float32,
    )
    invstd = torch.empty_strided((CHANNELS,), (1,), device=x.device, dtype=torch.float32)
    relu = torch.empty_strided(
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
        decay=RUNNING_VAR_DECAY,
        running_var_correction=RUNNING_VAR_CORRECTION,
        BLOCK_P=FINAL_BLOCK,
        num_warps=2,
        num_stages=3,
    )
    _affine_residual_relu_kernel[(triton.cdiv(TOTAL_ELEMENTS, POINTWISE_BLOCK),)](
        x,
        weight,
        bias,
        residual,
        invstd,
        mean,
        relu,
        total_elements=TOTAL_ELEMENTS,
        channels=CHANNELS,
        hw_size=HW,
        BLOCK_M=POINTWISE_BLOCK,
        num_warps=4,
        num_stages=3,
    )
    return invstd, relu, mean, running_mean, running_var


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true", help="Benchmark oracle vs torch.compile")
    parser.add_argument("--rtol", type=float, default=1e-2, help="Relative tolerance for correctness check")
    parser.add_argument("--atol", type=float, default=1e-2, help="Absolute tolerance for correctness check")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations for benchmark")
    parser.add_argument("--rep", type=int, default=200, help="Repetitions for benchmark")
    parser.add_argument("--no-skip-stochastic", action="store_true", help="Accepted for template CLI compatibility")
    parser.add_argument("--all-shapes", action="store_true", help="Benchmark across all shapes from shapes.txt")
    parser.add_argument("--show-hw", action="store_true", help="Print GPU hardware info and exit")
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
        ok = _run_check(instance, inputs, atol=args.atol, rtol=args.rtol)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)

    if args.bench:
        print(f"Benchmarking {REPRO_ID}...")
        if args.all_shapes:
            results = bench_oracle_all_shapes(
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")
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
                print(f"WARNING: oracle is slower than compile (ratio={result['ratio']:.3f}x)")
            floor_status = "true_floor" if result["status"] == "GOOD" else (
                "at_floor" if result["status"] == "AT_FLOOR" else "not_true_floor"
            )
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
