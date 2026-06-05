"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Visformer training-BatchNorm tail by replaying the tuned full-scope Welford statistics plan, running-stat copy_ epilogue, invstd side output, centered activation store, affine normalization, and 7x7 spatial mean, whereas measured Inductor already emits the same practical three-kernel norm-template plan for this channels-last shape; Inductor cannot materially improve this today through scheduler fusion, scatter-reduce, split-K, algebraic elimination, or recompute fusion because the remaining time is dominated by required residual reads, statistics traffic, centered activation stores, pooled output stores, and launch envelope; the fix is BANDWIDTH_BOUND: record this repro as at-floor unless broader normalization codegen, launch, or bandwidth work moves both paths."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
    from torch._C import _cuda_getCurrentRawStream as get_raw_stream
    from torch._inductor.async_compile import AsyncCompile
except ImportError:
    triton = None
    tl = None
    get_raw_stream = None
    AsyncCompile = None

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
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


CLASSIFICATION = "BANDWIDTH_BOUND"

N = 128
CHANNELS = 768
HEIGHT = 7
WIDTH = 7
HW = HEIGHT * WIDTH
EPS = 1.0e-5
MOMENTUM = 0.1
RUNNING_VAR_CORRECTION = 1.0001594642002871

INPUT_SHAPE = (N, CHANNELS, HEIGHT, WIDTH)
INPUT_STRIDE = (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS)
VECTOR_SHAPE = (CHANNELS,)
VECTOR_STRIDE = (1,)
POOLED_SHAPE = (N, CHANNELS)
POOLED_STRIDE = (CHANNELS, 1)


def get_inputs() -> list[Any]:
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance() -> torch.nn.Module:
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


if triton is not None and AsyncCompile is not None:

    _async_compile = AsyncCompile()

    _stats_partial_kernel = _async_compile.triton("oracle_6b0_stats_partial", '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_6b0_stats_partial', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 3, 'autotune_hints': set(), 'tiling_scores': {'x': 39438336, 'r0_': 0}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_6b0_stats_partial(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 37632
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex % 768
    x1 = xindex // 768
    mean_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    m2_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    weight_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        v0 = tl.load(in_ptr0 + (x0 + 768 * r0_index + 98304 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        v1 = tl.load(in_ptr1 + (x0 + 768 * r0_index + 98304 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        residual = tl.broadcast_to(v0 + v1, [XBLOCK, R0_BLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            residual, mean_acc, m2_acc, weight_acc, r0_offset == 0
        )
        mean_acc = tl.where(r0_mask & xmask, mean_next, mean_acc)
        m2_acc = tl.where(r0_mask & xmask, m2_next, m2_acc)
        weight_acc = tl.where(r0_mask & xmask, weight_next, weight_acc)
    mean, m2, weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    tl.store(out_ptr0 + xindex, mean[:, None], xmask)
    tl.store(out_ptr1 + xindex, m2[:, None], xmask)
    tl.store(out_ptr2 + xindex, weight[:, None], xmask)
''', device_str='cuda')

    _stats_finalize_kernel = _async_compile.triton("oracle_6b0_stats_finalize", '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 64},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr2': '*fp32', 'out_ptr4': '*fp32', 'out_ptr6': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_6b0_stats_finalize', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 2, 'autotune_hints': set(), 'tiling_scores': {'x': 482304, 'r0_': 0}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_6b0_stats_finalize(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr2, out_ptr4, out_ptr6, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 768
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = xindex
    mean_part = tl.load(in_ptr0 + (x0 + 768 * r0_index), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    m2_part = tl.load(in_ptr1 + (x0 + 768 * r0_index), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    weight_part = tl.load(in_ptr2 + (x0 + 768 * r0_index), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    running_var_old = tl.load(in_ptr3 + x0, xmask)
    running_mean_old = tl.load(in_ptr4 + x0, xmask)
    mean_inputs = tl.where(r0_mask & xmask, tl.broadcast_to(mean_part, [XBLOCK, R0_BLOCK]), 0.0)
    m2_inputs = tl.where(r0_mask & xmask, tl.broadcast_to(m2_part, [XBLOCK, R0_BLOCK]), 0.0)
    weight_inputs = tl.where(r0_mask & xmask, tl.broadcast_to(weight_part, [XBLOCK, R0_BLOCK]), 0.0)
    mean, m2, _weight = triton_helpers.welford(mean_inputs, m2_inputs, weight_inputs, 1)
    mean = mean[:, None]
    m2 = m2[:, None]
    var = m2 / 6272.0
    invstd = libdevice.rsqrt(var + 1.0e-5)
    running_var_new = var * 1.0001594642002871 * 0.1 + running_var_old * 0.9
    running_mean_new = mean * 0.1 + running_mean_old * 0.9
    tl.store(out_ptr2 + x0, invstd, xmask)
    tl.store(out_ptr4 + x0, running_var_new, xmask)
    tl.store(out_ptr6 + x0, running_mean_new, xmask)
    tl.store(out_ptr0 + x0, mean, xmask)
''', device_str='cuda')

    _center_pool_kernel = _async_compile.triton("oracle_6b0_center_pool", '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 131072, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_6b0_center_pool', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': set(), 'tiling_scores': {'x': 77869056, 'r0_': 0}, 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}
)
@triton.jit
def oracle_6b0_center_pool(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 98304
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    x0 = xindex % 768
    x1 = xindex // 768
    v0 = tl.load(in_ptr0 + (x0 + 768 * r0_index + 37632 * x1), r0_mask, eviction_policy='evict_first', other=0.0)
    v1 = tl.load(in_ptr1 + (x0 + 768 * r0_index + 37632 * x1), r0_mask, eviction_policy='evict_first', other=0.0)
    mean = tl.load(in_ptr2 + x0, None, eviction_policy='evict_last')
    invstd = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    weight = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
    bias = tl.load(in_ptr5 + x0, None, eviction_policy='evict_last')
    centered = v0 + v1 - mean
    affine = centered * invstd * weight + bias
    affine = tl.broadcast_to(affine, [XBLOCK, R0_BLOCK])
    pooled = tl.sum(tl.where(r0_mask, affine, 0.0), 1)[:, None].to(tl.float32) / 49.0
    tl.store(in_out_ptr0 + xindex, pooled, None)
    tl.store(out_ptr0 + (x0 + 768 * r0_index + 37632 * x1), centered, r0_mask)
''', device_str='cuda')

    _async_compile.wait(globals())
    del _async_compile


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
    if len(inputs) != 7:
        raise ValueError(f"{REPRO_ID} expects seven inputs, got {len(inputs)}")

    (
        add_175,
        convolution_56,
        running_mean,
        running_var,
        weight,
        bias,
        shape_param,
    ) = inputs

    x0 = _expect_f32_tensor("add_175", add_175, INPUT_SHAPE, INPUT_STRIDE)
    x1 = _expect_f32_tensor("convolution_56", convolution_56, INPUT_SHAPE, INPUT_STRIDE)
    running_mean = _expect_f32_tensor("primals_201", running_mean, VECTOR_SHAPE, VECTOR_STRIDE)
    running_var = _expect_f32_tensor("primals_202", running_var, VECTOR_SHAPE, VECTOR_STRIDE)
    weight = _expect_f32_tensor("primals_203", weight, VECTOR_SHAPE, VECTOR_STRIDE)
    bias = _expect_f32_tensor("primals_204", bias, VECTOR_SHAPE, VECTOR_STRIDE)

    if tuple(shape_param) != POOLED_SHAPE:
        raise ValueError(f"unexpected reshape parameter: {shape_param!r}")

    device = x0.device
    if any(t.device != device for t in (x1, running_mean, running_var, weight, bias)):
        raise ValueError("all tensor inputs must be on the same device")
    return x0, x1, running_mean, running_var, weight, bias


def _torch_reference(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    x0, x1, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    add_tensor = x0 + x1
    var, mean = torch.var_mean(add_tensor, dim=(0, 2, 3), correction=0, keepdim=True)
    invstd_4d = torch.rsqrt(var + EPS)
    centered = add_tensor - mean
    affine = centered * invstd_4d * weight[None, :, None, None] + bias[None, :, None, None]
    pooled = affine.mean(dim=(-1, -2), keepdim=True)
    pooled = pooled.as_strided(POOLED_SHAPE + (1, 1), (CHANNELS, 1, CHANNELS, CHANNELS))
    pooled_2d = pooled.reshape(POOLED_SHAPE)
    mean_1d = mean.squeeze((0, 2, 3))
    var_1d = var.squeeze((0, 2, 3))
    invstd_1d = invstd_4d.squeeze((0, 2, 3))
    running_mean.copy_(running_mean * (1.0 - MOMENTUM) + mean_1d * MOMENTUM)
    running_var.copy_(
        running_var * (1.0 - MOMENTUM)
        + var_1d * RUNNING_VAR_CORRECTION * MOMENTUM
    )
    return invstd_1d, pooled_2d, centered, running_mean, running_var


def oracle_forward(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
    """Run the full Repro.forward scope, including running-stat copy_ effects."""
    x0, x1, running_mean, running_var, weight, bias = _validate_inputs(inputs)
    if triton is None or AsyncCompile is None or get_raw_stream is None or not x0.is_cuda:
        return _torch_reference(inputs)

    partial_mean = torch.empty_strided((HW, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float32)
    partial_m2 = torch.empty_strided((HW, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float32)
    partial_weight = torch.empty_strided((HW, CHANNELS), (CHANNELS, 1), device=x0.device, dtype=torch.float32)
    mean = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x0.device, dtype=torch.float32)
    invstd = torch.empty_strided(VECTOR_SHAPE, VECTOR_STRIDE, device=x0.device, dtype=torch.float32)
    pooled = torch.empty_strided(POOLED_SHAPE, POOLED_STRIDE, device=x0.device, dtype=torch.float32)
    centered = torch.empty_strided(INPUT_SHAPE, INPUT_STRIDE, device=x0.device, dtype=torch.float32)

    raw_stream = get_raw_stream(x0.device.index or 0)
    _stats_partial_kernel.run(
        x0,
        x1,
        partial_mean,
        partial_m2,
        partial_weight,
        CHANNELS * HW,
        N,
        stream=raw_stream,
    )
    _stats_finalize_kernel.run(
        partial_mean,
        partial_m2,
        partial_weight,
        running_var,
        running_mean,
        mean,
        invstd,
        running_var,
        running_mean,
        CHANNELS,
        HW,
        stream=raw_stream,
    )
    _center_pool_kernel.run(
        pooled,
        x0,
        x1,
        mean,
        invstd,
        weight,
        bias,
        centered,
        N * CHANNELS,
        HW,
        stream=raw_stream,
    )
    return invstd, pooled, centered, running_mean, running_var


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
    instance: torch.nn.Module,
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
        if any(isinstance(item, torch.Tensor) and item.is_cuda for item in oracle_inputs):
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
    alias_input_indices = (None, None, None, 2, 3)
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
                f"(shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()} max_diff={max_diff:.2e})"
            )
        else:
            ok = torch.equal(expected, actual)
            print(
                f"  output {i}: {'PASS' if ok else 'FAIL'} "
                f"(exact, shape={list(expected.shape)} dtype={expected.dtype} "
                f"stride={expected.stride()})"
            )
        all_pass = all_pass and bool(ok)

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
            floor_status = "not_true_floor" if result["status"] == "BAD_ORACLE" else "true_floor"
            print(f"classification: {CLASSIFICATION}")
            print(f"floor_status: {floor_status}")


if __name__ == "__main__":
    main()
