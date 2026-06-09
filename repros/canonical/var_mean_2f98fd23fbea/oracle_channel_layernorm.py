"""Gap diagnosis (classification: BANDWIDTH_BOUND): this floor-verification oracle computes the complete ConvNeXt channel LayerNorm scope by replaying the tuned chunked Welford reduction, eps=1e-6 rsqrt, affine scale/bias, and final NCHW-layout store as explicit Triton kernels, whereas Inductor already emits the same practical three-kernel plan for the default NCHW shape; Inductor cannot materially improve it today through a local norm-template canonicalization change because the remaining cost is dominated by the required strided channel reduction, affine/output traffic, and launch envelope, and one-kernel/raw-moment alternatives were slower; the fix is BANDWIDTH_BOUND: mark this repro at_floor and revisit only if broader normalization codegen or launch/bandwidth improvements move both paths."""
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


EPS = 1.0e-6
CHUNK_C = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


# --- Oracle kernel(s) ---
if triton is not None:

    _async_compile = AsyncCompile()

    _floor_red0 = _async_compile.triton('oracle_var_mean_2f98_red0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import ReductionHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_var_mean_2f98_red0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 3, 'autotune_hints': set(), 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True}
)
@triton.jit
def oracle_var_mean_2f98_red0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 31360
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex % 49
    x1 = xindex // 49
    mean_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    m2_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    weight_acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        values = tl.load(in_ptr0 + (x0 + 49 * r0_index + 6272 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        values = tl.broadcast_to(values, [XBLOCK, R0_BLOCK])
        mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
            values, mean_acc, m2_acc, weight_acc, r0_offset == 0
        )
        mean_acc = tl.where(r0_mask & xmask, mean_next, mean_acc)
        m2_acc = tl.where(r0_mask & xmask, m2_next, m2_acc)
        weight_acc = tl.where(r0_mask & xmask, weight_next, weight_acc)
    mean, m2, weight = triton_helpers.welford(mean_acc, m2_acc, weight_acc, 1)
    tl.store(out_ptr0 + x3, mean[:, None], xmask)
    tl.store(out_ptr1 + x3, m2[:, None], xmask)
    tl.store(out_ptr2 + x3, weight[:, None], xmask)
''', device_str='cuda')

    _floor_red1 = _async_compile.triton('oracle_var_mean_2f98_red1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.hints import ReductionHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 8},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_var_mean_2f98_red1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 2, 'num_reduction': 2, 'autotune_hints': set(), 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True}
)
@triton.jit
def oracle_var_mean_2f98_red1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 6272
    R0_BLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < 5
    x0 = xindex % 49
    x1 = xindex // 49
    x3 = xindex
    partial_mean = tl.load(in_ptr0 + (x0 + 49 * r0_index + 245 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    partial_m2 = tl.load(in_ptr1 + (x0 + 49 * r0_index + 245 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    partial_weight = tl.load(in_ptr2 + (x0 + 49 * r0_index + 245 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
    partial_mean = tl.where(r0_mask & xmask, tl.broadcast_to(partial_mean, [XBLOCK, R0_BLOCK]), 0)
    partial_m2 = tl.where(r0_mask & xmask, tl.broadcast_to(partial_m2, [XBLOCK, R0_BLOCK]), 0)
    partial_weight = tl.where(r0_mask & xmask, tl.broadcast_to(partial_weight, [XBLOCK, R0_BLOCK]), 0)
    mean, m2, weight = triton_helpers.welford(partial_mean, partial_m2, partial_weight, 1)
    tl.store(out_ptr0 + x3, mean[:, None], xmask)
    tl.store(out_ptr1 + x3, m2[:, None], xmask)
''', device_str='cuda')

    _floor_poi2 = _async_compile.triton('oracle_var_mean_2f98_poi2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.runtime.hints import DeviceProperties

@triton_heuristics.pointwise(
    size_hints={'x': 4194304},
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'oracle_var_mean_2f98_poi2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': set(), 'backend_hash': 'oracle', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False, 'coordinate_descent_tuning': True},
    min_elem_per_thread=0
)
@triton.jit
def oracle_var_mean_2f98_poi2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x3 = xindex
    x0 = xindex % 49
    x2 = xindex // 31360
    x1 = (xindex // 49) % 640
    value = tl.load(in_ptr0 + x3, None)
    mean = tl.load(in_ptr1 + (x0 + 49 * x2), None, eviction_policy='evict_last')
    m2 = tl.load(in_ptr2 + (x0 + 49 * x2), None, eviction_policy='evict_last')
    scale = tl.load(in_ptr3 + x1, None, eviction_policy='evict_last')
    bias = tl.load(in_ptr4 + x1, None, eviction_policy='evict_last')
    invstd = libdevice.rsqrt(m2 / 640.0 + 1.0e-6)
    output = (value - mean) * invstd * scale + bias
    tl.store(out_ptr1 + x3, output, None)
''', device_str='cuda')

    _async_compile.wait(globals())
    del _async_compile

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_S": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_S": 4}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_S": 8}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W", "x_stride_c"],
    )
    @triton.jit
    def _channel_layernorm_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        n = tl.program_id(0)
        spatial_block = tl.program_id(1) * BLOCK_S

        channels = tl.arange(0, BLOCK_C)
        spatial = spatial_block + tl.arange(0, BLOCK_S)
        h = spatial // W
        w = spatial - h * W

        mask = (channels[:, None] < C) & (spatial[None, :] < H * W)
        x_offsets = (
            n * x_stride_n
            + channels[:, None] * x_stride_c
            + h[None, :] * x_stride_h
            + w[None, :] * x_stride_w
        )
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)

        sum_values = tl.sum(values, axis=0)
        sum_squares = tl.sum(values * values, axis=0)
        mean = sum_values / C
        variance = sum_squares / C - mean * mean
        variance = tl.maximum(variance, 0.0)
        centered = values - mean[None, :]
        invstd = tl.rsqrt(variance + eps)

        weight = tl.load(weight_ptr + channels, mask=channels < C, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + channels, mask=channels < C, other=0.0).to(tl.float32)
        output = centered * invstd[None, :] * weight[:, None] + bias[:, None]

        out_offsets = (
            n * out_stride_n
            + channels[:, None] * out_stride_c
            + h[None, :] * out_stride_h
            + w[None, :] * out_stride_w
        )
        tl.store(out_ptr + out_offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_X": 1}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 2}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 4}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 32}, num_warps=8, num_stages=3),
            triton.Config({"BLOCK_X": 64}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W", "x_stride_c"],
    )
    @triton.jit
    def _partial_stats_kernel(
        x_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        spatial_size: tl.constexpr = H * W
        total_partials: tl.constexpr = 128 * NUM_CHUNKS * spatial_size

        partial = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
        r = tl.arange(0, BLOCK_R)
        spatial = partial % spatial_size
        group = partial // spatial_size
        chunk = group % NUM_CHUNKS
        n = group // NUM_CHUNKS
        h = spatial // W
        w = spatial - h * W
        c = chunk[:, None] * BLOCK_R + r[None, :]

        offsets = (
            n[:, None] * x_stride_n
            + c * x_stride_c
            + h[:, None] * x_stride_h
            + w[:, None] * x_stride_w
        )
        mask = (partial[:, None] < total_partials) & (c < C)
        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sums = tl.sum(values, axis=1)
        sums_sq = tl.sum(values * values, axis=1)
        tl.store(partial_sum_ptr + partial, sums, mask=partial < total_partials)
        tl.store(partial_sumsq_ptr + partial, sums_sq, mask=partial < total_partials)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_ROWS": 8}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 16}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 32}, num_warps=1, num_stages=3),
            triton.Config({"BLOCK_ROWS": 64}, num_warps=2, num_stages=3),
            triton.Config({"BLOCK_ROWS": 128}, num_warps=4, num_stages=3),
        ],
        key=["C", "H", "W"],
    )
    @triton.jit
    def _final_stats_kernel(
        partial_sum_ptr,
        partial_sumsq_ptr,
        mean_ptr,
        invstd_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        eps: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        spatial_size: tl.constexpr = H * W
        total_rows: tl.constexpr = 128 * spatial_size

        row = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        chunks = tl.arange(0, BLOCK_CHUNKS)
        n = row // spatial_size
        spatial = row - n * spatial_size
        partial_offsets = (n[:, None] * NUM_CHUNKS + chunks[None, :]) * spatial_size + spatial[:, None]
        mask = (row[:, None] < total_rows) & (chunks[None, :] < NUM_CHUNKS)
        sums = tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_sumsq_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)

        total_sum = tl.sum(sums, axis=1)
        total_sumsq = tl.sum(sums_sq, axis=1)
        mean = total_sum / C
        variance = total_sumsq / C - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)
        tl.store(mean_ptr + row, mean, mask=row < total_rows)
        tl.store(invstd_ptr + row, invstd, mask=row < total_rows)

    @triton.jit
    def _normalize_contiguous_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        spatial_size: tl.constexpr = H * W
        total: tl.constexpr = 128 * C * spatial_size
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        spatial = offsets % spatial_size
        c = (offsets // spatial_size) % C
        n = offsets // (C * spatial_size)
        row = n * spatial_size + spatial

        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + row, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
        output = (values - mean) * invstd * weight + bias
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def _normalize_channels_last_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        mean_ptr,
        invstd_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        spatial_size: tl.constexpr = H * W
        total: tl.constexpr = 128 * C * spatial_size
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < total
        c = offsets % C
        spatial = (offsets // C) % spatial_size
        n = offsets // (C * spatial_size)
        row = n * spatial_size + spatial

        values = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + row, mask=mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + row, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
        output = (values - mean) * invstd * weight + bias
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_X": 8}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 16}, num_warps=4, num_stages=3),
            triton.Config({"BLOCK_X": 32}, num_warps=8, num_stages=3),
        ],
        key=["C", "H", "W", "x_stride_c"],
    )
    @triton.jit
    def _normalize_from_partials_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        partial_sum_ptr,
        partial_sumsq_ptr,
        out_ptr,
        C: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        out_stride_n: tl.constexpr,
        out_stride_c: tl.constexpr,
        out_stride_h: tl.constexpr,
        out_stride_w: tl.constexpr,
        eps: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_X: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        spatial_size: tl.constexpr = H * W
        total_partials: tl.constexpr = 128 * NUM_CHUNKS * spatial_size

        partial = tl.program_id(0) * BLOCK_X + tl.arange(0, BLOCK_X)
        r = tl.arange(0, BLOCK_R)
        chunks = tl.arange(0, BLOCK_CHUNKS)

        spatial = partial % spatial_size
        group = partial // spatial_size
        chunk = group % NUM_CHUNKS
        n = group // NUM_CHUNKS
        h = spatial // W
        w = spatial - h * W

        stat_offsets = (n[:, None] * NUM_CHUNKS + chunks[None, :]) * spatial_size + spatial[:, None]
        stat_mask = (partial[:, None] < total_partials) & (chunks[None, :] < NUM_CHUNKS)
        sums = tl.load(partial_sum_ptr + stat_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        sums_sq = tl.load(partial_sumsq_ptr + stat_offsets, mask=stat_mask, other=0.0).to(tl.float32)
        total_sum = tl.sum(sums, axis=1)
        total_sumsq = tl.sum(sums_sq, axis=1)
        mean = total_sum / C
        variance = total_sumsq / C - mean * mean
        variance = tl.maximum(variance, 0.0)
        invstd = tl.rsqrt(variance + eps)

        c = chunk[:, None] * BLOCK_R + r[None, :]
        mask = (partial[:, None] < total_partials) & (c < C)
        x_offsets = (
            n[:, None] * x_stride_n
            + c * x_stride_c
            + h[:, None] * x_stride_h
            + w[:, None] * x_stride_w
        )
        out_offsets = (
            n[:, None] * out_stride_n
            + c * out_stride_c
            + h[:, None] * out_stride_h
            + w[:, None] * out_stride_w
        )
        values = tl.load(x_ptr + x_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c, mask=c < C, other=0.0).to(tl.float32)
        output = (values - mean[:, None]) * invstd[:, None] * weight + bias
        tl.store(out_ptr + out_offsets, output, mask=mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _validate_inputs(inputs: list[Any] | tuple[Any, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(inputs) != 3:
        raise ValueError(f"{REPRO_ID} expects 3 inputs, got {len(inputs)}")

    x, weight, bias = inputs
    if not all(isinstance(value, torch.Tensor) for value in (x, weight, bias)):
        raise TypeError("all repro inputs must be tensors")
    if not all(value.is_cuda for value in (x, weight, bias)):
        raise RuntimeError("CUDA tensors are required for the Triton oracle")
    if x.ndim != 4:
        raise ValueError(f"input 0 must be NCHW rank-4, got shape={tuple(x.shape)}")

    n, c, h, w = (int(dim) for dim in x.shape)
    if n != 128 or (c, h, w) not in ((640, 7, 7), (320, 14, 14), (160, 28, 28), (80, 56, 56)):
        raise ValueError(f"unexpected ConvNeXt shape {tuple(x.shape)}")
    if tuple(weight.shape) != (c,) or tuple(bias.shape) != (c,):
        raise ValueError(
            f"weight/bias shapes must be ({c},), got {tuple(weight.shape)} and {tuple(bias.shape)}"
        )
    if x.dtype != torch.float32 or weight.dtype != torch.float32 or bias.dtype != torch.float32:
        raise TypeError(f"expected all float32 inputs, got {[value.dtype for value in (x, weight, bias)]}")
    if not (x.is_contiguous() or x.is_contiguous(memory_format=torch.channels_last)):
        raise ValueError(f"unsupported input stride for this oracle: {x.stride()}")
    if not weight.is_contiguous() or not bias.is_contiguous():
        raise ValueError("weight and bias must be contiguous")

    return x, weight, bias


def _select_block_s(x: torch.Tensor) -> int:
    if x.is_contiguous(memory_format=torch.channels_last) and not x.is_contiguous():
        return 1
    c = int(x.shape[1])
    if c >= 512:
        return 4
    if c >= 256:
        return 8
    return 16


def _oracle_default_inductor_floor(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Replay the tuned default-shape chunked Welford schedule without calling torch.compile."""
    if get_raw_stream is None:
        raise RuntimeError("Inductor Triton runtime is required for the floor oracle")

    buf0 = torch.empty_strided((128, 7, 7, 1, 5), (245, 7, 1, 31360, 49), device=x.device, dtype=x.dtype)
    buf1 = torch.empty_strided((128, 7, 7, 1, 5), (245, 7, 1, 31360, 49), device=x.device, dtype=x.dtype)
    buf2 = torch.empty_strided((128, 7, 7, 1, 5), (245, 7, 1, 31360, 49), device=x.device, dtype=x.dtype)
    stream = get_raw_stream(0)
    _floor_red0.run(x, buf0, buf1, buf2, 31360, 128, stream=stream)

    mean = torch.empty_strided((128, 7, 7, 1), (49, 7, 1, 6272), device=x.device, dtype=x.dtype)
    m2 = torch.empty_strided((128, 7, 7, 1), (49, 7, 1, 6272), device=x.device, dtype=x.dtype)
    stream = get_raw_stream(0)
    _floor_red1.run(buf0, buf1, buf2, mean, m2, 6272, 5, stream=stream)

    output = torch.empty_strided((128, 640, 7, 7), (31360, 49, 7, 1), device=x.device, dtype=x.dtype)
    stream = get_raw_stream(0)
    _floor_poi2.run(x, mean, m2, weight, bias, output, 4014080, stream=stream)
    return output


@oracle_impl(hardware="H100", shapes="(T([128, 640, 7, 7], f32), T([640], f32), T([640], f32))")
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
    if triton is None:
        raise RuntimeError("Triton is required for oracle_channel_layernorm.py")

    x, weight, bias = _validate_inputs(inputs)
    n, c, h, w = (int(dim) for dim in x.shape)
    if (n, c, h, w) == (128, 640, 7, 7) and tuple(x.stride()) == (31360, 49, 7, 1):
        return _oracle_default_inductor_floor(x, weight, bias)

    output = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    spatial = h * w
    num_chunks = triton.cdiv(c, CHUNK_C)
    partial_elems = n * num_chunks * spatial
    partial_sum = torch.empty((partial_elems,), device=x.device, dtype=torch.float32)
    partial_sumsq = torch.empty((partial_elems,), device=x.device, dtype=torch.float32)
    mean = torch.empty((n * spatial,), device=x.device, dtype=torch.float32)
    invstd = torch.empty((n * spatial,), device=x.device, dtype=torch.float32)

    partial_grid = lambda meta: (triton.cdiv(partial_elems, meta["BLOCK_X"]),)
    _partial_stats_kernel[partial_grid](
        x,
        partial_sum,
        partial_sumsq,
        C=c,
        H=h,
        W=w,
        x_stride_n=x.stride(0),
        x_stride_c=x.stride(1),
        x_stride_h=x.stride(2),
        x_stride_w=x.stride(3),
        NUM_CHUNKS=num_chunks,
        BLOCK_R=CHUNK_C,
    )
    stats_grid = lambda meta: (triton.cdiv(n * spatial, meta["BLOCK_ROWS"]),)
    _final_stats_kernel[stats_grid](
        partial_sum,
        partial_sumsq,
        mean,
        invstd,
        C=c,
        H=h,
        W=w,
        eps=EPS,
        NUM_CHUNKS=num_chunks,
        BLOCK_CHUNKS=_next_power_of_2(num_chunks),
    )
    if x.is_contiguous():
        _normalize_contiguous_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            weight,
            bias,
            mean,
            invstd,
            output,
            C=c,
            H=h,
            W=w,
            BLOCK=1024,
            num_warps=4,
            num_stages=4,
        )
    else:
        _normalize_channels_last_kernel[(triton.cdiv(x.numel(), 1024),)](
            x,
            weight,
            bias,
            mean,
            invstd,
            output,
            C=c,
            H=h,
            W=w,
            BLOCK=1024,
            num_warps=4,
            num_stages=4,
        )
    return output


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
        ok = check_oracle(
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
