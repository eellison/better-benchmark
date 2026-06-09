"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete VoVNet fixed-center maxpool-backward scatter, ReLU-gated BN-backward channel reductions, full input-gradient tensor, and weight-gradient vector with Triton reductions over the generated offset domain, whereas Inductor currently materializes the maxpool scatter_add result and then lowers the BN mask, two channel sums, and final tensor epilogue as separate generic kernels; Inductor cannot do this today because scheduler/codegen does not represent maxpool-backward offsets as a structured scatter-reduce producer that can feed both channel reductions and the dependent BN-backward epilogue without materializing the 14x14 scatter tensor; the fix is SCATTER_REDUCE: add a maxpool-backward scatter-reduce lowering that maps fixed center offsets directly into the BN reduction/output consumers and emits the complete return tuple."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None



from oracle_harness import (
    oracle_impl,
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_ab550e4e948e"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "torchbench_timm_vovnet_train_001_9ab7e113"

N = 32
C = 768
POOL_H = 7
POOL_W = 7
H = 14
W = 14
HW = H * W
N_HW = N * HW
REDUCTION_SCALE = 1.0 / N_HW
BLOCK_M = 512
BLOCK_C = 4
LINEAR_BLOCK = 512
BLOCK_TILES = 16
NUM_M_TILES = triton.cdiv(N_HW, BLOCK_M) if triton is not None else math.ceil(N_HW / BLOCK_M)


def make_inputs(device: torch.device) -> tuple[object, ...]:
    module = _load_repro_module()
    inputs = module.make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


if triton is not None:

    @triton.jit
    def _partial_bn_sums_kernel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial0_ptr,
        partial1_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        c_mask = c_offsets < C_
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets * weight_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)

        center_active = active & pool_site[:, None]
        source_offsets18 = (
            n_idx[:, None] * getitem18_stride_n
            + c_offsets[None, :] * getitem18_stride_c
            + pool_h[:, None] * getitem18_stride_h
            + pool_w[:, None] * getitem18_stride_w
        )
        source_offsets33 = (
            n_idx[:, None] * getitem33_stride_n
            + c_offsets[None, :] * getitem33_stride_c
            + pool_h[:, None] * getitem33_stride_h
            + pool_w[:, None] * getitem33_stride_w
        )
        scatter = (
            tl.load(getitem18_ptr + source_offsets18, mask=center_active, other=0.0).to(tl.float32)
            + tl.load(getitem33_ptr + source_offsets33, mask=center_active, other=0.0).to(tl.float32)
        )
        grad = tl.where(active, tl.where(gate, scatter, full), 0.0)

        sum_grad = tl.sum(grad, axis=0)
        sum_centered = tl.sum(grad * tl.where(active, centered, 0.0), axis=0)
        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial0_ptr + partial_offsets, sum_grad, mask=c_mask)
        tl.store(partial1_ptr + partial_offsets, sum_centered, mask=c_mask)

    @triton.jit
    def _finalize_bn_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        NUM_M_TILES_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        active = (c_offsets[None, :] < C_) & (tile_offsets[:, None] < NUM_M_TILES_)
        partial_offsets = tile_offsets[:, None] * C_ + c_offsets[None, :]

        sum0_vals = tl.load(partial0_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum1_vals = tl.load(partial1_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)

        mask = c_offsets < C_
        tl.store(sum0_ptr + c_offsets, sum0, mask=mask)
        tl.store(sum1_ptr + c_offsets, sum1, mask=mask)
        tl.store(out1_ptr + c_offsets, sum1 * invstd, mask=mask)

    @triton.jit
    def _bn_input_grad_kernel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        c_mask = c_offsets < C_
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets * weight_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets * bias_stride_c, mask=c_mask, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + h_idx[:, None] * x_stride_h
            + w_idx[:, None] * x_stride_w
        )
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)

        pool_active = active & pool_site[:, None]
        pool_offsets = (
            n_idx[:, None] * offsets_stride_n
            + c_offsets[None, :] * offsets_stride_c
            + pool_h[:, None] * offsets_stride_h
            + pool_w[:, None] * offsets_stride_w
        )
        offset_value = tl.load(offsets_ptr + pool_offsets, mask=pool_active, other=0)
        center_active = pool_active & (offset_value == 4)
        source_offsets18 = (
            n_idx[:, None] * getitem18_stride_n
            + c_offsets[None, :] * getitem18_stride_c
            + pool_h[:, None] * getitem18_stride_h
            + pool_w[:, None] * getitem18_stride_w
        )
        source_offsets33 = (
            n_idx[:, None] * getitem33_stride_n
            + c_offsets[None, :] * getitem33_stride_c
            + pool_h[:, None] * getitem33_stride_h
            + pool_w[:, None] * getitem33_stride_w
        )
        scatter = (
            tl.load(getitem18_ptr + source_offsets18, mask=center_active, other=0.0).to(tl.float32)
            + tl.load(getitem33_ptr + source_offsets33, mask=center_active, other=0.0).to(tl.float32)
        )
        grad = tl.where(active, tl.where(gate, scatter, full), 0.0)

        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        scale = invstd * weight
        out = (grad - centered * var_term[None, :] - mean_term[None, :]) * scale[None, :]
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + h_idx[:, None] * out0_stride_h
            + w_idx[:, None] * out0_stride_w
        )
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _channel_bn_scatter_reduce_kernel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        out0_ptr,
        out1_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        c = tl.program_id(0)
        block_offsets = tl.arange(0, BLOCK_M_)

        mean = tl.load(mean_ptr + c * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        sum_grad = tl.full((), 0.0, tl.float32)
        sum_centered = tl.full((), 0.0, tl.float32)
        for block_start in tl.range(0, N_HW_, BLOCK_M_):
            m_offsets = block_start + block_offsets
            active = m_offsets < N_HW_
            n_idx = m_offsets // HW_
            spatial = m_offsets - n_idx * HW_
            h_idx = spatial // W_
            w_idx = spatial - h_idx * W_
            pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
            pool_h = h_idx // 2
            pool_w = w_idx // 2

            x_offsets = (
                n_idx * x_stride_n
                + c * x_stride_c
                + h_idx * x_stride_h
                + w_idx * x_stride_w
            )
            x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
            centered = x - mean
            affine = centered * invstd * weight + bias
            gate = (affine > 0.0) | (affine != affine)

            pool_active = active & pool_site
            pool_offsets = (
                n_idx * offsets_stride_n
                + c * offsets_stride_c
                + pool_h * offsets_stride_h
                + pool_w * offsets_stride_w
            )
            offset_value = tl.load(offsets_ptr + pool_offsets, mask=pool_active, other=0)
            center_active = pool_active & (offset_value == 4)
            source_offsets18 = (
                n_idx * getitem18_stride_n
                + c * getitem18_stride_c
                + pool_h * getitem18_stride_h
                + pool_w * getitem18_stride_w
            )
            source_offsets33 = (
                n_idx * getitem33_stride_n
                + c * getitem33_stride_c
                + pool_h * getitem33_stride_h
                + pool_w * getitem33_stride_w
            )
            scatter = (
                tl.load(getitem18_ptr + source_offsets18, mask=center_active, other=0.0).to(tl.float32)
                + tl.load(getitem33_ptr + source_offsets33, mask=center_active, other=0.0).to(tl.float32)
            )
            grad = tl.where(active, tl.where(gate, scatter, full), 0.0)
            sum_grad += tl.sum(grad, axis=0)
            sum_centered += tl.sum(grad * tl.where(active, centered, 0.0), axis=0)

        tl.store(out1_ptr + c, sum_centered * invstd)
        mean_term = sum_grad * REDUCTION_SCALE_
        var_term = sum_centered * REDUCTION_SCALE_ * invstd * invstd
        scale = invstd * weight

        for block_start in tl.range(0, N_HW_, BLOCK_M_):
            m_offsets = block_start + block_offsets
            active = m_offsets < N_HW_
            n_idx = m_offsets // HW_
            spatial = m_offsets - n_idx * HW_
            h_idx = spatial // W_
            w_idx = spatial - h_idx * W_
            pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
            pool_h = h_idx // 2
            pool_w = w_idx // 2

            x_offsets = (
                n_idx * x_stride_n
                + c * x_stride_c
                + h_idx * x_stride_h
                + w_idx * x_stride_w
            )
            x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
            centered = x - mean
            affine = centered * invstd * weight + bias
            gate = (affine > 0.0) | (affine != affine)

            pool_active = active & pool_site
            pool_offsets = (
                n_idx * offsets_stride_n
                + c * offsets_stride_c
                + pool_h * offsets_stride_h
                + pool_w * offsets_stride_w
            )
            offset_value = tl.load(offsets_ptr + pool_offsets, mask=pool_active, other=0)
            center_active = pool_active & (offset_value == 4)
            source_offsets18 = (
                n_idx * getitem18_stride_n
                + c * getitem18_stride_c
                + pool_h * getitem18_stride_h
                + pool_w * getitem18_stride_w
            )
            source_offsets33 = (
                n_idx * getitem33_stride_n
                + c * getitem33_stride_c
                + pool_h * getitem33_stride_h
                + pool_w * getitem33_stride_w
            )
            scatter = (
                tl.load(getitem18_ptr + source_offsets18, mask=center_active, other=0.0).to(tl.float32)
                + tl.load(getitem33_ptr + source_offsets33, mask=center_active, other=0.0).to(tl.float32)
            )
            grad = tl.where(active, tl.where(gate, scatter, full), 0.0)
            out = (grad - centered * var_term - mean_term) * scale
            out_offsets = (
                n_idx * out0_stride_n
                + c * out0_stride_c
                + h_idx * out0_stride_h
                + w_idx * out0_stride_w
            )
            tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _center_scatter_grad_for_channel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        c: tl.constexpr,
        m_offsets,
        active,
        mean,
        invstd,
        weight,
        bias,
        full,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
    ):
        n_idx = m_offsets // HW_
        spatial = m_offsets - n_idx * HW_
        h_idx = spatial // W_
        w_idx = spatial - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        x_offsets = (
            n_idx * x_stride_n
            + c * x_stride_c
            + h_idx * x_stride_h
            + w_idx * x_stride_w
        )
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean
        affine = centered * invstd * weight + bias
        gate = (affine > 0.0) | (affine != affine)

        pool_active = active & pool_site
        pool_offsets = (
            n_idx * offsets_stride_n
            + c * offsets_stride_c
            + pool_h * offsets_stride_h
            + pool_w * offsets_stride_w
        )
        offset_value = tl.load(offsets_ptr + pool_offsets, mask=pool_active, other=0)
        center_active = pool_active & (offset_value == 4)
        source_offsets18 = (
            n_idx * getitem18_stride_n
            + c * getitem18_stride_c
            + pool_h * getitem18_stride_h
            + pool_w * getitem18_stride_w
        )
        source_offsets33 = (
            n_idx * getitem33_stride_n
            + c * getitem33_stride_c
            + pool_h * getitem33_stride_h
            + pool_w * getitem33_stride_w
        )
        scatter = (
            tl.load(getitem18_ptr + source_offsets18, mask=center_active, other=0.0).to(tl.float32)
            + tl.load(getitem33_ptr + source_offsets33, mask=center_active, other=0.0).to(tl.float32)
        )
        grad = tl.where(active, tl.where(gate, scatter, full), 0.0)
        return grad, centered, n_idx, h_idx, w_idx

    @triton.jit
    def _channel_partial_sums_kernel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial0_ptr,
        partial1_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        m_offsets = tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        active = m_offsets < N_HW_

        mean = tl.load(mean_ptr + c * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)
        grad, centered, _, _, _ = _center_scatter_grad_for_channel(
            getitem18_ptr,
            getitem33_ptr,
            offsets_ptr,
            x_ptr,
            c,
            m_offsets,
            active,
            mean,
            invstd,
            weight,
            bias,
            full,
            getitem18_stride_n,
            getitem18_stride_c,
            getitem18_stride_h,
            getitem18_stride_w,
            getitem33_stride_n,
            getitem33_stride_c,
            getitem33_stride_h,
            getitem33_stride_w,
            offsets_stride_n,
            offsets_stride_c,
            offsets_stride_h,
            offsets_stride_w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            W_,
            HW_,
        )
        partial_offset = c * NUM_TILES_ + tile
        tl.store(partial0_ptr + partial_offset, tl.sum(grad, axis=0))
        tl.store(partial1_ptr + partial_offset, tl.sum(grad * tl.where(active, centered, 0.0), axis=0))

    @triton.jit
    def _channel_finalize_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        invstd_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        invstd_stride_c: tl.constexpr,
        NUM_TILES_: tl.constexpr,
        BLOCK_TILES_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile_offsets = tl.arange(0, BLOCK_TILES_)
        mask = tile_offsets < NUM_TILES_
        partial_offsets = c * NUM_TILES_ + tile_offsets
        sum0 = tl.sum(tl.load(partial0_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        sum1 = tl.sum(tl.load(partial1_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32), axis=0)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        tl.store(sum0_ptr + c, sum0)
        tl.store(sum1_ptr + c, sum1)
        tl.store(out1_ptr + c, sum1 * invstd)

    @triton.jit
    def _channel_output_kernel(
        getitem18_ptr,
        getitem33_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
    ):
        c = tl.program_id(0)
        tile = tl.program_id(1)
        m_offsets = tile * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        active = m_offsets < N_HW_

        mean = tl.load(mean_ptr + c * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c).to(tl.float32)
        grad, centered, n_idx, h_idx, w_idx = _center_scatter_grad_for_channel(
            getitem18_ptr,
            getitem33_ptr,
            offsets_ptr,
            x_ptr,
            c,
            m_offsets,
            active,
            mean,
            invstd,
            weight,
            bias,
            full,
            getitem18_stride_n,
            getitem18_stride_c,
            getitem18_stride_h,
            getitem18_stride_w,
            getitem33_stride_n,
            getitem33_stride_c,
            getitem33_stride_h,
            getitem33_stride_w,
            offsets_stride_n,
            offsets_stride_c,
            offsets_stride_h,
            offsets_stride_w,
            x_stride_n,
            x_stride_c,
            x_stride_h,
            x_stride_w,
            W_,
            HW_,
        )

        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        scale = invstd * weight
        out = (grad - centered * var_term - mean_term) * scale
        out_offsets = (
            n_idx * out0_stride_n
            + c * out0_stride_c
            + h_idx * out0_stride_h
            + w_idx * out0_stride_w
        )
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _linear_bn_input_grad_kernel(
        getitem18_ptr,
        getitem33_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        TOTAL_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        linear = tl.program_id(0) * BLOCK_ + tl.arange(0, BLOCK_)
        active = linear < TOTAL_
        w_idx = linear % W_
        h_idx = (linear // W_) % (HW_ // W_)
        c = (linear // HW_) % C_
        n_idx = linear // (C_ * HW_)
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        mean = tl.load(mean_ptr + c * mean_stride_c, mask=active, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c, mask=active, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c, mask=active, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c, mask=active, other=0.0).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c, mask=active, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        x_offsets = n_idx * x_stride_n + c * x_stride_c + h_idx * x_stride_h + w_idx * x_stride_w
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean
        affine = centered * invstd * weight + bias
        gate = (affine > 0.0) | (affine != affine)

        source_mask = active & pool_site
        source_offsets18 = (
            n_idx * getitem18_stride_n
            + c * getitem18_stride_c
            + pool_h * getitem18_stride_h
            + pool_w * getitem18_stride_w
        )
        source_offsets33 = (
            n_idx * getitem33_stride_n
            + c * getitem33_stride_c
            + pool_h * getitem33_stride_h
            + pool_w * getitem33_stride_w
        )
        scatter = (
            tl.load(getitem18_ptr + source_offsets18, mask=source_mask, other=0.0).to(tl.float32)
            + tl.load(getitem33_ptr + source_offsets33, mask=source_mask, other=0.0).to(tl.float32)
        )
        grad = tl.where(active, tl.where(gate, scatter, full), 0.0)
        mean_term = sum0 * REDUCTION_SCALE_
        var_term = sum1 * REDUCTION_SCALE_ * invstd * invstd
        scale = invstd * weight
        out = (grad - centered * var_term - mean_term) * scale
        out_offsets = (
            n_idx * out0_stride_n
            + c * out0_stride_c
            + h_idx * out0_stride_h
            + w_idx * out0_stride_w
        )
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _nc_bn_input_grad_kernel(
        getitem18_ptr,
        getitem33_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        getitem18_stride_n: tl.constexpr,
        getitem18_stride_c: tl.constexpr,
        getitem18_stride_h: tl.constexpr,
        getitem18_stride_w: tl.constexpr,
        getitem33_stride_n: tl.constexpr,
        getitem33_stride_c: tl.constexpr,
        getitem33_stride_h: tl.constexpr,
        getitem33_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        weight_stride_c: tl.constexpr,
        bias_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_: tl.constexpr,
    ):
        c = tl.program_id(0)
        n_idx = tl.program_id(1)
        hw = tl.arange(0, BLOCK_)
        active = hw < HW_
        h_idx = hw // W_
        w_idx = hw - h_idx * W_
        pool_site = ((h_idx & 1) == 1) & ((w_idx & 1) == 1)
        pool_h = h_idx // 2
        pool_w = w_idx // 2

        mean = tl.load(mean_ptr + c * mean_stride_c).to(tl.float32)
        invstd = tl.load(invstd_ptr + c * invstd_stride_c).to(tl.float32)
        weight = tl.load(weight_ptr + c * weight_stride_c).to(tl.float32)
        bias = tl.load(bias_ptr + c * bias_stride_c).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        x_offsets = n_idx * x_stride_n + c * x_stride_c + h_idx * x_stride_h + w_idx * x_stride_w
        x = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x - mean
        affine = centered * invstd * weight + bias
        gate = (affine > 0.0) | (affine != affine)

        source_offsets18 = (
            n_idx * getitem18_stride_n
            + c * getitem18_stride_c
            + pool_h * getitem18_stride_h
            + pool_w * getitem18_stride_w
        )
        source_offsets33 = (
            n_idx * getitem33_stride_n
            + c * getitem33_stride_c
            + pool_h * getitem33_stride_h
            + pool_w * getitem33_stride_w
        )
        scatter = (
            tl.load(getitem18_ptr + source_offsets18, mask=active & pool_site, other=0.0).to(tl.float32)
            + tl.load(getitem33_ptr + source_offsets33, mask=active & pool_site, other=0.0).to(tl.float32)
        )
        grad = tl.where(active, tl.where(gate, scatter, full), 0.0)
        out = (grad - centered * (sum1 * REDUCTION_SCALE_ * invstd * invstd) - sum0 * REDUCTION_SCALE_) * (
            invstd * weight
        )
        out_offsets = n_idx * out0_stride_n + c * out0_stride_c + h_idx * out0_stride_h + w_idx * out0_stride_w
        tl.store(out0_ptr + out_offsets, out, mask=active)


def oracle_maxpool_bn_scatter_reduce(
    getitem_18: torch.Tensor,
    getitem_33: torch.Tensor,
    arg180_1: torch.Tensor,
    arg176_1: torch.Tensor,
    arg177_1: torch.Tensor,
    arg178_1: torch.Tensor,
    arg60_1: torch.Tensor,
    arg61_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    if triton is None:
        raise RuntimeError("triton is not available")
    if getitem_18.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.empty_strided(
        tuple(arg176_1.shape),
        tuple(arg176_1.stride()),
        device=arg176_1.device,
        dtype=arg176_1.dtype,
    )
    out1 = torch.empty((C,), device=arg176_1.device, dtype=arg176_1.dtype)
    partial0 = torch.empty((NUM_M_TILES, C), device=arg176_1.device, dtype=torch.float32)
    partial1 = torch.empty_like(partial0)
    sum0 = torch.empty((C,), device=arg176_1.device, dtype=torch.float32)
    sum1 = torch.empty_like(sum0)

    grid = (triton.cdiv(C, BLOCK_C), NUM_M_TILES)
    _partial_bn_sums_kernel[grid](
        getitem_18,
        getitem_33,
        arg180_1,
        arg176_1,
        arg177_1,
        arg178_1,
        arg60_1,
        arg61_1,
        full,
        partial0,
        partial1,
        getitem18_stride_n=getitem_18.stride(0),
        getitem18_stride_c=getitem_18.stride(1),
        getitem18_stride_h=getitem_18.stride(2),
        getitem18_stride_w=getitem_18.stride(3),
        getitem33_stride_n=getitem_33.stride(0),
        getitem33_stride_c=getitem_33.stride(1),
        getitem33_stride_h=getitem_33.stride(2),
        getitem33_stride_w=getitem_33.stride(3),
        offsets_stride_n=arg180_1.stride(0),
        offsets_stride_c=arg180_1.stride(1),
        offsets_stride_h=arg180_1.stride(2),
        offsets_stride_w=arg180_1.stride(3),
        x_stride_n=arg176_1.stride(0),
        x_stride_c=arg176_1.stride(1),
        x_stride_h=arg176_1.stride(2),
        x_stride_w=arg176_1.stride(3),
        mean_stride_c=arg177_1.stride(1),
        invstd_stride_c=arg178_1.stride(1),
        weight_stride_c=arg60_1.stride(0),
        bias_stride_c=arg61_1.stride(0),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    _finalize_bn_sums_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        arg178_1,
        sum0,
        sum1,
        out1,
        invstd_stride_c=arg178_1.stride(1),
        C_=C,
        NUM_M_TILES_=NUM_M_TILES,
        BLOCK_C_=BLOCK_C,
        BLOCK_TILES_=BLOCK_TILES,
        num_warps=1,
    )
    total = N * C * HW
    _linear_bn_input_grad_kernel[(triton.cdiv(total, LINEAR_BLOCK),)](
        getitem_18,
        getitem_33,
        arg176_1,
        arg177_1,
        arg178_1,
        arg60_1,
        arg61_1,
        full,
        sum0,
        sum1,
        out0,
        getitem18_stride_n=getitem_18.stride(0),
        getitem18_stride_c=getitem_18.stride(1),
        getitem18_stride_h=getitem_18.stride(2),
        getitem18_stride_w=getitem_18.stride(3),
        getitem33_stride_n=getitem_33.stride(0),
        getitem33_stride_c=getitem_33.stride(1),
        getitem33_stride_h=getitem_33.stride(2),
        getitem33_stride_w=getitem_33.stride(3),
        x_stride_n=arg176_1.stride(0),
        x_stride_c=arg176_1.stride(1),
        x_stride_h=arg176_1.stride(2),
        x_stride_w=arg176_1.stride(3),
        mean_stride_c=arg177_1.stride(1),
        invstd_stride_c=arg178_1.stride(1),
        weight_stride_c=arg60_1.stride(0),
        bias_stride_c=arg61_1.stride(0),
        out0_stride_n=out0.stride(0),
        out0_stride_c=out0.stride(1),
        out0_stride_h=out0.stride(2),
        out0_stride_w=out0.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        TOTAL_=total,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_=LINEAR_BLOCK,
        num_warps=4,
    )
    return out0, out1


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    return module.Repro().to(device)(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    if device.type == "cuda":
        from triton.testing import do_bench

        return do_bench(fn, warmup=warmup, rep=rep, return_mode="min") * 1000.0

    for _ in range(max(0, warmup)):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


@oracle_impl(hardware="H100", shapes="(T([32, 1888, 7, 7], f32), T([32, 768, 7, 7], f32), T([32, 768, 7, 7], i8, gen=Index(5, 4)), T([32, 768, 14, 14], f32), T([1, 768, 1, 1], f32), T([1, 768, 1, 1], f32), T([768], f32), T([768], f32), T([], f32), S([24576, 49]), S([24576, 49]), S([32, 768, 14, 14]))")
def oracle_forward(inputs):
    return oracle_maxpool_bn_scatter_reduce(*inputs)


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

    if args.show_hw:
        import json
        print(json.dumps(get_hardware_info(), indent=2))
        return

    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = _harness_get_inputs(REPRO_DIR)
    instance = _harness_get_repro_instance(REPRO_DIR)

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
                oracle_forward,
                REPRO_DIR,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            for result in results:
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile "
                          f"for {result['repro_id']} (ratio={result['ratio']:.3f}x)")
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


if __name__ == "__main__":
    main()
