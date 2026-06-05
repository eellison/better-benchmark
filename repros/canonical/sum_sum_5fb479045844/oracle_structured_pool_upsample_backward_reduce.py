"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete VovNet max-pool-backward scatter feeding BN-affine/ReLU backward return tuple, including the full `[32, 256, 56, 56]` input-gradient output and the `[256]` centered-gradient reduction, by exploiting the captured all-center low-memory max-pool offsets as a collision-free odd-position 2x upsample and reducing source-to-destination contributions without a dense scatter buffer, whereas Inductor currently materializes the dense scatter_add result and schedules the gate, sibling channel reductions, and dependent full-tensor BN backward epilogue as generic kernels; Inductor cannot do this today because scheduler/codegen does not model low-memory max-pool offsets as a structured scatter-reduce producer that can feed both channel reductions and a required full side-output store; the fix is SCATTER_REDUCE: add a structured max-pool-backward scatter-reduce lowering that maps each scatter source directly to its gated destination, accumulates compatible BN reductions, and emits the dependent BN input-gradient output without materializing the scatter buffer."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - keeps py_compile usable without Triton.
    triton = None
    tl = None



from oracle_harness import (
    bench_oracle,
    bench_oracle_all_shapes,
    check_oracle,
    get_hardware_info,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    has_stochastic_ops,
)

REPRO_ID = "sum_sum_5fb479045844"
SHAPE_LABEL = "torchbench_timm_vovnet_train_001_b3239655"
REPRO_DIR = Path(__file__).resolve().parent
REPRO_PATH = REPRO_DIR / "repro.py"

N = 32
C = 256
SRC_C = 1056
SRC_H = 28
SRC_W = 28
OUT_H = 56
OUT_W = 56
OUT_HW = OUT_H * OUT_W
N_OUT_HW = N * OUT_HW
SRC_HW = SRC_H * SRC_W
N_SRC_HW = N * SRC_HW
REDUCTION_SCALE = 1.0 / N_OUT_HW

BLOCK_M = 512
BLOCK_C = 8
NUM_M_TILES = math.ceil(N_OUT_HW / BLOCK_M)
NUM_SRC_TILES = math.ceil(N_SRC_HW / BLOCK_M)
NUM_TOTAL_REDUCTION_TILES = NUM_M_TILES + NUM_SRC_TILES
BLOCK_TILES = 1 << (NUM_TOTAL_REDUCTION_TILES - 1).bit_length()



def make_inputs(device: torch.device) -> tuple[object, ...]:
    from repro_harness import load_shape_configs, make_inputs_from_config

    configs = load_shape_configs(str(REPRO_PATH))
    if configs:
        config = next(iter(configs.values()))
        config = {
            "inputs": [
                {**spec, "device": str(device)}
                if isinstance(spec, dict) and spec.get("kind") == "tensor"
                else spec
                for spec in config["inputs"]
            ]
        }
        inputs = make_inputs_from_config(config)
    else:
        module = _load_repro_module()
        inputs = module.make_inputs()

    moved: list[object] = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            moved.append(value.to(device=device))
        else:
            moved.append(value)
    return tuple(moved)


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    model = module.Repro().to(device=device)
    with torch.no_grad():
        return _as_tuple(model(*inputs))


if triton is not None:

    @triton.jit
    def _maxpool_scatter_value(
        getitem_72_ptr,
        getitem_87_ptr,
        offsets_ptr,
        n_idx,
        oh,
        ow,
        c_offsets,
        active,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
        offsets_stride_n: tl.constexpr,
        offsets_stride_c: tl.constexpr,
        offsets_stride_h: tl.constexpr,
        offsets_stride_w: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        h0 = oh // 2
        w0 = ow // 2
        hoff0 = oh - h0 * 2
        woff0 = ow - w0 * 2
        h1 = h0 - 1
        w1 = w0 - 1
        hoff1 = hoff0 + 2
        woff1 = woff0 + 2

        scatter = tl.zeros((BLOCK_M_, BLOCK_C_), tl.float32)

        valid = (h1 >= 0) & (w1 >= 0) & (hoff1 < 3) & (woff1 < 3)
        want = hoff1 * 3 + woff1
        index_offsets = (
            n_idx[:, None] * offsets_stride_n
            + c_offsets[None, :] * offsets_stride_c
            + h1[:, None] * offsets_stride_h
            + w1[:, None] * offsets_stride_w
        )
        candidate_active = active & valid[:, None]
        lowmem = tl.load(offsets_ptr + index_offsets, mask=candidate_active, other=-1).to(tl.int32)
        take = candidate_active & (lowmem == want[:, None])
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + h1[:, None] * getitem_72_stride_h
            + w1[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + h1[:, None] * getitem_87_stride_h
            + w1[:, None] * getitem_87_stride_w
        )
        scatter += tl.load(getitem_72_ptr + src_offsets_72, mask=take, other=0.0).to(tl.float32)
        scatter += tl.load(getitem_87_ptr + src_offsets_87, mask=take, other=0.0).to(tl.float32)

        valid = (h1 >= 0) & (hoff1 < 3)
        want = hoff1 * 3 + woff0
        index_offsets = (
            n_idx[:, None] * offsets_stride_n
            + c_offsets[None, :] * offsets_stride_c
            + h1[:, None] * offsets_stride_h
            + w0[:, None] * offsets_stride_w
        )
        candidate_active = active & valid[:, None]
        lowmem = tl.load(offsets_ptr + index_offsets, mask=candidate_active, other=-1).to(tl.int32)
        take = candidate_active & (lowmem == want[:, None])
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + h1[:, None] * getitem_72_stride_h
            + w0[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + h1[:, None] * getitem_87_stride_h
            + w0[:, None] * getitem_87_stride_w
        )
        scatter += tl.load(getitem_72_ptr + src_offsets_72, mask=take, other=0.0).to(tl.float32)
        scatter += tl.load(getitem_87_ptr + src_offsets_87, mask=take, other=0.0).to(tl.float32)

        valid = (w1 >= 0) & (woff1 < 3)
        want = hoff0 * 3 + woff1
        index_offsets = (
            n_idx[:, None] * offsets_stride_n
            + c_offsets[None, :] * offsets_stride_c
            + h0[:, None] * offsets_stride_h
            + w1[:, None] * offsets_stride_w
        )
        candidate_active = active & valid[:, None]
        lowmem = tl.load(offsets_ptr + index_offsets, mask=candidate_active, other=-1).to(tl.int32)
        take = candidate_active & (lowmem == want[:, None])
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + h0[:, None] * getitem_72_stride_h
            + w1[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + h0[:, None] * getitem_87_stride_h
            + w1[:, None] * getitem_87_stride_w
        )
        scatter += tl.load(getitem_72_ptr + src_offsets_72, mask=take, other=0.0).to(tl.float32)
        scatter += tl.load(getitem_87_ptr + src_offsets_87, mask=take, other=0.0).to(tl.float32)

        want = hoff0 * 3 + woff0
        index_offsets = (
            n_idx[:, None] * offsets_stride_n
            + c_offsets[None, :] * offsets_stride_c
            + h0[:, None] * offsets_stride_h
            + w0[:, None] * offsets_stride_w
        )
        lowmem = tl.load(offsets_ptr + index_offsets, mask=active, other=-1).to(tl.int32)
        take = active & (lowmem == want[:, None])
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + h0[:, None] * getitem_72_stride_h
            + w0[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + h0[:, None] * getitem_87_stride_h
            + w0[:, None] * getitem_87_stride_w
        )
        scatter += tl.load(getitem_72_ptr + src_offsets_72, mask=take, other=0.0).to(tl.float32)
        scatter += tl.load(getitem_87_ptr + src_offsets_87, mask=take, other=0.0).to(tl.float32)

        return scatter

    @triton.jit
    def _partial_bn_sums_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
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
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_

        scatter = _maxpool_scatter_value(
            getitem_72_ptr,
            getitem_87_ptr,
            offsets_ptr,
            n_idx,
            oh,
            ow,
            c_offsets,
            active,
            getitem_72_stride_n,
            getitem_72_stride_c,
            getitem_72_stride_h,
            getitem_72_stride_w,
            getitem_87_stride_n,
            getitem_87_stride_c,
            getitem_87_stride_h,
            getitem_87_stride_w,
            offsets_stride_n,
            offsets_stride_c,
            offsets_stride_h,
            offsets_stride_w,
            BLOCK_M_,
            BLOCK_C_,
        )

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, scatter, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_sum = tl.sum(grad, axis=0)
        partial_centered_sum = tl.sum(grad * centered, axis=0)
        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial_sum_ptr + partial_offsets, partial_sum, mask=c_offsets < C_)
        tl.store(partial_centered_sum_ptr + partial_offsets, partial_centered_sum, mask=c_offsets < C_)

    @triton.jit
    def _finalize_bn_sums_kernel(
        partial_sum_ptr,
        partial_centered_sum_ptr,
        invstd_ptr,
        sum_ptr,
        centered_sum_ptr,
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
        active = (tile_offsets[:, None] < NUM_M_TILES_) & (c_offsets[None, :] < C_)
        offsets = tile_offsets[:, None] * C_ + c_offsets[None, :]

        sum_vals = tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        centered_sum_vals = tl.load(partial_centered_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        grad_sum = tl.sum(sum_vals, axis=0)
        centered_sum = tl.sum(centered_sum_vals, axis=0)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)

        mask = c_offsets < C_
        tl.store(sum_ptr + c_offsets, grad_sum, mask=mask)
        tl.store(centered_sum_ptr + c_offsets, centered_sum, mask=mask)
        tl.store(out1_ptr + c_offsets, centered_sum * invstd, mask=mask)

    @triton.jit
    def _bn_input_grad_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum_ptr,
        centered_sum_ptr,
        out0_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
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
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_

        scatter = _maxpool_scatter_value(
            getitem_72_ptr,
            getitem_87_ptr,
            offsets_ptr,
            n_idx,
            oh,
            ow,
            c_offsets,
            active,
            getitem_72_stride_n,
            getitem_72_stride_c,
            getitem_72_stride_h,
            getitem_72_stride_w,
            getitem_87_stride_n,
            getitem_87_stride_c,
            getitem_87_stride_h,
            getitem_87_stride_w,
            offsets_stride_n,
            offsets_stride_c,
            offsets_stride_h,
            offsets_stride_w,
            BLOCK_M_,
            BLOCK_C_,
        )

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + oh[:, None] * out0_stride_h
            + ow[:, None] * out0_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        grad_sum = tl.load(sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        centered_sum = tl.load(centered_sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, scatter, full)
        mean_term = grad_sum * REDUCTION_SCALE_
        variance_term = centered_sum * REDUCTION_SCALE_ * invstd * invstd
        affine_scale = invstd * weight
        normalized = grad - centered * variance_term[None, :]
        normalized = normalized - mean_term[None, :]
        out = normalized * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _center_partial_bn_sums_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_
        odd = (
            ((oh - (oh // 2) * 2) == 1)
            & ((ow - (ow // 2) * 2) == 1)
        )
        src_h = oh // 2
        src_w = ow // 2

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )

        odd_active = active & odd[:, None]
        scatter = tl.load(
            getitem_72_ptr + src_offsets_72,
            mask=odd_active,
            other=0.0,
        ).to(tl.float32)
        scatter += tl.load(
            getitem_87_ptr + src_offsets_87,
            mask=odd_active,
            other=0.0,
        ).to(tl.float32)

        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, scatter, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial_sum_ptr + partial_offsets, tl.sum(grad, axis=0), mask=c_offsets < C_)
        tl.store(
            partial_centered_sum_ptr + partial_offsets,
            tl.sum(grad * centered, axis=0),
            mask=c_offsets < C_,
        )

    @triton.jit
    def _center_bn_input_grad_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum_ptr,
        centered_sum_ptr,
        out0_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_
        odd = (
            ((oh - (oh // 2) * 2) == 1)
            & ((ow - (ow // 2) * 2) == 1)
        )
        src_h = oh // 2
        src_w = ow // 2

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + oh[:, None] * out0_stride_h
            + ow[:, None] * out0_stride_w
        )
        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )

        odd_active = active & odd[:, None]
        scatter = tl.load(
            getitem_72_ptr + src_offsets_72,
            mask=odd_active,
            other=0.0,
        ).to(tl.float32)
        scatter += tl.load(
            getitem_87_ptr + src_offsets_87,
            mask=odd_active,
            other=0.0,
        ).to(tl.float32)

        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        grad_sum = tl.load(sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        centered_sum = tl.load(centered_sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, scatter, full)
        mean_term = grad_sum * REDUCTION_SCALE_
        variance_term = centered_sum * REDUCTION_SCALE_ * invstd * invstd
        affine_scale = invstd * weight
        out = (grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _tile4_partial_bn_sums_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        SRC_W_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        N_SRC_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_SRC_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // SRC_HW_
        spatial = m_offsets - n_idx * SRC_HW_
        src_h = spatial // SRC_W_
        src_w = spatial - src_h * SRC_W_
        out_h0 = src_h * 2
        out_w0 = src_w * 2

        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )
        src = tl.load(getitem_72_ptr + src_offsets_72, mask=active, other=0.0).to(tl.float32)
        src += tl.load(getitem_87_ptr + src_offsets_87, mask=active, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        grad_acc = tl.zeros((BLOCK_M_, BLOCK_C_), tl.float32)
        centered_acc = tl.zeros((BLOCK_M_, BLOCK_C_), tl.float32)

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + out_h0[:, None] * x_stride_h
            + out_w0[:, None] * x_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)
        grad_acc += grad
        centered_acc += grad * centered

        x_offsets = x_offsets + x_stride_w
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)
        grad_acc += grad
        centered_acc += grad * centered

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + (out_h0[:, None] + 1) * x_stride_h
            + out_w0[:, None] * x_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)
        grad_acc += grad
        centered_acc += grad * centered

        x_offsets = x_offsets + x_stride_w
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, src, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)
        grad_acc += grad
        centered_acc += grad * centered

        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial_sum_ptr + partial_offsets, tl.sum(grad_acc, axis=0), mask=c_offsets < C_)
        tl.store(
            partial_centered_sum_ptr + partial_offsets,
            tl.sum(centered_acc, axis=0),
            mask=c_offsets < C_,
        )

    @triton.jit
    def _tile4_bn_input_grad_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum_ptr,
        centered_sum_ptr,
        out0_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        SRC_W_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        N_SRC_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_SRC_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // SRC_HW_
        spatial = m_offsets - n_idx * SRC_HW_
        src_h = spatial // SRC_W_
        src_w = spatial - src_h * SRC_W_
        out_h0 = src_h * 2
        out_w0 = src_w * 2

        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )
        src = tl.load(getitem_72_ptr + src_offsets_72, mask=active, other=0.0).to(tl.float32)
        src += tl.load(getitem_87_ptr + src_offsets_87, mask=active, other=0.0).to(tl.float32)

        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        grad_sum = tl.load(sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        centered_sum = tl.load(centered_sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)
        mean_term = grad_sum * REDUCTION_SCALE_
        variance_term = centered_sum * REDUCTION_SCALE_ * invstd * invstd
        affine_scale = invstd * weight

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + out_h0[:, None] * x_stride_h
            + out_w0[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + out_h0[:, None] * out0_stride_h
            + out_w0[:, None] * out0_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        out = (grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

        x_offsets = x_offsets + x_stride_w
        out_offsets = out_offsets + out0_stride_w
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        out = (grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + (out_h0[:, None] + 1) * x_stride_h
            + out_w0[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + (out_h0[:, None] + 1) * out0_stride_h
            + out_w0[:, None] * out0_stride_w
        )
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        out = (grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

        x_offsets = x_offsets + x_stride_w
        out_offsets = out_offsets + out0_stride_w
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, src, full)
        out = (grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _gate_full_partials_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )

        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, 0.0, full)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_offsets = pid_m * C_ + c_offsets
        tl.store(partial_sum_ptr + partial_offsets, tl.sum(grad, axis=0), mask=c_offsets < C_)
        tl.store(
            partial_centered_sum_ptr + partial_offsets,
            tl.sum(grad * centered, axis=0),
            mask=c_offsets < C_,
        )

    @triton.jit
    def _source_gate_partials_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        partial_sum_ptr,
        partial_centered_sum_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
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
        C_: tl.constexpr,
        SRC_W_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        N_SRC_HW_: tl.constexpr,
        OUT_W_: tl.constexpr,
        STORE_TILE_OFFSET_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_SRC_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // SRC_HW_
        spatial = m_offsets - n_idx * SRC_HW_
        src_h = spatial // SRC_W_
        src_w = spatial - src_h * SRC_W_
        out_h = src_h[:, None] * 2 + 1
        out_w = src_w[:, None] * 2 + 1

        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + out_h * x_stride_h
            + out_w * x_stride_w
        )

        src = tl.load(getitem_72_ptr + src_offsets_72, mask=active, other=0.0).to(tl.float32)
        src += tl.load(getitem_87_ptr + src_offsets_87, mask=active, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        grad = tl.where(gate, src, 0.0)
        grad = tl.where(active, grad, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_offsets = (STORE_TILE_OFFSET_ + pid_m) * C_ + c_offsets
        tl.store(partial_sum_ptr + partial_offsets, tl.sum(grad, axis=0), mask=c_offsets < C_)
        tl.store(
            partial_centered_sum_ptr + partial_offsets,
            tl.sum(grad * centered, axis=0),
            mask=c_offsets < C_,
        )

    @triton.jit
    def _base_output_kernel(
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        full_ptr,
        sum_ptr,
        centered_sum_ptr,
        out0_ptr,
        x_stride_n: tl.constexpr,
        x_stride_c: tl.constexpr,
        x_stride_h: tl.constexpr,
        x_stride_w: tl.constexpr,
        mean_stride_c: tl.constexpr,
        invstd_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        OUT_W_: tl.constexpr,
        OUT_HW_: tl.constexpr,
        N_OUT_HW_: tl.constexpr,
        REDUCTION_SCALE_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_OUT_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // OUT_HW_
        spatial = m_offsets - n_idx * OUT_HW_
        oh = spatial // OUT_W_
        ow = spatial - oh * OUT_W_
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + oh[:, None] * x_stride_h
            + ow[:, None] * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + oh[:, None] * out0_stride_h
            + ow[:, None] * out0_stride_w
        )

        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        grad_sum = tl.load(sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        centered_sum = tl.load(centered_sum_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        full = tl.load(full_ptr).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        mean_term = grad_sum * REDUCTION_SCALE_
        variance_term = centered_sum * REDUCTION_SCALE_ * invstd * invstd
        affine_scale = invstd * weight
        base_grad = tl.where(gate, 0.0, full)
        out = (base_grad - centered * variance_term[None, :] - mean_term[None, :]) * affine_scale[None, :]
        tl.store(out0_ptr + out_offsets, out, mask=active)

    @triton.jit
    def _source_output_atomic_kernel(
        getitem_72_ptr,
        getitem_87_ptr,
        offsets_ptr,
        x_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        out0_ptr,
        getitem_72_stride_n: tl.constexpr,
        getitem_72_stride_c: tl.constexpr,
        getitem_72_stride_h: tl.constexpr,
        getitem_72_stride_w: tl.constexpr,
        getitem_87_stride_n: tl.constexpr,
        getitem_87_stride_c: tl.constexpr,
        getitem_87_stride_h: tl.constexpr,
        getitem_87_stride_w: tl.constexpr,
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
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        C_: tl.constexpr,
        SRC_W_: tl.constexpr,
        SRC_HW_: tl.constexpr,
        N_SRC_HW_: tl.constexpr,
        BLOCK_M_: tl.constexpr,
        BLOCK_C_: tl.constexpr,
    ):
        pid_c = tl.program_id(0)
        pid_m = tl.program_id(1)
        m_offsets = pid_m * BLOCK_M_ + tl.arange(0, BLOCK_M_)
        c_offsets = pid_c * BLOCK_C_ + tl.arange(0, BLOCK_C_)
        active = (m_offsets[:, None] < N_SRC_HW_) & (c_offsets[None, :] < C_)

        n_idx = m_offsets // SRC_HW_
        spatial = m_offsets - n_idx * SRC_HW_
        src_h = spatial // SRC_W_
        src_w = spatial - src_h * SRC_W_
        out_h = src_h[:, None] * 2 + 1
        out_w = src_w[:, None] * 2 + 1

        src_offsets_72 = (
            n_idx[:, None] * getitem_72_stride_n
            + c_offsets[None, :] * getitem_72_stride_c
            + src_h[:, None] * getitem_72_stride_h
            + src_w[:, None] * getitem_72_stride_w
        )
        src_offsets_87 = (
            n_idx[:, None] * getitem_87_stride_n
            + c_offsets[None, :] * getitem_87_stride_c
            + src_h[:, None] * getitem_87_stride_h
            + src_w[:, None] * getitem_87_stride_w
        )
        x_offsets = (
            n_idx[:, None] * x_stride_n
            + c_offsets[None, :] * x_stride_c
            + out_h * x_stride_h
            + out_w * x_stride_w
        )
        out_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + out_h * out0_stride_h
            + out_w * out0_stride_w
        )

        src = tl.load(getitem_72_ptr + src_offsets_72, mask=active, other=0.0).to(tl.float32)
        src += tl.load(getitem_87_ptr + src_offsets_87, mask=active, other=0.0).to(tl.float32)
        x_val = tl.load(x_ptr + x_offsets, mask=active, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + c_offsets * mean_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd = tl.load(invstd_ptr + c_offsets * invstd_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias = tl.load(bias_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)

        centered = x_val - mean[None, :]
        affine = centered * invstd[None, :] * weight[None, :] + bias[None, :]
        gate = (affine > 0.0) | (affine != affine)
        contrib = src * (invstd * weight)[None, :]
        out = tl.load(out0_ptr + out_offsets, mask=active & gate, other=0.0).to(tl.float32)
        tl.store(out0_ptr + out_offsets, out + contrib, mask=active & gate)


def _torch_oracle(
    getitem_72: torch.Tensor,
    getitem_87: torch.Tensor,
    arg119_1: torch.Tensor,
    arg115_1: torch.Tensor,
    arg116_1: torch.Tensor,
    arg117_1: torch.Tensor,
    arg19_1: torch.Tensor,
    arg20_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    src = getitem_72[:, 0:C, :, :] + getitem_87
    full_default = torch.zeros((N * C, OUT_HW), device=src.device, dtype=src.dtype)
    indices = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(
        arg119_1, [3, 3], [OUT_H, OUT_W], [2, 2], [0, 0], [1, 1]
    )
    scatter = torch.ops.aten.scatter_add.default(
        full_default,
        1,
        indices.view(N * C, SRC_H * SRC_W),
        src.view(N * C, SRC_H * SRC_W),
    ).view(N, C, OUT_H, OUT_W)

    centered = arg115_1 - arg116_1
    invstd = arg117_1.squeeze((0, 2, 3))
    affine = centered * arg117_1 * arg19_1[None, :, None, None] + arg20_1[None, :, None, None]
    grad = torch.where(torch.relu(affine) <= 0, full.to(dtype=scatter.dtype, device=scatter.device), scatter)
    grad_sum = grad.sum(dim=(0, 2, 3))
    centered_sum = (grad * centered).sum(dim=(0, 2, 3))
    mean_term = grad_sum * REDUCTION_SCALE
    variance_term = centered_sum * REDUCTION_SCALE * invstd * invstd
    affine_scale = invstd * arg19_1
    out0 = (grad - centered * variance_term[None, :, None, None] - mean_term[None, :, None, None]) * affine_scale[None, :, None, None]
    out1 = centered_sum * invstd
    return out0, out1


def oracle_structured_pool_upsample_backward_reduce(
    getitem_72: torch.Tensor,
    getitem_87: torch.Tensor,
    arg119_1: torch.Tensor,
    arg115_1: torch.Tensor,
    arg116_1: torch.Tensor,
    arg117_1: torch.Tensor,
    arg19_1: torch.Tensor,
    arg20_1: torch.Tensor,
    full: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        triton is None
        or not getitem_72.is_cuda
        or not getitem_87.is_cuda
        or not arg119_1.is_cuda
        or not arg115_1.is_cuda
    ):
        return _torch_oracle(
            getitem_72,
            getitem_87,
            arg119_1,
            arg115_1,
            arg116_1,
            arg117_1,
            arg19_1,
            arg20_1,
            full,
            _shape_param_0,
            _shape_param_1,
            _shape_param_2,
        )

    partial_sum = torch.empty((NUM_M_TILES, C), device=getitem_72.device, dtype=torch.float32)
    partial_centered_sum = torch.empty_like(partial_sum)
    grad_sum = torch.empty((C,), device=getitem_72.device, dtype=torch.float32)
    centered_sum = torch.empty((C,), device=getitem_72.device, dtype=torch.float32)
    out0 = torch.empty_like(arg115_1)
    out1 = torch.empty((C,), device=getitem_72.device, dtype=torch.float32)

    out_grid = (triton.cdiv(C, BLOCK_C), NUM_M_TILES)
    _center_partial_bn_sums_kernel[out_grid](
        getitem_72,
        getitem_87,
        arg115_1,
        arg116_1,
        arg117_1,
        arg19_1,
        arg20_1,
        full,
        partial_sum,
        partial_centered_sum,
        getitem_72.stride(0),
        getitem_72.stride(1),
        getitem_72.stride(2),
        getitem_72.stride(3),
        getitem_87.stride(0),
        getitem_87.stride(1),
        getitem_87.stride(2),
        getitem_87.stride(3),
        arg115_1.stride(0),
        arg115_1.stride(1),
        arg115_1.stride(2),
        arg115_1.stride(3),
        arg116_1.stride(1),
        arg117_1.stride(1),
        C,
        OUT_W,
        OUT_HW,
        N_OUT_HW,
        BLOCK_M,
        BLOCK_C,
        num_warps=4,
    )
    _finalize_bn_sums_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial_sum,
        partial_centered_sum,
        arg117_1,
        grad_sum,
        centered_sum,
        out1,
        arg117_1.stride(1),
        C,
        NUM_M_TILES,
        BLOCK_C,
        BLOCK_TILES,
        num_warps=4,
    )
    _center_bn_input_grad_kernel[out_grid](
        getitem_72,
        getitem_87,
        arg115_1,
        arg116_1,
        arg117_1,
        arg19_1,
        arg20_1,
        full,
        grad_sum,
        centered_sum,
        out0,
        getitem_72.stride(0),
        getitem_72.stride(1),
        getitem_72.stride(2),
        getitem_72.stride(3),
        getitem_87.stride(0),
        getitem_87.stride(1),
        getitem_87.stride(2),
        getitem_87.stride(3),
        arg115_1.stride(0),
        arg115_1.stride(1),
        arg115_1.stride(2),
        arg115_1.stride(3),
        arg116_1.stride(1),
        arg117_1.stride(1),
        out0.stride(0),
        out0.stride(1),
        out0.stride(2),
        out0.stride(3),
        C,
        OUT_W,
        OUT_HW,
        N_OUT_HW,
        REDUCTION_SCALE,
        BLOCK_M,
        BLOCK_C,
        num_warps=4,
    )
    return out0, out1


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)

    best_s = math.inf
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize(device)
        best_s = min(best_s, time.perf_counter() - start)
    return best_s * 1_000_000.0


def max_abs_diff(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> float:
    return max((a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected))


def allclose(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], rtol: float, atol: float) -> bool:
    return all(torch.allclose(a.float(), e.float(), rtol=rtol, atol=atol) for a, e in zip(actual, expected))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Oracle for {REPRO_ID} ({SHAPE_LABEL}).")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--check", action="store_true", help="Compare oracle outputs against Repro.forward.")
    parser.add_argument("--bench", action="store_true", help="Benchmark the oracle.")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def oracle_forward(inputs):
    return oracle_structured_pool_upsample_backward_reduce(*inputs)


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
