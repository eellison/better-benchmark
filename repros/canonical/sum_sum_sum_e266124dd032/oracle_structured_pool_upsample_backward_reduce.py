"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete four-output RepVGG adaptive-average-pool backward, shared two-branch BN-affine/ReLU gate, and both BN-backward return branches with Triton channel reductions plus channels-last side-output stores, whereas Inductor currently lowers the zero-fill as_strided_scatter, expand/div pool-gradient producer, ReLU mask, sibling channel reductions, and two dependent full-tensor BN-backward epilogues as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not model the zero-fill structured scatter/expand as an average-pool-backward scatter-reduce producer that can feed multiple BN reduction consumers and required full side-output stores; the fix is SCATTER_REDUCE: add a structured average-pool-backward scatter-reduce lowering that maps each pooled-gradient source directly into the shared gated channel reductions and emits all dependent BN-backward outputs for the full return tuple."""
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


REPRO_ID = "sum_sum_sum_e266124dd032"
REPRO_DIR = Path(__file__).resolve().parent
REPO_ROOT = REPRO_DIR.parents[2]
REPRO_PATH = REPRO_DIR / "repro.py"
SHAPE_LABEL = "timm_repvgg_a2_train_8282b150"

N = 128
C = 1408
H = 7
W = 7
HW = H * W
N_HW = N * HW
INV_HW = 1.0 / HW
REDUCTION_SCALE = 1.0 / N_HW
BLOCK_M = 128
BLOCK_C = 16
BLOCK_TILES = 64
NUM_M_TILES = triton.cdiv(N_HW, BLOCK_M) if triton is not None else math.ceil(N_HW / BLOCK_M)



def _load_repro_module():
    spec = importlib.util.spec_from_file_location(f"{REPRO_ID}_repro", REPRO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load repro module from {REPRO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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


if triton is not None:

    @triton.jit
    def _partial_shared_gate_sums_kernel(
        mm_ptr,
        x1_ptr,
        mean1_ptr,
        invstd1_ptr,
        weight1_ptr,
        bias1_ptr,
        x2_ptr,
        mean2_ptr,
        invstd2_ptr,
        weight2_ptr,
        bias2_ptr,
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        x1_stride_n: tl.constexpr,
        x1_stride_c: tl.constexpr,
        x1_stride_h: tl.constexpr,
        x1_stride_w: tl.constexpr,
        x2_stride_n: tl.constexpr,
        x2_stride_c: tl.constexpr,
        x2_stride_h: tl.constexpr,
        x2_stride_w: tl.constexpr,
        mean1_stride_c: tl.constexpr,
        invstd1_stride_c: tl.constexpr,
        weight1_stride_c: tl.constexpr,
        bias1_stride_c: tl.constexpr,
        mean2_stride_c: tl.constexpr,
        invstd2_stride_c: tl.constexpr,
        weight2_stride_c: tl.constexpr,
        bias2_stride_c: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
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

        mm_offsets = n_idx[:, None] * mm_stride_n + c_offsets[None, :] * mm_stride_c
        x1_offsets = (
            n_idx[:, None] * x1_stride_n
            + c_offsets[None, :] * x1_stride_c
            + h_idx[:, None] * x1_stride_h
            + w_idx[:, None] * x1_stride_w
        )
        x2_offsets = (
            n_idx[:, None] * x2_stride_n
            + c_offsets[None, :] * x2_stride_c
            + h_idx[:, None] * x2_stride_h
            + w_idx[:, None] * x2_stride_w
        )

        mean1 = tl.load(mean1_ptr + c_offsets * mean1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + c_offsets * invstd1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets * weight1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets * bias1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + c_offsets * mean2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd2 = tl.load(invstd2_ptr + c_offsets * invstd2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c_offsets * weight2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c_offsets * bias2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)

        x1 = tl.load(x1_ptr + x1_offsets, mask=active, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + x2_offsets, mask=active, other=0.0).to(tl.float32)
        mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

        centered1 = x1 - mean1[None, :]
        centered2 = x2 - mean2[None, :]
        affine1 = centered1 * invstd1[None, :] * weight1[None, :] + bias1[None, :]
        affine2 = centered2 * invstd2[None, :] * weight2[None, :] + bias2[None, :]
        affine = affine1 + affine2
        gate = ((affine > 0.0) | (affine != affine)) & active
        grad = tl.where(gate, mm * INV_HW_, 0.0)

        sum_grad = tl.sum(grad, axis=0)
        sum_centered1 = tl.sum(grad * tl.where(active, centered1, 0.0), axis=0)
        sum_centered2 = tl.sum(grad * tl.where(active, centered2, 0.0), axis=0)
        partial_offsets = pid_m * C_ + c_offsets
        partial_mask = c_offsets < C_
        tl.store(partial0_ptr + partial_offsets, sum_grad, mask=partial_mask)
        tl.store(partial1_ptr + partial_offsets, sum_centered1, mask=partial_mask)
        tl.store(partial2_ptr + partial_offsets, sum_centered2, mask=partial_mask)

    @triton.jit
    def _finalize_shared_gate_sums_kernel(
        partial0_ptr,
        partial1_ptr,
        partial2_ptr,
        invstd1_ptr,
        invstd2_ptr,
        sum0_ptr,
        sum1_ptr,
        sum2_ptr,
        out1_ptr,
        out3_ptr,
        invstd1_stride_c: tl.constexpr,
        invstd2_stride_c: tl.constexpr,
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
        sum2_vals = tl.load(partial2_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
        sum0 = tl.sum(sum0_vals, axis=0)
        sum1 = tl.sum(sum1_vals, axis=0)
        sum2 = tl.sum(sum2_vals, axis=0)
        invstd1 = tl.load(invstd1_ptr + c_offsets * invstd1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd2 = tl.load(invstd2_ptr + c_offsets * invstd2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)

        mask = c_offsets < C_
        tl.store(sum0_ptr + c_offsets, sum0, mask=mask)
        tl.store(sum1_ptr + c_offsets, sum1, mask=mask)
        tl.store(sum2_ptr + c_offsets, sum2, mask=mask)
        tl.store(out1_ptr + c_offsets, sum2 * invstd2, mask=mask)
        tl.store(out3_ptr + c_offsets, sum1 * invstd1, mask=mask)

    @triton.jit
    def _shared_gate_bn_outputs_kernel(
        mm_ptr,
        x1_ptr,
        mean1_ptr,
        invstd1_ptr,
        weight1_ptr,
        bias1_ptr,
        x2_ptr,
        mean2_ptr,
        invstd2_ptr,
        weight2_ptr,
        bias2_ptr,
        sum0_ptr,
        sum1_ptr,
        sum2_ptr,
        out0_ptr,
        out2_ptr,
        mm_stride_n: tl.constexpr,
        mm_stride_c: tl.constexpr,
        x1_stride_n: tl.constexpr,
        x1_stride_c: tl.constexpr,
        x1_stride_h: tl.constexpr,
        x1_stride_w: tl.constexpr,
        x2_stride_n: tl.constexpr,
        x2_stride_c: tl.constexpr,
        x2_stride_h: tl.constexpr,
        x2_stride_w: tl.constexpr,
        mean1_stride_c: tl.constexpr,
        invstd1_stride_c: tl.constexpr,
        weight1_stride_c: tl.constexpr,
        bias1_stride_c: tl.constexpr,
        mean2_stride_c: tl.constexpr,
        invstd2_stride_c: tl.constexpr,
        weight2_stride_c: tl.constexpr,
        bias2_stride_c: tl.constexpr,
        out0_stride_n: tl.constexpr,
        out0_stride_c: tl.constexpr,
        out0_stride_h: tl.constexpr,
        out0_stride_w: tl.constexpr,
        out2_stride_n: tl.constexpr,
        out2_stride_c: tl.constexpr,
        out2_stride_h: tl.constexpr,
        out2_stride_w: tl.constexpr,
        C_: tl.constexpr,
        W_: tl.constexpr,
        HW_: tl.constexpr,
        N_HW_: tl.constexpr,
        INV_HW_: tl.constexpr,
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

        mm_offsets = n_idx[:, None] * mm_stride_n + c_offsets[None, :] * mm_stride_c
        x1_offsets = (
            n_idx[:, None] * x1_stride_n
            + c_offsets[None, :] * x1_stride_c
            + h_idx[:, None] * x1_stride_h
            + w_idx[:, None] * x1_stride_w
        )
        x2_offsets = (
            n_idx[:, None] * x2_stride_n
            + c_offsets[None, :] * x2_stride_c
            + h_idx[:, None] * x2_stride_h
            + w_idx[:, None] * x2_stride_w
        )
        out0_offsets = (
            n_idx[:, None] * out0_stride_n
            + c_offsets[None, :] * out0_stride_c
            + h_idx[:, None] * out0_stride_h
            + w_idx[:, None] * out0_stride_w
        )
        out2_offsets = (
            n_idx[:, None] * out2_stride_n
            + c_offsets[None, :] * out2_stride_c
            + h_idx[:, None] * out2_stride_h
            + w_idx[:, None] * out2_stride_w
        )

        mean1 = tl.load(mean1_ptr + c_offsets * mean1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd1 = tl.load(invstd1_ptr + c_offsets * invstd1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight1 = tl.load(weight1_ptr + c_offsets * weight1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias1 = tl.load(bias1_ptr + c_offsets * bias1_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        mean2 = tl.load(mean2_ptr + c_offsets * mean2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        invstd2 = tl.load(invstd2_ptr + c_offsets * invstd2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        weight2 = tl.load(weight2_ptr + c_offsets * weight2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        bias2 = tl.load(bias2_ptr + c_offsets * bias2_stride_c, mask=c_offsets < C_, other=0.0).to(tl.float32)
        sum0 = tl.load(sum0_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        sum1 = tl.load(sum1_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)
        sum2 = tl.load(sum2_ptr + c_offsets, mask=c_offsets < C_, other=0.0).to(tl.float32)

        x1 = tl.load(x1_ptr + x1_offsets, mask=active, other=0.0).to(tl.float32)
        x2 = tl.load(x2_ptr + x2_offsets, mask=active, other=0.0).to(tl.float32)
        mm = tl.load(mm_ptr + mm_offsets, mask=active, other=0.0).to(tl.float32)

        centered1 = x1 - mean1[None, :]
        centered2 = x2 - mean2[None, :]
        affine1 = centered1 * invstd1[None, :] * weight1[None, :] + bias1[None, :]
        affine2 = centered2 * invstd2[None, :] * weight2[None, :] + bias2[None, :]
        affine = affine1 + affine2
        gate = ((affine > 0.0) | (affine != affine)) & active
        grad = tl.where(gate, mm * INV_HW_, 0.0)

        mean_term = sum0 * REDUCTION_SCALE_
        var1 = sum1 * REDUCTION_SCALE_ * invstd1 * invstd1
        var2 = sum2 * REDUCTION_SCALE_ * invstd2 * invstd2
        scale1 = invstd1 * weight1
        scale2 = invstd2 * weight2
        out0 = (grad - centered2 * var2[None, :] - mean_term[None, :]) * scale2[None, :]
        out2 = (grad - centered1 * var1[None, :] - mean_term[None, :]) * scale1[None, :]

        tl.store(out0_ptr + out0_offsets, out0, mask=active)
        tl.store(out2_ptr + out2_offsets, out2, mask=active)


def oracle_structured_pool_upsample_backward_reduce(
    mm: torch.Tensor,
    convolution_42: torch.Tensor,
    getitem_119: torch.Tensor,
    rsqrt_59: torch.Tensor,
    primals_343: torch.Tensor,
    primals_344: torch.Tensor,
    convolution_43: torch.Tensor,
    getitem_121: torch.Tensor,
    rsqrt_60: torch.Tensor,
    primals_349: torch.Tensor,
    primals_350: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del _shape_param_0, _shape_param_1
    if triton is None:
        raise RuntimeError("triton is not available")
    if mm.device.type != "cuda":
        raise RuntimeError("triton oracle requires CUDA inputs")

    out0 = torch.empty_strided(
        tuple(convolution_43.shape),
        tuple(convolution_43.stride()),
        device=convolution_43.device,
        dtype=convolution_43.dtype,
    )
    out1 = torch.empty((C,), device=convolution_43.device, dtype=convolution_43.dtype)
    out2 = torch.empty_strided(
        tuple(convolution_42.shape),
        tuple(convolution_42.stride()),
        device=convolution_42.device,
        dtype=convolution_42.dtype,
    )
    out3 = torch.empty((C,), device=convolution_42.device, dtype=convolution_42.dtype)

    partial0 = torch.empty((NUM_M_TILES, C), device=mm.device, dtype=torch.float32)
    partial1 = torch.empty_like(partial0)
    partial2 = torch.empty_like(partial0)
    sum0 = torch.empty((C,), device=mm.device, dtype=torch.float32)
    sum1 = torch.empty_like(sum0)
    sum2 = torch.empty_like(sum0)

    grid = (triton.cdiv(C, BLOCK_C), NUM_M_TILES)
    _partial_shared_gate_sums_kernel[grid](
        mm,
        convolution_42,
        getitem_119,
        rsqrt_59,
        primals_343,
        primals_344,
        convolution_43,
        getitem_121,
        rsqrt_60,
        primals_349,
        primals_350,
        partial0,
        partial1,
        partial2,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        x1_stride_n=convolution_42.stride(0),
        x1_stride_c=convolution_42.stride(1),
        x1_stride_h=convolution_42.stride(2),
        x1_stride_w=convolution_42.stride(3),
        x2_stride_n=convolution_43.stride(0),
        x2_stride_c=convolution_43.stride(1),
        x2_stride_h=convolution_43.stride(2),
        x2_stride_w=convolution_43.stride(3),
        mean1_stride_c=getitem_119.stride(1),
        invstd1_stride_c=rsqrt_59.stride(1),
        weight1_stride_c=primals_343.stride(0),
        bias1_stride_c=primals_344.stride(0),
        mean2_stride_c=getitem_121.stride(1),
        invstd2_stride_c=rsqrt_60.stride(1),
        weight2_stride_c=primals_349.stride(0),
        bias2_stride_c=primals_350.stride(0),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )

    _finalize_shared_gate_sums_kernel[(triton.cdiv(C, BLOCK_C),)](
        partial0,
        partial1,
        partial2,
        rsqrt_59,
        rsqrt_60,
        sum0,
        sum1,
        sum2,
        out1,
        out3,
        invstd1_stride_c=rsqrt_59.stride(1),
        invstd2_stride_c=rsqrt_60.stride(1),
        C_=C,
        NUM_M_TILES_=NUM_M_TILES,
        BLOCK_C_=BLOCK_C,
        BLOCK_TILES_=BLOCK_TILES,
        num_warps=1,
    )

    _shared_gate_bn_outputs_kernel[grid](
        mm,
        convolution_42,
        getitem_119,
        rsqrt_59,
        primals_343,
        primals_344,
        convolution_43,
        getitem_121,
        rsqrt_60,
        primals_349,
        primals_350,
        sum0,
        sum1,
        sum2,
        out0,
        out2,
        mm_stride_n=mm.stride(0),
        mm_stride_c=mm.stride(1),
        x1_stride_n=convolution_42.stride(0),
        x1_stride_c=convolution_42.stride(1),
        x1_stride_h=convolution_42.stride(2),
        x1_stride_w=convolution_42.stride(3),
        x2_stride_n=convolution_43.stride(0),
        x2_stride_c=convolution_43.stride(1),
        x2_stride_h=convolution_43.stride(2),
        x2_stride_w=convolution_43.stride(3),
        mean1_stride_c=getitem_119.stride(1),
        invstd1_stride_c=rsqrt_59.stride(1),
        weight1_stride_c=primals_343.stride(0),
        bias1_stride_c=primals_344.stride(0),
        mean2_stride_c=getitem_121.stride(1),
        invstd2_stride_c=rsqrt_60.stride(1),
        weight2_stride_c=primals_349.stride(0),
        bias2_stride_c=primals_350.stride(0),
        out0_stride_n=out0.stride(0),
        out0_stride_c=out0.stride(1),
        out0_stride_h=out0.stride(2),
        out0_stride_w=out0.stride(3),
        out2_stride_n=out2.stride(0),
        out2_stride_c=out2.stride(1),
        out2_stride_h=out2.stride(2),
        out2_stride_w=out2.stride(3),
        C_=C,
        W_=W,
        HW_=HW,
        N_HW_=N_HW,
        INV_HW_=INV_HW,
        REDUCTION_SCALE_=REDUCTION_SCALE,
        BLOCK_M_=BLOCK_M,
        BLOCK_C_=BLOCK_C,
        num_warps=4,
    )
    return out0, out1, out2, out3


def reference_outputs(inputs: tuple[object, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    module = _load_repro_module()
    if device.type != "cuda":
        module.device = lambda *unused_args, **unused_kwargs: device
    return module.Repro().to(device)(*inputs)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _max_diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-8)
    return diff.max().item(), rel.max().item()


def run_check(device: torch.device, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    with torch.no_grad():
        expected = reference_outputs(inputs, device)
        actual = oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)

    if len(actual) != len(expected):
        print(f"tuple length mismatch: actual={len(actual)} expected={len(expected)}")
        return False

    ok = True
    for idx, (got, ref) in enumerate(zip(actual, expected)):
        max_abs, max_rel = _max_diff(got, ref)
        shape_ok = got.shape == ref.shape
        dtype_ok = got.dtype == ref.dtype
        stride_ok = got.stride() == ref.stride()
        value_ok = torch.allclose(got.float(), ref.float(), rtol=rtol, atol=atol)
        ok = ok and shape_ok and dtype_ok and stride_ok and value_ok
        print(
            f"output[{idx}]: shape={list(got.shape)} dtype={got.dtype} "
            f"stride={got.stride()} expected_stride={ref.stride()} "
            f"max_abs={max_abs:.6e} max_rel={max_rel:.6e} "
            f"shape_match={shape_ok} dtype_match={dtype_ok} "
            f"stride_match={stride_ok} allclose={value_ok}"
        )

    print(f"Correctness: {'PASS' if ok else 'FAIL'}")
    return ok


def benchmark(fn: Callable[[], object], device: torch.device, warmup: int, rep: int) -> float:
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


def run_bench(device: torch.device, warmup: int, rep: int) -> None:
    torch.manual_seed(0)
    inputs = make_inputs(device)
    logical_read_bytes = (2 * N * C * HW + N * C) * 4
    logical_write_bytes = (2 * N * C * HW + 2 * C) * 4
    print(
        f"oracle shape: mm=f32[{N}, {C}], activations=f32[{N}, {C}, {H}, {W}] "
        f"shape={SHAPE_LABEL} device={device}"
    )
    print(f"direct logical traffic: {(logical_read_bytes + logical_write_bytes) / 1e6:.1f} MB")

    with torch.no_grad():
        oracle_structured_pool_upsample_backward_reduce(*inputs)
        synchronize(device)
        oracle_us = benchmark(
            lambda: oracle_structured_pool_upsample_backward_reduce(*inputs),
            device,
            warmup,
            rep,
        )
    print(
        f"oracle_structured_pool_upsample_backward_reduce: {oracle_us:.3f} us "
        f"impl=triton shape={SHAPE_LABEL} device={device} warmup={warmup} rep={rep}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare complete oracle outputs and strides to repro.py")
    parser.add_argument("--bench", action="store_true", help="time the Triton oracle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=50)
    args = parser.parse_args()

    if not args.check and not args.bench:
        parser.error("select at least one mode: --check and/or --bench")

    device = torch.device(args.device)
    if args.check and not run_check(device=device, rtol=args.rtol, atol=args.atol):
        sys.exit(1)
    if args.bench:
        run_bench(device=device, warmup=args.warmup, rep=args.rep)


if __name__ == "__main__":
    main()
