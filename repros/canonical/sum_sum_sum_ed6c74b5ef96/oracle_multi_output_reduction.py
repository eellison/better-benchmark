"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full Repro.forward return by inverting the low-memory max-pool-backward scatter-add from `add_53[:, 480:768]`/`arg336_1`, adding `getitem_198` and `getitem_201`, and feeding the four sibling BN/ReLU-backward channel groups to emit all eight tensor and channel-reduction outputs without materializing the `[128, 288, 35, 35]` scatter result, whereas Inductor currently materializes the scatter-add/view/channels-last clone/add chain and then schedules the four masked BN backward reductions and tensor epilogues as separate generic kernels; Inductor cannot do this today because its scheduler/codegen cannot represent a structured scatter-add producer as a recomputable source for multiple sibling channel reductions plus their dependent per-element epilogues; the fix is SCATTER_REDUCE: add guarded scatter-add producer fusion that sinks low-memory max-pool backward scatter reconstruction into downstream channel reductions and shares the produced channel summaries with the tensor epilogue kernels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

# Import shared oracle infrastructure (installed via pip install -e .)
from oracle_harness import (
    oracle_impl,
    get_inputs as _harness_get_inputs,
    get_repro_instance as _harness_get_repro_instance,
    check_oracle,
    bench_oracle,
    bench_oracle_all_shapes,
    get_hardware_info,
    has_stochastic_ops,
)


N = 128
CTOTAL = 288
C0 = 64
C1 = 64
C2 = 96
C3 = 64
ADD53_OFFSET_C = 480
ADD53_C = 768
POOL_H = 17
POOL_W = 17
POOL_HW = POOL_H * POOL_W
H = 35
W = 35
HW = H * W
SPATIAL = N * HW
REDUCE_SCALE = 1.0 / SPATIAL
CLASSIFICATION = "SCATTER_REDUCE"

DEFAULT_BLOCK_C = 4
DEFAULT_BLOCK_K = 256
DEFAULT_FINAL_BLOCK_C = 8
DEFAULT_EPILOGUE_BLOCK_K = 128


def get_inputs():
    """Load inputs from the repro's make_inputs (scope invariant: same inputs)."""
    return _harness_get_inputs(REPRO_DIR)


def get_repro_instance():
    """Create a Repro() instance for reference comparison."""
    return _harness_get_repro_instance(REPRO_DIR)


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for this oracle")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Triton oracle")


def _validate_inputs(inputs: tuple[object, ...] | list[object]) -> None:
    if len(inputs) != 28:
        raise ValueError(f"expected 28 repro inputs, got {len(inputs)}")
    tensors = inputs[:25]
    if any(not isinstance(x, torch.Tensor) for x in tensors):
        raise TypeError("first 25 repro inputs must be tensors")
    if any(x.device.type != "cuda" for x in tensors):
        raise RuntimeError("oracle expects CUDA tensors from make_inputs()")

    (
        add_53,
        arg336_1,
        getitem_198,
        getitem_201,
        arg320_1,
        arg321_1,
        arg322_1,
        arg65_1,
        arg66_1,
        full_1,
        arg316_1,
        arg317_1,
        arg318_1,
        arg62_1,
        arg63_1,
        arg307_1,
        arg308_1,
        arg309_1,
        arg55_1,
        arg56_1,
        arg301_1,
        arg302_1,
        arg303_1,
        arg50_1,
        arg51_1,
        shape0,
        shape1,
        shape2,
    ) = inputs

    expected = {
        "add_53": ((N, 768, POOL_H, POOL_W), torch.float32),
        "arg336_1": ((N, CTOTAL, POOL_H, POOL_W), torch.int8),
        "getitem_198": ((N, CTOTAL, H, W), torch.float32),
        "getitem_201": ((N, CTOTAL, H, W), torch.float32),
        "arg320_1": ((N, C3, H, W), torch.float32),
        "arg321_1": ((1, C3, 1, 1), torch.float32),
        "arg322_1": ((1, C3, 1, 1), torch.float32),
        "arg65_1": ((C3,), torch.float32),
        "arg66_1": ((C3,), torch.float32),
        "full_1": ((), torch.float32),
        "arg316_1": ((N, C2, H, W), torch.float32),
        "arg317_1": ((1, C2, 1, 1), torch.float32),
        "arg318_1": ((1, C2, 1, 1), torch.float32),
        "arg62_1": ((C2,), torch.float32),
        "arg63_1": ((C2,), torch.float32),
        "arg307_1": ((N, C1, H, W), torch.float32),
        "arg308_1": ((1, C1, 1, 1), torch.float32),
        "arg309_1": ((1, C1, 1, 1), torch.float32),
        "arg55_1": ((C1,), torch.float32),
        "arg56_1": ((C1,), torch.float32),
        "arg301_1": ((N, C0, H, W), torch.float32),
        "arg302_1": ((1, C0, 1, 1), torch.float32),
        "arg303_1": ((1, C0, 1, 1), torch.float32),
        "arg50_1": ((C0,), torch.float32),
        "arg51_1": ((C0,), torch.float32),
    }
    values = {
        "add_53": add_53,
        "arg336_1": arg336_1,
        "getitem_198": getitem_198,
        "getitem_201": getitem_201,
        "arg320_1": arg320_1,
        "arg321_1": arg321_1,
        "arg322_1": arg322_1,
        "arg65_1": arg65_1,
        "arg66_1": arg66_1,
        "full_1": full_1,
        "arg316_1": arg316_1,
        "arg317_1": arg317_1,
        "arg318_1": arg318_1,
        "arg62_1": arg62_1,
        "arg63_1": arg63_1,
        "arg307_1": arg307_1,
        "arg308_1": arg308_1,
        "arg309_1": arg309_1,
        "arg55_1": arg55_1,
        "arg56_1": arg56_1,
        "arg301_1": arg301_1,
        "arg302_1": arg302_1,
        "arg303_1": arg303_1,
        "arg50_1": arg50_1,
        "arg51_1": arg51_1,
    }
    for name, (shape, dtype) in expected.items():
        tensor = values[name]
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != dtype:
            raise ValueError(f"{name} dtype {tensor.dtype} != {dtype}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if tuple(shape0) != (N * CTOTAL, POOL_HW):
        raise ValueError(f"unexpected shape parameter 0: {shape0}")
    if tuple(shape1) != (N * CTOTAL, POOL_HW):
        raise ValueError(f"unexpected shape parameter 1: {shape1}")
    if tuple(shape2) != (N, CTOTAL, H, W):
        raise ValueError(f"unexpected shape parameter 2: {shape2}")


if triton is not None:

    # Preserve the repro's separate aten mul/add rounding before the ReLU mask.
    @triton.jit
    def _f32_mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _f32_add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            [a, b],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _scatter_candidate(
        add53_ptr,
        indices_ptr,
        n,
        c,
        h,
        w,
        oh,
        ow,
        active,
    ):
        kh = h - oh * 2
        kw = w - ow * 2
        valid = (
            active
            & (oh >= 0)
            & (oh < 17)
            & (ow >= 0)
            & (ow < 17)
            & (kh >= 0)
            & (kh < 3)
            & (kw >= 0)
            & (kw < 3)
        )
        safe_oh = tl.minimum(tl.maximum(oh, 0), 16)
        safe_ow = tl.minimum(tl.maximum(ow, 0), 16)
        local_offset = (kh * 3 + kw).to(tl.int32)
        index_offsets = n * (288 * 289) + c * 289 + safe_oh * 17 + safe_ow
        selected = (
            tl.load(indices_ptr + index_offsets, mask=valid, other=-1).to(tl.int32)
            == local_offset
        )
        add_offsets = (
            n * (768 * 289)
            + (480 + c) * 289
            + safe_oh * 17
            + safe_ow
        )
        return tl.load(add53_ptr + add_offsets, mask=valid & selected, other=0.0).to(
            tl.float32
        )

    @triton.jit
    def _load_fused_source(
        add53_ptr,
        indices_ptr,
        getitem198_ptr,
        getitem201_ptr,
        n,
        c,
        h,
        w,
        active,
    ):
        dense_offsets = n * (288 * 1225) + c * 1225 + h * 35 + w
        source = (
            tl.load(getitem198_ptr + dense_offsets, mask=active, other=0.0).to(
                tl.float32
            )
            + tl.load(getitem201_ptr + dense_offsets, mask=active, other=0.0).to(
                tl.float32
            )
        )

        oh0 = h // 2
        ow0 = w // 2
        oh1 = oh0 - 1
        ow1 = ow0 - 1
        source += _scatter_candidate(add53_ptr, indices_ptr, n, c, h, w, oh0, ow0, active)
        source += _scatter_candidate(add53_ptr, indices_ptr, n, c, h, w, oh0, ow1, active)
        source += _scatter_candidate(add53_ptr, indices_ptr, n, c, h, w, oh1, ow0, active)
        source += _scatter_candidate(add53_ptr, indices_ptr, n, c, h, w, oh1, ow1, active)
        return source

    @triton.jit
    def _load_branch_activation(
        arg301_ptr,
        arg307_ptr,
        arg316_ptr,
        arg320_ptr,
        n,
        c,
        spatial,
        active,
    ):
        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        low_safe = tl.where((c >= 0) & (c < 64), low, 0)
        mid_safe = tl.where((c >= 64) & (c < 128), mid, 0)
        wide_safe = tl.where((c >= 128) & (c < 224), wide, 0)
        high_safe = tl.where(c >= 224, high, 0)

        low_mask = active & (c < 64)
        mid_mask = active & (c >= 64) & (c < 128)
        wide_mask = active & (c >= 128) & (c < 224)
        high_mask = active & (c >= 224)

        return (
            tl.load(
                arg301_ptr + n * (64 * 1225) + low_safe * 1225 + spatial,
                mask=low_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg307_ptr + n * (64 * 1225) + mid_safe * 1225 + spatial,
                mask=mid_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg316_ptr + n * (96 * 1225) + wide_safe * 1225 + spatial,
                mask=wide_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg320_ptr + n * (64 * 1225) + high_safe * 1225 + spatial,
                mask=high_mask,
                other=0.0,
            ).to(tl.float32)
        )

    @triton.jit
    def _load_branch_vector(
        low_ptr,
        mid_ptr,
        wide_ptr,
        high_ptr,
        c,
        c_mask,
    ):
        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        low_safe = tl.where((c >= 0) & (c < 64), low, 0)
        mid_safe = tl.where((c >= 64) & (c < 128), mid, 0)
        wide_safe = tl.where((c >= 128) & (c < 224), wide, 0)
        high_safe = tl.where(c >= 224, high, 0)

        low_mask = c_mask & (c < 64)
        mid_mask = c_mask & (c >= 64) & (c < 128)
        wide_mask = c_mask & (c >= 128) & (c < 224)
        high_mask = c_mask & (c >= 224)

        return (
            tl.load(low_ptr + low_safe, mask=low_mask, other=0.0).to(tl.float32)
            + tl.load(mid_ptr + mid_safe, mask=mid_mask, other=0.0).to(tl.float32)
            + tl.load(wide_ptr + wide_safe, mask=wide_mask, other=0.0).to(tl.float32)
            + tl.load(high_ptr + high_safe, mask=high_mask, other=0.0).to(tl.float32)
        )

    @triton.jit
    def _partial_reduction_kernel(
        add53_ptr,
        indices_ptr,
        getitem198_ptr,
        getitem201_ptr,
        full_ptr,
        arg301_ptr,
        arg302_ptr,
        arg303_ptr,
        arg50_ptr,
        arg51_ptr,
        arg307_ptr,
        arg308_ptr,
        arg309_ptr,
        arg55_ptr,
        arg56_ptr,
        arg316_ptr,
        arg317_ptr,
        arg318_ptr,
        arg62_ptr,
        arg63_ptr,
        arg320_ptr,
        arg321_ptr,
        arg322_ptr,
        arg65_ptr,
        arg66_ptr,
        partial_sum0_ptr,
        partial_sum1_ptr,
        num_tiles: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        c_mask = c < 288
        k_mask = k < 156800
        n = k // 1225
        spatial = k - n * 1225
        h = spatial // 35
        w = spatial - h * 35
        active = k_mask[:, None] & c_mask[None, :]

        c_mat = c[None, :]
        n_mat = n[:, None]
        h_mat = h[:, None]
        w_mat = w[:, None]
        spatial_mat = spatial[:, None]

        source = _load_fused_source(
            add53_ptr,
            indices_ptr,
            getitem198_ptr,
            getitem201_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            active,
        )
        x = _load_branch_activation(
            arg301_ptr,
            arg307_ptr,
            arg316_ptr,
            arg320_ptr,
            n_mat,
            c_mat,
            spatial_mat,
            active,
        )

        mean = _load_branch_vector(arg302_ptr, arg308_ptr, arg317_ptr, arg321_ptr, c, c_mask)
        invstd = _load_branch_vector(arg303_ptr, arg309_ptr, arg318_ptr, arg322_ptr, c, c_mask)
        gamma = _load_branch_vector(arg50_ptr, arg55_ptr, arg62_ptr, arg65_ptr, c, c_mask)
        beta = _load_branch_vector(arg51_ptr, arg56_ptr, arg63_ptr, arg66_ptr, c, c_mask)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        normalized = _f32_mul(centered, invstd[None, :])
        scaled = _f32_mul(normalized, gamma[None, :])
        relu_input = _f32_add(scaled, beta[None, :])
        where_value = tl.where(relu_input <= 0.0, fill, source)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)

        sum0 = tl.sum(where_value, axis=0)
        sum1 = tl.sum(where_value * centered, axis=0)
        partial_offsets = c * num_tiles + tl.program_id(1)
        tl.store(partial_sum0_ptr + partial_offsets, sum0, mask=c_mask)
        tl.store(partial_sum1_ptr + partial_offsets, sum1, mask=c_mask)

    @triton.jit
    def _finalize_reductions_kernel(
        partial_sum0_ptr,
        partial_sum1_ptr,
        arg303_ptr,
        arg309_ptr,
        arg318_ptr,
        arg322_ptr,
        sum0_ptr,
        sum1_ptr,
        out1_ptr,
        out3_ptr,
        out5_ptr,
        out7_ptr,
        num_tiles: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_TILES: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        tiles = tl.arange(0, BLOCK_TILES)
        c_mask = c < 288
        tile_mask = tiles < num_tiles
        offsets = tiles[:, None] + c[None, :] * num_tiles
        active = tile_mask[:, None] & c_mask[None, :]

        partial0 = tl.load(partial_sum0_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        partial1 = tl.load(partial_sum1_ptr + offsets, mask=active, other=0.0).to(
            tl.float32
        )
        total0 = tl.sum(partial0, axis=0)
        total1 = tl.sum(partial1, axis=0)
        invstd = _load_branch_vector(arg303_ptr, arg309_ptr, arg318_ptr, arg322_ptr, c, c_mask)
        vec = total1 * invstd

        tl.store(sum0_ptr + c, total0, mask=c_mask)
        tl.store(sum1_ptr + c, total1, mask=c_mask)

        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        tl.store(out7_ptr + low, vec, mask=c_mask & (c < 64))
        tl.store(out5_ptr + mid, vec, mask=c_mask & (c >= 64) & (c < 128))
        tl.store(
            out3_ptr + wide,
            vec,
            mask=c_mask & (c >= 128) & (c < 224),
        )
        tl.store(out1_ptr + high, vec, mask=c_mask & (c >= 224))

    @triton.jit
    def _epilogue_kernel(
        add53_ptr,
        indices_ptr,
        getitem198_ptr,
        getitem201_ptr,
        full_ptr,
        arg301_ptr,
        arg302_ptr,
        arg303_ptr,
        arg50_ptr,
        arg51_ptr,
        arg307_ptr,
        arg308_ptr,
        arg309_ptr,
        arg55_ptr,
        arg56_ptr,
        arg316_ptr,
        arg317_ptr,
        arg318_ptr,
        arg62_ptr,
        arg63_ptr,
        arg320_ptr,
        arg321_ptr,
        arg322_ptr,
        arg65_ptr,
        arg66_ptr,
        sum0_ptr,
        sum1_ptr,
        out0_ptr,
        out2_ptr,
        out4_ptr,
        out6_ptr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        c_mask = c < 288
        k_mask = k < 156800
        n = k // 1225
        spatial = k - n * 1225
        h = spatial // 35
        w = spatial - h * 35
        active = k_mask[:, None] & c_mask[None, :]

        c_mat = c[None, :]
        n_mat = n[:, None]
        h_mat = h[:, None]
        w_mat = w[:, None]
        spatial_mat = spatial[:, None]

        source = _load_fused_source(
            add53_ptr,
            indices_ptr,
            getitem198_ptr,
            getitem201_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            active,
        )
        x = _load_branch_activation(
            arg301_ptr,
            arg307_ptr,
            arg316_ptr,
            arg320_ptr,
            n_mat,
            c_mat,
            spatial_mat,
            active,
        )

        mean = _load_branch_vector(arg302_ptr, arg308_ptr, arg317_ptr, arg321_ptr, c, c_mask)
        invstd = _load_branch_vector(arg303_ptr, arg309_ptr, arg318_ptr, arg322_ptr, c, c_mask)
        gamma = _load_branch_vector(arg50_ptr, arg55_ptr, arg62_ptr, arg65_ptr, c, c_mask)
        beta = _load_branch_vector(arg51_ptr, arg56_ptr, arg63_ptr, arg66_ptr, c, c_mask)
        total0 = tl.load(sum0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        total1 = tl.load(sum1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        normalized = _f32_mul(centered, invstd[None, :])
        scaled = _f32_mul(normalized, gamma[None, :])
        relu_input = _f32_add(scaled, beta[None, :])
        where_value = tl.where(relu_input <= 0.0, fill, source)
        variance_term = (
            total1[None, :] * 0.0000063775510204081635 * invstd[None, :] * invstd[None, :]
        )
        mean_term = total0[None, :] * 0.0000063775510204081635
        out = (
            where_value
            - centered * variance_term
            - mean_term
        ) * (invstd[None, :] * gamma[None, :])

        low = c_mat
        mid = c_mat - 64
        wide = c_mat - 128
        high = c_mat - 224
        out_low_offsets = n_mat * (64 * 1225) + low * 1225 + spatial_mat
        out_mid_offsets = n_mat * (64 * 1225) + mid * 1225 + spatial_mat
        out_wide_offsets = n_mat * (96 * 1225) + wide * 1225 + spatial_mat
        out_high_offsets = n_mat * (64 * 1225) + high * 1225 + spatial_mat

        tl.store(out6_ptr + out_low_offsets, out, mask=active & (c_mat < 64))
        tl.store(
            out4_ptr + out_mid_offsets,
            out,
            mask=active & (c_mat >= 64) & (c_mat < 128),
        )
        tl.store(
            out2_ptr + out_wide_offsets,
            out,
            mask=active & (c_mat >= 128) & (c_mat < 224),
        )
        tl.store(out0_ptr + out_high_offsets, out, mask=active & (c_mat >= 224))


def _oracle_fused(
    add_53: torch.Tensor,
    arg336_1: torch.Tensor,
    getitem_198: torch.Tensor,
    getitem_201: torch.Tensor,
    arg320_1: torch.Tensor,
    arg321_1: torch.Tensor,
    arg322_1: torch.Tensor,
    arg65_1: torch.Tensor,
    arg66_1: torch.Tensor,
    full_1: torch.Tensor,
    arg316_1: torch.Tensor,
    arg317_1: torch.Tensor,
    arg318_1: torch.Tensor,
    arg62_1: torch.Tensor,
    arg63_1: torch.Tensor,
    arg307_1: torch.Tensor,
    arg308_1: torch.Tensor,
    arg309_1: torch.Tensor,
    arg55_1: torch.Tensor,
    arg56_1: torch.Tensor,
    arg301_1: torch.Tensor,
    arg302_1: torch.Tensor,
    arg303_1: torch.Tensor,
    arg50_1: torch.Tensor,
    arg51_1: torch.Tensor,
    _shape_param_0,
    _shape_param_1,
    _shape_param_2,
    *,
    block_c: int = DEFAULT_BLOCK_C,
    block_k: int = DEFAULT_BLOCK_K,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
    epilogue_block_k: int = DEFAULT_EPILOGUE_BLOCK_K,
) -> tuple[torch.Tensor, ...]:
    del _shape_param_0, _shape_param_1, _shape_param_2
    _require_triton()

    device = add_53.device
    num_tiles = triton.cdiv(SPATIAL, block_k)
    partial_sum0 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    partial_sum1 = torch.empty((CTOTAL, num_tiles), device=device, dtype=torch.float32)
    sum0 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)
    sum1 = torch.empty((CTOTAL,), device=device, dtype=torch.float32)

    out0 = torch.empty((N, C3, H, W), device=device, dtype=torch.float32)
    out1 = torch.empty((C3,), device=device, dtype=torch.float32)
    out2 = torch.empty((N, C2, H, W), device=device, dtype=torch.float32)
    out3 = torch.empty((C2,), device=device, dtype=torch.float32)
    out4 = torch.empty((N, C1, H, W), device=device, dtype=torch.float32)
    out5 = torch.empty((C1,), device=device, dtype=torch.float32)
    out6 = torch.empty((N, C0, H, W), device=device, dtype=torch.float32)
    out7 = torch.empty((C0,), device=device, dtype=torch.float32)

    _partial_reduction_kernel[(triton.cdiv(CTOTAL, block_c), num_tiles)](
        add_53,
        arg336_1,
        getitem_198,
        getitem_201,
        full_1,
        arg301_1,
        arg302_1,
        arg303_1,
        arg50_1,
        arg51_1,
        arg307_1,
        arg308_1,
        arg309_1,
        arg55_1,
        arg56_1,
        arg316_1,
        arg317_1,
        arg318_1,
        arg62_1,
        arg63_1,
        arg320_1,
        arg321_1,
        arg322_1,
        arg65_1,
        arg66_1,
        partial_sum0,
        partial_sum1,
        num_tiles=num_tiles,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=8,
    )
    _finalize_reductions_kernel[(triton.cdiv(CTOTAL, final_block_c),)](
        partial_sum0,
        partial_sum1,
        arg303_1,
        arg309_1,
        arg318_1,
        arg322_1,
        sum0,
        sum1,
        out1,
        out3,
        out5,
        out7,
        num_tiles=num_tiles,
        BLOCK_C=final_block_c,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(CTOTAL, block_c), triton.cdiv(SPATIAL, epilogue_block_k))](
        add_53,
        arg336_1,
        getitem_198,
        getitem_201,
        full_1,
        arg301_1,
        arg302_1,
        arg303_1,
        arg50_1,
        arg51_1,
        arg307_1,
        arg308_1,
        arg309_1,
        arg55_1,
        arg56_1,
        arg316_1,
        arg317_1,
        arg318_1,
        arg62_1,
        arg63_1,
        arg320_1,
        arg321_1,
        arg322_1,
        arg65_1,
        arg66_1,
        sum0,
        sum1,
        out0,
        out2,
        out4,
        out6,
        BLOCK_C=block_c,
        BLOCK_K=epilogue_block_k,
        num_warps=8,
    )

    return out0, out1, out2, out3, out4, out5, out6, out7


@oracle_impl(hardware="H100", shapes="(T([128, 768, 17, 17], f32), T([128, 288, 17, 17], i8, gen=Index(9)), T([128, 288, 35, 35], f32), T([128, 288, 35, 35], f32), T([128, 64, 35, 35], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([], f32), T([128, 96, 35, 35], f32), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([128, 64, 35, 35], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 35, 35], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), S([36864, 289]), S([36864, 289]), S([128, 288, 35, 35]))")
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
    _validate_inputs(inputs)
    return _oracle_fused(*inputs)


def _normalize_outputs(out) -> list[torch.Tensor]:
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (tuple, list)):
        return [x for x in out if isinstance(x, torch.Tensor)]
    return []


def _strict_scope_check(instance, inputs) -> bool:
    with torch.no_grad():
        eager = instance(*inputs)
        actual = oracle_forward(inputs)
        torch.cuda.synchronize()

    eager_list = _normalize_outputs(eager)
    actual_list = _normalize_outputs(actual)
    if len(eager_list) != len(actual_list):
        print(
            f"  strict scope: FAIL output_count oracle={len(actual_list)} "
            f"eager={len(eager_list)}"
        )
        return False

    ok = True
    for idx, (got, expected) in enumerate(zip(actual_list, eager_list)):
        shape_ok = got.shape == expected.shape
        dtype_ok = got.dtype == expected.dtype
        stride_ok = got.stride() == expected.stride()
        if shape_ok:
            if got.is_floating_point() and expected.is_floating_point():
                max_diff = (got.float() - expected.float()).abs().max().item()
            else:
                max_diff = 0.0 if torch.equal(got, expected) else float("inf")
        else:
            max_diff = float("nan")
        item_ok = shape_ok and dtype_ok and stride_ok
        ok = ok and item_ok
        print(
            f"  strict scope output {idx}: shape_match={shape_ok} "
            f"dtype_match={dtype_ok} stride_match={stride_ok} "
            f"max_diff={max_diff:.2e} "
            f"expected_stride={tuple(expected.stride())} "
            f"oracle_stride={tuple(got.stride())}"
        )
    print(f"  strict scope: {'PASS' if ok else 'FAIL'}")
    return ok


def _annotate_result(result: dict[str, object]) -> dict[str, object]:
    oracle_us = float(result["oracle_us"])
    compile_us = float(result["compile_us"])
    true_floor = oracle_us < compile_us
    annotated = dict(result)
    annotated.update(
        {
            "classification": CLASSIFICATION,
            "true_floor": true_floor,
            "actionable": result.get("status") == "GOOD",
            "scope": "full_repro_forward",
            "timing": "oracle_harness.bench_oracle",
        }
    )
    return annotated


# --- CLI entry point ---
def main():
    parser = argparse.ArgumentParser(
        description=f"Oracle for {REPRO_ID}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true",
                        help="Verify correctness against eager Repro")
    parser.add_argument("--bench", action="store_true",
                        help="Benchmark oracle vs torch.compile via oracle_harness.bench_oracle")
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
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

    # Report if stochastic ops detected in source
    source_has_stochastic_ops = has_stochastic_ops(REPRO_PATH)
    if source_has_stochastic_ops:
        print(f"NOTE: {REPRO_ID} contains stochastic ops; affected outputs will be auto-skipped")

    if args.check:
        print(f"Checking {REPRO_ID}...")
        ok = check_oracle(
            oracle_forward,
            instance,
            inputs,
            atol=args.atol,
            rtol=args.rtol,
            skip_stochastic=source_has_stochastic_ops and not args.no_skip_stochastic,
        )
        ok = _strict_scope_check(instance, inputs) and ok
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
                print(json.dumps(_annotate_result(result), sort_keys=True))
                if result["status"] == "BAD_ORACLE":
                    print(f"WARNING: oracle is slower than compile for "
                          f"{result['repro_id']} (ratio={result['ratio']:.3f}x)")
        else:
            result = bench_oracle(
                oracle_forward,
                instance,
                inputs,
                REPRO_ID,
                warmup=args.warmup,
                rep=args.rep,
            )
            print(json.dumps(_annotate_result(result), sort_keys=True))
            if result["status"] == "BAD_ORACLE":
                print(f"WARNING: oracle is slower than compile "
                      f"(ratio={result['ratio']:.3f}x)")


if __name__ == "__main__":
    main()
