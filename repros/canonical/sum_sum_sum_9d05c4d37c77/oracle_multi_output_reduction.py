"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete eight-output Inception backward scope from Repro.forward by recomputing the avg-pool-backward plus three-add shared producer inside the four channel-split BN/ReLU-backward reductions and their dependent tensor epilogues, returning the four input-gradient tensors and four scale-gradient vectors with the eager layouts, whereas Inductor currently materializes the avg-pool/add/slice/where producers and schedules the sibling channel reductions plus tensor epilogues as separate generic kernels; Inductor cannot do this today because the scheduler/codegen cannot keep a shared stencil producer virtual across multiple disjoint channel reductions and reuse the finalized channel summaries inside their full-tensor epilogues; the fix is SCHEDULER_FUSION: add a multi-output reduction schedule that fuses shared stencil/pointwise producers into same-shape channel reductions and emits the dependent BN-backward epilogue stores without materializing the slice/where intermediates."""
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
CTOTAL = 256
C0 = 64
C1 = 64
C2 = 96
C3 = 32
H = 35
W = 35
HW = H * W
SPATIAL = N * HW

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
    if len(inputs) != 26:
        raise ValueError(f"expected 26 repro inputs, got {len(inputs)}")
    if any(not isinstance(x, torch.Tensor) for x in inputs):
        raise TypeError("all repro inputs must be tensors")
    if any(x.device.type != "cuda" for x in inputs):
        raise RuntimeError("oracle expects CUDA tensors from make_inputs()")

    expected = [
        ("getitem_225", (N, CTOTAL, H, W)),
        ("arg277_1", (N, CTOTAL, H, W)),
        ("getitem_234", (N, CTOTAL, H, W)),
        ("getitem_240", (N, CTOTAL, H, W)),
        ("getitem_243", (N, CTOTAL, H, W)),
        ("arg274_1", (N, C3, H, W)),
        ("arg275_1", (1, C3, 1, 1)),
        ("arg276_1", (1, C3, 1, 1)),
        ("arg29_1", (C3,)),
        ("arg30_1", (C3,)),
        ("full_1", ()),
        ("arg270_1", (N, C2, H, W)),
        ("arg271_1", (1, C2, 1, 1)),
        ("arg272_1", (1, C2, 1, 1)),
        ("arg26_1", (C2,)),
        ("arg27_1", (C2,)),
        ("arg261_1", (N, C1, H, W)),
        ("arg262_1", (1, C1, 1, 1)),
        ("arg263_1", (1, C1, 1, 1)),
        ("arg19_1", (C1,)),
        ("arg20_1", (C1,)),
        ("arg255_1", (N, C0, H, W)),
        ("arg256_1", (1, C0, 1, 1)),
        ("arg257_1", (1, C0, 1, 1)),
        ("arg14_1", (C0,)),
        ("arg15_1", (C0,)),
    ]
    for tensor, (name, shape) in zip(inputs, expected):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name} shape {tuple(tensor.shape)} != {shape}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} dtype {tensor.dtype} != torch.float32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


if triton is not None:

    @triton.jit
    def _load_avg_pool_backward(
        grad_ptr,
        n,
        c,
        h,
        w,
        active,
    ):
        base = n * (256 * 1225) + c * 1225
        hm = h > 0
        hp = h < 34
        wm = w > 0
        wp = w < 34

        center = base + h * 35 + w
        total = tl.load(grad_ptr + center, mask=active, other=0.0).to(tl.float32)
        total += tl.load(
            grad_ptr + (h - 1) * 35 + w + base,
            mask=active & hm,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + (h + 1) * 35 + w + base,
            mask=active & hp,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + h * 35 + (w - 1) + base,
            mask=active & wm,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + h * 35 + (w + 1) + base,
            mask=active & wp,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + (h - 1) * 35 + (w - 1) + base,
            mask=active & hm & wm,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + (h - 1) * 35 + (w + 1) + base,
            mask=active & hm & wp,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + (h + 1) * 35 + (w - 1) + base,
            mask=active & hp & wm,
            other=0.0,
        ).to(tl.float32)
        total += tl.load(
            grad_ptr + (h + 1) * 35 + (w + 1) + base,
            mask=active & hp & wp,
            other=0.0,
        ).to(tl.float32)
        return total * 0.1111111111111111

    @triton.jit
    def _load_fused_source(
        getitem225_ptr,
        getitem234_ptr,
        getitem240_ptr,
        getitem243_ptr,
        n,
        c,
        h,
        w,
        spatial,
        active,
    ):
        offsets = n * (256 * 1225) + c * 1225 + spatial
        source = _load_avg_pool_backward(getitem225_ptr, n, c, h, w, active)
        source += tl.load(getitem234_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source += tl.load(getitem240_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        source += tl.load(getitem243_ptr + offsets, mask=active, other=0.0).to(tl.float32)
        return source

    @triton.jit
    def _load_branch_activation(
        arg255_ptr,
        arg261_ptr,
        arg270_ptr,
        arg274_ptr,
        n,
        c,
        spatial,
        active,
    ):
        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        low_safe = tl.where(c < 64, low, 0)
        mid_safe = tl.where((c >= 64) & (c < 128), mid, 0)
        wide_safe = tl.where((c >= 128) & (c < 224), wide, 0)
        high_safe = tl.where(c >= 224, high, 0)

        low_mask = active & (c < 64)
        mid_mask = active & (c >= 64) & (c < 128)
        wide_mask = active & (c >= 128) & (c < 224)
        high_mask = active & (c >= 224)

        return (
            tl.load(
                arg255_ptr + n * (64 * 1225) + low_safe * 1225 + spatial,
                mask=low_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg261_ptr + n * (64 * 1225) + mid_safe * 1225 + spatial,
                mask=mid_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg270_ptr + n * (96 * 1225) + wide_safe * 1225 + spatial,
                mask=wide_mask,
                other=0.0,
            ).to(tl.float32)
            + tl.load(
                arg274_ptr + n * (32 * 1225) + high_safe * 1225 + spatial,
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
        low_safe = tl.where(c < 64, low, 0)
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
        getitem225_ptr,
        getitem234_ptr,
        getitem240_ptr,
        getitem243_ptr,
        full_ptr,
        arg255_ptr,
        arg256_ptr,
        arg257_ptr,
        arg14_ptr,
        arg15_ptr,
        arg261_ptr,
        arg262_ptr,
        arg263_ptr,
        arg19_ptr,
        arg20_ptr,
        arg270_ptr,
        arg271_ptr,
        arg272_ptr,
        arg26_ptr,
        arg27_ptr,
        arg274_ptr,
        arg275_ptr,
        arg276_ptr,
        arg29_ptr,
        arg30_ptr,
        partial_sum0_ptr,
        partial_sum1_ptr,
        num_tiles: tl.constexpr,
        BLOCK_C: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
        k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        c_mask = c < 256
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
            getitem225_ptr,
            getitem234_ptr,
            getitem240_ptr,
            getitem243_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            spatial_mat,
            active,
        )
        x = _load_branch_activation(
            arg255_ptr,
            arg261_ptr,
            arg270_ptr,
            arg274_ptr,
            n_mat,
            c_mat,
            spatial_mat,
            active,
        )

        mean = _load_branch_vector(arg256_ptr, arg262_ptr, arg271_ptr, arg275_ptr, c, c_mask)
        invstd = _load_branch_vector(arg257_ptr, arg263_ptr, arg272_ptr, arg276_ptr, c, c_mask)
        gamma = _load_branch_vector(arg14_ptr, arg19_ptr, arg26_ptr, arg29_ptr, c, c_mask)
        beta = _load_branch_vector(arg15_ptr, arg20_ptr, arg27_ptr, arg30_ptr, c, c_mask)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        relu_input = centered * invstd[None, :] * gamma[None, :] + beta[None, :]
        where_value = tl.where(relu_input <= 0.0, fill, source)
        where_value = tl.where(active, where_value, 0.0)
        centered = tl.where(active, centered, 0.0)

        partial_offsets = c * num_tiles + tl.program_id(1)
        tl.store(partial_sum0_ptr + partial_offsets, tl.sum(where_value, axis=0), mask=c_mask)
        tl.store(
            partial_sum1_ptr + partial_offsets,
            tl.sum(where_value * centered, axis=0),
            mask=c_mask,
        )

    @triton.jit
    def _finalize_reductions_kernel(
        partial_sum0_ptr,
        partial_sum1_ptr,
        arg257_ptr,
        arg263_ptr,
        arg272_ptr,
        arg276_ptr,
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
        c_mask = c < 256
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
        invstd = _load_branch_vector(arg257_ptr, arg263_ptr, arg272_ptr, arg276_ptr, c, c_mask)
        vec = total1 * invstd

        tl.store(sum0_ptr + c, total0, mask=c_mask)
        tl.store(sum1_ptr + c, total1, mask=c_mask)

        low = c
        mid = c - 64
        wide = c - 128
        high = c - 224
        tl.store(out7_ptr + low, vec, mask=c_mask & (c < 64))
        tl.store(out5_ptr + mid, vec, mask=c_mask & (c >= 64) & (c < 128))
        tl.store(out3_ptr + wide, vec, mask=c_mask & (c >= 128) & (c < 224))
        tl.store(out1_ptr + high, vec, mask=c_mask & (c >= 224))

    @triton.jit
    def _epilogue_kernel(
        getitem225_ptr,
        getitem234_ptr,
        getitem240_ptr,
        getitem243_ptr,
        full_ptr,
        arg255_ptr,
        arg256_ptr,
        arg257_ptr,
        arg14_ptr,
        arg15_ptr,
        arg261_ptr,
        arg262_ptr,
        arg263_ptr,
        arg19_ptr,
        arg20_ptr,
        arg270_ptr,
        arg271_ptr,
        arg272_ptr,
        arg26_ptr,
        arg27_ptr,
        arg274_ptr,
        arg275_ptr,
        arg276_ptr,
        arg29_ptr,
        arg30_ptr,
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
        c_mask = c < 256
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
            getitem225_ptr,
            getitem234_ptr,
            getitem240_ptr,
            getitem243_ptr,
            n_mat,
            c_mat,
            h_mat,
            w_mat,
            spatial_mat,
            active,
        )
        x = _load_branch_activation(
            arg255_ptr,
            arg261_ptr,
            arg270_ptr,
            arg274_ptr,
            n_mat,
            c_mat,
            spatial_mat,
            active,
        )

        mean = _load_branch_vector(arg256_ptr, arg262_ptr, arg271_ptr, arg275_ptr, c, c_mask)
        invstd = _load_branch_vector(arg257_ptr, arg263_ptr, arg272_ptr, arg276_ptr, c, c_mask)
        gamma = _load_branch_vector(arg14_ptr, arg19_ptr, arg26_ptr, arg29_ptr, c, c_mask)
        beta = _load_branch_vector(arg15_ptr, arg20_ptr, arg27_ptr, arg30_ptr, c, c_mask)
        total0 = tl.load(sum0_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        total1 = tl.load(sum1_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
        fill = tl.load(full_ptr).to(tl.float32)

        centered = x - mean[None, :]
        relu_input = centered * invstd[None, :] * gamma[None, :] + beta[None, :]
        where_value = tl.where(relu_input <= 0.0, fill, source)
        variance_term = total1[None, :] * 0.0000063775510204081635 * invstd[None, :] * invstd[None, :]
        mean_term = total0[None, :] * 0.0000063775510204081635
        out = (where_value - centered * variance_term - mean_term) * (
            invstd[None, :] * gamma[None, :]
        )

        low = c_mat
        mid = c_mat - 64
        wide = c_mat - 128
        high = c_mat - 224
        low_offsets = n_mat * (64 * 1225) + low * 1225 + spatial_mat
        mid_offsets = n_mat * (64 * 1225) + mid * 1225 + spatial_mat
        wide_offsets = n_mat * (96 * 1225) + wide * 1225 + spatial_mat
        high_offsets = n_mat * (32 * 1225) + high * 1225 + spatial_mat

        tl.store(out6_ptr + low_offsets, out, mask=active & (c_mat < 64))
        tl.store(out4_ptr + mid_offsets, out, mask=active & (c_mat >= 64) & (c_mat < 128))
        tl.store(out2_ptr + wide_offsets, out, mask=active & (c_mat >= 128) & (c_mat < 224))
        tl.store(out0_ptr + high_offsets, out, mask=active & (c_mat >= 224))


def _oracle_fused(
    getitem_225: torch.Tensor,
    _arg277_1: torch.Tensor,
    getitem_234: torch.Tensor,
    getitem_240: torch.Tensor,
    getitem_243: torch.Tensor,
    arg274_1: torch.Tensor,
    arg275_1: torch.Tensor,
    arg276_1: torch.Tensor,
    arg29_1: torch.Tensor,
    arg30_1: torch.Tensor,
    full_1: torch.Tensor,
    arg270_1: torch.Tensor,
    arg271_1: torch.Tensor,
    arg272_1: torch.Tensor,
    arg26_1: torch.Tensor,
    arg27_1: torch.Tensor,
    arg261_1: torch.Tensor,
    arg262_1: torch.Tensor,
    arg263_1: torch.Tensor,
    arg19_1: torch.Tensor,
    arg20_1: torch.Tensor,
    arg255_1: torch.Tensor,
    arg256_1: torch.Tensor,
    arg257_1: torch.Tensor,
    arg14_1: torch.Tensor,
    arg15_1: torch.Tensor,
    *,
    block_c: int = DEFAULT_BLOCK_C,
    block_k: int = DEFAULT_BLOCK_K,
    final_block_c: int = DEFAULT_FINAL_BLOCK_C,
    epilogue_block_k: int = DEFAULT_EPILOGUE_BLOCK_K,
) -> tuple[torch.Tensor, ...]:
    _require_triton()

    device = getitem_225.device
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
        getitem_225,
        getitem_234,
        getitem_240,
        getitem_243,
        full_1,
        arg255_1,
        arg256_1,
        arg257_1,
        arg14_1,
        arg15_1,
        arg261_1,
        arg262_1,
        arg263_1,
        arg19_1,
        arg20_1,
        arg270_1,
        arg271_1,
        arg272_1,
        arg26_1,
        arg27_1,
        arg274_1,
        arg275_1,
        arg276_1,
        arg29_1,
        arg30_1,
        partial_sum0,
        partial_sum1,
        num_tiles=num_tiles,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=4,
    )
    _finalize_reductions_kernel[(triton.cdiv(CTOTAL, final_block_c),)](
        partial_sum0,
        partial_sum1,
        arg257_1,
        arg263_1,
        arg272_1,
        arg276_1,
        sum0,
        sum1,
        out1,
        out3,
        out5,
        out7,
        num_tiles=num_tiles,
        BLOCK_C=final_block_c,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=4,
    )
    _epilogue_kernel[(triton.cdiv(CTOTAL, block_c), triton.cdiv(SPATIAL, epilogue_block_k))](
        getitem_225,
        getitem_234,
        getitem_240,
        getitem_243,
        full_1,
        arg255_1,
        arg256_1,
        arg257_1,
        arg14_1,
        arg15_1,
        arg261_1,
        arg262_1,
        arg263_1,
        arg19_1,
        arg20_1,
        arg270_1,
        arg271_1,
        arg272_1,
        arg26_1,
        arg27_1,
        arg274_1,
        arg275_1,
        arg276_1,
        arg29_1,
        arg30_1,
        sum0,
        sum1,
        out0,
        out2,
        out4,
        out6,
        BLOCK_C=block_c,
        BLOCK_K=epilogue_block_k,
        num_warps=4,
    )

    return out0, out1, out2, out3, out4, out5, out6, out7


@oracle_impl(hardware="H100", shapes="(T([128, 256, 35, 35], f32), T([128, 256, 35, 35], f32), T([128, 256, 35, 35], f32), T([128, 256, 35, 35], f32), T([128, 256, 35, 35], f32), T([128, 32, 35, 35], f32), T([1, 32, 1, 1], f32), T([1, 32, 1, 1], f32), T([32], f32), T([32], f32), T([], f32), T([128, 96, 35, 35], f32), T([1, 96, 1, 1], f32), T([1, 96, 1, 1], f32), T([96], f32), T([96], f32), T([128, 64, 35, 35], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32), T([128, 64, 35, 35], f32), T([1, 64, 1, 1], f32), T([1, 64, 1, 1], f32), T([64], f32), T([64], f32))")
def oracle_forward(inputs):
    """Run the oracle computation.

    SCOPE INVARIANT: Must accept the same inputs as Repro.forward() and return
    the same outputs (same count, same shapes, same dtypes, same strides).
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
        ok = ok and shape_ok and dtype_ok and stride_ok
        print(
            f"  strict scope output {idx}: "
            f"shape_match={shape_ok} dtype_match={dtype_ok} stride_match={stride_ok} "
            f"oracle_stride={got.stride()} eager_stride={expected.stride()}"
        )
    print(f"  strict scope: {'PASS' if ok else 'FAIL'}")
    return ok


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

    if args.show_hw:
        print(json.dumps(get_hardware_info(), indent=2))
        return

    # Default: run both --check and --bench
    if not args.check and not args.bench:
        args.check = args.bench = True

    inputs = get_inputs()
    instance = get_repro_instance()

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
            if result["status"] == "BAD_ORACLE":
                print(
                    f"WARNING: oracle is slower than compile "
                    f"(ratio={result['ratio']:.3f}x)"
                )


if __name__ == "__main__":
    main()
