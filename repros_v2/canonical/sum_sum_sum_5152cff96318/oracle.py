"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvNeXtV2 GRN-backward-style scope from `Repro.forward`, including the bf16-to-f32 NCHW/NHWC producers, the two per-pixel channel reductions, the returned bf16 `[N,C,H,W]` tensor with its spatial-swapped stride, and all three returned f32 channel reductions, whereas Inductor lowers the shared producers, row reductions, full-tensor epilogue/permute, and final channel reductions as separate generic kernels over large intermediates; Inductor cannot do this today because its scheduler does not form one full-scope multi-output reduction plan across reductions with different axes and a dependent bf16 tensor side output; the fix is SCHEDULER_FUSION: teach the reduction scheduler to recognize this GRN DAG, compute the row summaries once, sink the tensor epilogue store, and reuse the same pass for the compatible channel partials."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _grn_partials_kernel(
    arg0_ptr,
    weight_ptr,
    arg2_ptr,
    mean_ptr,
    scale_ptr,
    out_ptr,
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    C: tl.constexpr,
    H: tl.constexpr,
    GRN_C: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)[None, :]
    c = tl.arange(0, C_BLOCK)[:, None]
    row_mask = rows < TOTAL_ROWS
    c_mask = c < C
    mask = row_mask & c_mask

    hw = H * H
    h = rows % H
    w = (rows // H) % H
    n = rows // hw

    arg0_offsets = n * C * hw + c + h * H * C + w * C
    arg2_offsets = n * C * hw + h + w * H * C + c * H
    row_offsets = n * hw + h + w * H

    x = tl.load(arg0_ptr + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    arg2 = tl.load(arg2_ptr + arg2_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + row_offsets, mask=row_mask, other=0.0).to(tl.float32)

    centered_scaled = (arg2 - mean) * scale
    weighted = x * weight
    row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
    row_dot = tl.sum(tl.where(mask, weighted * centered_scaled, 0.0), axis=0)

    out_f32 = (scale * (1.0 / GRN_C)) * (
        weighted * GRN_C - row_sum[None, :] - centered_scaled * row_dot[None, :]
    )
    out_bf16 = _round_to_bf16_f32(out_f32)
    out_offsets = n * C * hw + c + h * C + w * H * C
    tl.store(out_ptr + out_offsets, out_bf16, mask=mask)

    partial_offsets = tl.program_id(0) * C + tl.arange(0, C_BLOCK)
    c_vec_mask = tl.arange(0, C_BLOCK) < C
    tl.store(
        partial0_ptr + partial_offsets,
        tl.sum(tl.where(mask, x * centered_scaled, 0.0), axis=1),
        mask=c_vec_mask,
    )
    tl.store(
        partial1_ptr + partial_offsets,
        tl.sum(tl.where(mask, x, 0.0), axis=1),
        mask=c_vec_mask,
    )
    tl.store(
        partial2_ptr + partial_offsets,
        tl.sum(tl.where(mask, out_bf16, 0.0), axis=1),
        mask=c_vec_mask,
    )


@triton.jit
def _finalize_partials_kernel(
    partial0_ptr,
    partial1_ptr,
    partial2_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    C: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    r = tl.arange(0, BLOCK_TILES)
    acc0 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_TILES], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_TILES], dtype=tl.float32)

    for start in range(0, N_TILES, BLOCK_TILES):
        tile = start + r
        mask = tile < N_TILES
        offsets = tile * C + c
        acc0 += tl.load(partial0_ptr + offsets, mask=mask, other=0.0)
        acc1 += tl.load(partial1_ptr + offsets, mask=mask, other=0.0)
        acc2 += tl.load(partial2_ptr + offsets, mask=mask, other=0.0)

    tl.store(out0_ptr + c, tl.sum(acc0, axis=0))
    tl.store(out1_ptr + c, tl.sum(acc1, axis=0))
    tl.store(out3_ptr + c, tl.sum(acc2, axis=0))


def _oracle_impl(
    inputs,
    *,
    C: int,
    H: int,
    GRN_C: int,
    BLOCK_M: int,
    C_BLOCK: int,
    BLOCK_TILES: int,
    partial_warps: int,
    final_warps: int,
):
    arg0, weight, arg2, mean, scale = inputs
    total_rows = N * H * H
    n_tiles = triton.cdiv(total_rows, BLOCK_M)

    out0 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    out1 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    out3 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    out2 = torch.empty_strided(
        (N, C, H, H),
        (C * H * H, 1, C, H * C),
        device=arg0.device,
        dtype=torch.bfloat16,
    )
    partial0 = torch.empty_strided((n_tiles, C), (C, 1), device=arg0.device, dtype=torch.float32)
    partial1 = torch.empty_strided((n_tiles, C), (C, 1), device=arg0.device, dtype=torch.float32)
    partial2 = torch.empty_strided((n_tiles, C), (C, 1), device=arg0.device, dtype=torch.float32)

    _grn_partials_kernel[(n_tiles,)](
        arg0,
        weight,
        arg2,
        mean,
        scale,
        out2,
        partial0,
        partial1,
        partial2,
        C,
        H,
        GRN_C,
        total_rows,
        BLOCK_M=BLOCK_M,
        C_BLOCK=C_BLOCK,
        num_warps=partial_warps,
    )
    _finalize_partials_kernel[(C,)](
        partial0,
        partial1,
        partial2,
        out0,
        out1,
        out3,
        C,
        n_tiles,
        BLOCK_TILES=BLOCK_TILES,
        num_warps=final_warps,
    )
    return out0, out1, out2, out3


@oracle_impl(
    hardware="B200",
    point="678a07bc",
    C=160,
    H=28,
    GRN_C=160,
    BLOCK_M=16,
    C_BLOCK=256,
    BLOCK_TILES=1024,
    partial_warps=8,
    final_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="b0299cd5",
    C=320,
    H=14,
    GRN_C=160,
    BLOCK_M=8,
    C_BLOCK=512,
    BLOCK_TILES=1024,
    partial_warps=8,
    final_warps=8,
)
def oracle_forward(inputs, **kwargs):
    return _oracle_impl(inputs, **kwargs)
