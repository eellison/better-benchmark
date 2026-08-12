"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 masked batch-norm-backward fragment by materializing the returned bf16 `where` tensor, split-K reducing both shared per-channel summaries from that rounded producer, and using the finalized summaries to emit the returned fp32 vectors and bf16 full-tensor epilogue, whereas Inductor schedules the masked producer, sibling reductions, and dependent broadcast epilogue as generic reduction/pointwise work; Inductor cannot do this today because its scheduler/codegen has no cooperative split-K multi-output reduction template that coordinates a live returned bf16 producer, compatible channel reductions, and the dependent BN-backward epilogue while preserving the explicit bf16 rounding boundary; the fix is COOPERATIVE_SPLIT_K: teach reduction scheduling to split compatible channel reductions across N/H/W, co-finalize their summaries, and fuse the returned producer plus tensor/vector epilogues into one full-scope plan."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


REDUCE_SCALE = 1.0172526041666666e-05


@triton.jit
def _where_partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    arg4_ptr,
    arg5_ptr,
    out_where_ptr,
    partial_where_ptr,
    partial_prod_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    A0_SN: tl.constexpr,
    A0_SC: tl.constexpr,
    A0_SH: tl.constexpr,
    A0_SW: tl.constexpr,
    A1_SN: tl.constexpr,
    A1_SC: tl.constexpr,
    A1_SH: tl.constexpr,
    A1_SW: tl.constexpr,
    A2_SN: tl.constexpr,
    A2_SC: tl.constexpr,
    A2_SH: tl.constexpr,
    A2_SW: tl.constexpr,
    A4_SN: tl.constexpr,
    A4_SC: tl.constexpr,
    A4_SH: tl.constexpr,
    A4_SW: tl.constexpr,
    O_SN: tl.constexpr,
    O_SC: tl.constexpr,
    O_SH: tl.constexpr,
    O_SW: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K)

    n = k_offsets // (H * W)
    spatial = k_offsets - n * (H * W)
    h = spatial // W
    w = spatial - h * W

    off0 = n * A0_SN + c_offsets * A0_SC + h * A0_SH + w * A0_SW
    off1 = n * A1_SN + c_offsets * A1_SC + h * A1_SH + w * A1_SW
    off2 = n * A2_SN + c_offsets * A2_SC + h * A2_SH + w * A2_SW
    off4 = n * A4_SN + c_offsets * A4_SC + h * A4_SH + w * A4_SW
    out_off = n * O_SN + c_offsets * O_SC + h * O_SH + w * O_SW

    lhs = tl.load(arg0_ptr + off0, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + off1, mask=active, other=0.0).to(tl.float32)
    mask_source = tl.load(arg2_ptr + off2, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(arg3_ptr).to(tl.float32)

    add_rounded = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    where_value = tl.where(mask_source <= 0.0, full_value, add_rounded)
    where_value = tl.where(active, where_value, 0.0)
    tl.store(out_where_ptr + out_off, where_value, mask=active)

    mean = tl.load(arg5_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + off4, mask=active, other=0.0).to(tl.float32) - mean

    partial_where = tl.sum(where_value, axis=0)
    partial_prod = tl.sum(where_value * tl.where(active, centered, 0.0), axis=0)
    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    partial_base = tl.program_id(1) * C + c_vec
    tl.store(partial_where_ptr + partial_base, partial_where, mask=c_vec < C)
    tl.store(partial_prod_ptr + partial_base, partial_prod, mask=c_vec < C)


@triton.jit
def _finalize_kernel(
    partial_where_ptr,
    partial_prod_ptr,
    arg6_ptr,
    sum_where_out_ptr,
    sum_prod_tmp_ptr,
    sum_prod_scaled_out_ptr,
    C: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    p_offsets = tl.arange(0, BLOCK_P)[:, None]
    active = (c_offsets < C) & (p_offsets < NUM_K_BLOCKS)
    offsets = p_offsets * C + c_offsets

    partial_where = tl.load(partial_where_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    partial_prod = tl.load(partial_prod_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.sum(partial_where, axis=0)
    sum_prod = tl.sum(partial_prod, axis=0)

    c_vec = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    invstd = tl.load(arg6_ptr + c_vec, mask=c_vec < C, other=0.0).to(tl.float32)
    tl.store(sum_where_out_ptr + c_vec, sum_where, mask=c_vec < C)
    tl.store(sum_prod_tmp_ptr + c_vec, sum_prod, mask=c_vec < C)
    tl.store(sum_prod_scaled_out_ptr + c_vec, sum_prod * invstd, mask=c_vec < C)


@triton.jit
def _epilogue_kernel(
    out_where_ptr,
    arg4_ptr,
    arg5_ptr,
    arg6_ptr,
    arg7_ptr,
    sum_where_ptr,
    sum_prod_ptr,
    out_tensor_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    K: tl.constexpr,
    A4_SN: tl.constexpr,
    A4_SC: tl.constexpr,
    A4_SH: tl.constexpr,
    A4_SW: tl.constexpr,
    O_SN: tl.constexpr,
    O_SC: tl.constexpr,
    O_SH: tl.constexpr,
    O_SW: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)[:, None]
    active = (c_offsets < C) & (k_offsets < K)

    n = k_offsets // (H * W)
    spatial = k_offsets - n * (H * W)
    h = spatial // W
    w = spatial - h * W

    off4 = n * A4_SN + c_offsets * A4_SC + h * A4_SH + w * A4_SW
    out_off = n * O_SN + c_offsets * O_SC + h * O_SH + w * O_SW

    where_value = tl.load(out_where_ptr + out_off, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(arg5_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + off4, mask=active, other=0.0).to(tl.float32) - mean
    invstd = tl.load(arg6_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    affine_weight = tl.load(arg7_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    sum_where = tl.load(sum_where_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)
    sum_prod = tl.load(sum_prod_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(tl.float32)

    mean_term = sum_where * REDUCE_SCALE_
    centered_term = sum_prod * REDUCE_SCALE_ * invstd * invstd
    output_scale = invstd * affine_weight
    out = (where_value - centered * centered_term - mean_term) * output_scale
    tl.store(out_tensor_ptr + out_off, out, mask=active)


def _ceil_pow2(value: int) -> int:
    return 1 << (value - 1).bit_length()


@oracle_impl(hardware="B200", point="ef12a986", BLOCK_K=256, BLOCK_C=32, num_warps=8)
@oracle_impl(hardware="B200", point="a61533e5", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="bdb05274", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="485be796", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="99632bd2", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="5d88cdc4", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="e2f9eee9", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="32dbfb55", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="2969d190", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="71b6c98b", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="fff118a1", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="e9130656", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="7f1b8880", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="61bd3a8c", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="2f0e8753", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="8addce5e", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="d10f1ba2", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="25b4fe89", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="7e35b537", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="ff6e3deb", BLOCK_K=512, BLOCK_C=8, num_warps=8)
@oracle_impl(hardware="B200", point="d5d9de3b", BLOCK_K=512, BLOCK_C=8, num_warps=8)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    num_warps: int,
):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n, c, h, w = (int(dim) for dim in arg1.shape)
    k = n * h * w
    num_k_blocks = triton.cdiv(k, BLOCK_K)
    block_p = _ceil_pow2(num_k_blocks)

    out_stride = tuple(int(s) for s in arg1.stride())
    out_where = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    out_tensor = torch.empty_strided(
        (n, c, h, w),
        out_stride,
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    sum_where = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    sum_prod_scaled = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    sum_prod_tmp = torch.empty((c,), device=arg1.device, dtype=torch.float32)
    partial_where = torch.empty(
        (num_k_blocks, c),
        device=arg1.device,
        dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_where)

    a0s = tuple(int(s) for s in arg0.stride())
    a1s = tuple(int(s) for s in arg1.stride())
    a2s = tuple(int(s) for s in arg2.stride())
    a4s = tuple(int(s) for s in arg4.stride())

    grid = (triton.cdiv(c, BLOCK_C), num_k_blocks)
    _where_partial_reduce_kernel[grid](
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        out_where,
        partial_where,
        partial_prod,
        N=n,
        C=c,
        H=h,
        W=w,
        K=k,
        A0_SN=a0s[0],
        A0_SC=a0s[1],
        A0_SH=a0s[2],
        A0_SW=a0s[3],
        A1_SN=a1s[0],
        A1_SC=a1s[1],
        A1_SH=a1s[2],
        A1_SW=a1s[3],
        A2_SN=a2s[0],
        A2_SC=a2s[1],
        A2_SH=a2s[2],
        A2_SW=a2s[3],
        A4_SN=a4s[0],
        A4_SC=a4s[1],
        A4_SH=a4s[2],
        A4_SW=a4s[3],
        O_SN=out_stride[0],
        O_SC=out_stride[1],
        O_SH=out_stride[2],
        O_SW=out_stride[3],
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _finalize_kernel[(triton.cdiv(c, BLOCK_C),)](
        partial_where,
        partial_prod,
        arg6,
        sum_where,
        sum_prod_tmp,
        sum_prod_scaled,
        C=c,
        NUM_K_BLOCKS=num_k_blocks,
        BLOCK_P=block_p,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    _epilogue_kernel[grid](
        out_where,
        arg4,
        arg5,
        arg6,
        arg7,
        sum_where,
        sum_prod_tmp,
        out_tensor,
        N=n,
        C=c,
        H=h,
        W=w,
        K=k,
        A4_SN=a4s[0],
        A4_SC=a4s[1],
        A4_SH=a4s[2],
        A4_SW=a4s[3],
        O_SN=out_stride[0],
        O_SC=out_stride[1],
        O_SH=out_stride[2],
        O_SW=out_stride[3],
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )

    return out_where, sum_where, sum_prod_scaled, out_tensor
