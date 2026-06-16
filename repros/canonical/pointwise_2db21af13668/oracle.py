"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 Swin QKV split/layout/pad scope by writing the final padded Q, transposed-K, and V outputs directly from the contiguous `[B*49,3*H*32]` projection, including the captured Q scale and zero padding, whereas Inductor lowers the reshape/permute/unbind branches as sibling layout materializations plus separate constant-pad work; Inductor cannot do this today because its scheduler does not fuse multiple users of the QKV producer when each branch needs a different output layout and pad epilogue; the fix is SCHEDULER_FUSION: teach layout-copy scheduling to group QKV split materializations and sink branch-local scale/pad epilogues into one multi-output copy."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _swin_qkv_pad_kernel(
    in_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    HEADS: tl.constexpr,
    YBLOCK: tl.constexpr,
):
    y = tl.program_id(0) * YBLOCK + tl.arange(0, YBLOCK)[:, None]
    d = tl.arange(0, 32)[None, :]

    token = y % 56
    row = y // 56
    head = row % HEADS
    batch = row // HEADS
    qkv_plane: tl.constexpr = HEADS * 32
    source = (batch * 49 + token) * (3 * qkv_plane) + head * 32 + d
    load_mask = token < 49

    q_values = tl.load(in_ptr + source, mask=load_mask, other=0.0)
    k_values = tl.load(in_ptr + source + qkv_plane, mask=load_mask, other=0.0)
    v_values = tl.load(in_ptr + source + 2 * qkv_plane, mask=load_mask, other=0.0)

    qv_offsets = y * 32 + d
    k_offsets = (row * 32 + d) * 56 + token
    tl.store(q_out_ptr + qv_offsets, q_values * 0.1767766952966369)
    tl.store(k_out_ptr + k_offsets, k_values)
    tl.store(v_out_ptr + qv_offsets, v_values)


@oracle_impl(hardware="B200", point="5a91604c", YBLOCK=64, num_warps=4)
@oracle_impl(hardware="B200", point="5c047c65", YBLOCK=32, num_warps=4)
@oracle_impl(hardware="B200", point="25d25c52", YBLOCK=32, num_warps=4)
@oracle_impl(hardware="B200", point="353652d2", YBLOCK=32, num_warps=4)
def oracle_forward(inputs, *, YBLOCK, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _pad0, _shape_param_5, _shape_param_6, _pad1, _shape_param_8, _shape_param_9, _pad2 = inputs

    batch = int(_shape_param_0[0])
    heads = int(_shape_param_1[3])
    rows = batch * heads
    qv_shape = (rows, 56, 32)
    k_shape = (rows, 32, 56)

    q_out = torch.empty_strided(qv_shape, (56 * 32, 32, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    k_out = torch.empty_strided(k_shape, (32 * 56, 56, 1), device=arg0_1.device, dtype=arg0_1.dtype)
    v_out = torch.empty_strided(qv_shape, (56 * 32, 32, 1), device=arg0_1.device, dtype=arg0_1.dtype)

    grid = (triton.cdiv(rows * 56, YBLOCK),)
    _swin_qkv_pad_kernel[grid](
        arg0_1,
        q_out,
        k_out,
        v_out,
        HEADS=heads,
        YBLOCK=YBLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return q_out, k_out, v_out
