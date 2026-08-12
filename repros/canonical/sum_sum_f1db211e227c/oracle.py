"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 GhostNet gated product-spatial-reduction scope, including the returned f32 gate tensor, the bf16 product sum round trip before the gate, the bf16 gated row tensor, and the final returned f32 channel vector from that bf16 tensor, whereas Inductor lowers the decomposed multiply, spatial sum, cast/gate, materialized side outputs, and dependent channel sum through generic reduction and pointwise schedules; Inductor cannot do this today because its scheduler does not form one full-scope dependent-reduction plan that preserves visible bf16 rounding boundaries and sibling output stores across the spatial and batch reductions; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse returned row producers with downstream channel reductions while keeping explicit dtype boundaries and output scope in one guarded plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _round_bf16_to_fp32(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _spatial_gate_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    arg3_ptr,
    gate_out_ptr,
    row_out_ptr,
    arg0_s0: tl.constexpr,
    arg0_s1: tl.constexpr,
    arg0_s2: tl.constexpr,
    arg0_s3: tl.constexpr,
    arg1_s0: tl.constexpr,
    arg1_s1: tl.constexpr,
    arg1_s2: tl.constexpr,
    arg1_s3: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    c_block = tl.program_id(1)

    c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)[None, :]
    c_vec = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_vec < C
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for base in tl.range(0, HW, BLOCK_HW):
        hw = base + tl.arange(0, BLOCK_HW)[:, None]
        h = hw // W
        w = hw - h * W
        elem_mask = (hw < HW) & (c < C)

        arg0_offsets = n * arg0_s0 + c * arg0_s1 + h * arg0_s2 + w * arg0_s3
        arg1_offsets = n * arg1_s0 + c * arg1_s1 + h * arg1_s2 + w * arg1_s3

        lhs = tl.load(arg0_ptr + arg0_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        rhs = tl.load(arg1_ptr + arg1_offsets, mask=elem_mask, other=0.0).to(tl.float32)
        product = (lhs * rhs).to(tl.bfloat16).to(tl.float32)
        acc += tl.sum(tl.where(elem_mask, product, 0.0), axis=0)

    rounded_sum = _round_bf16_to_fp32(acc)

    row_offsets = n * C + c_vec
    gate = tl.load(arg2_ptr + row_offsets, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(gate_out_ptr + row_offsets, gate, mask=c_mask)

    scaled = rounded_sum * 0.16666666666666666
    fill = tl.load(arg3_ptr).to(tl.float32)
    gated = tl.where((gate > -3.0) & (gate < 3.0), scaled, fill)
    tl.store(row_out_ptr + row_offsets, gated.to(tl.bfloat16), mask=c_mask)


@triton.jit
def _channel_sum_kernel(
    row_out_ptr,
    final_out_ptr,
    C: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    n = tl.arange(0, BLOCK_N)[:, None]
    mask = (n < N) & (c[None, :] < C)
    vals = tl.load(row_out_ptr + n * C + c[None, :], mask=mask, other=0.0).to(tl.float32)
    total = tl.sum(tl.where(mask, vals, 0.0), axis=0)
    final_f32 = _round_bf16_to_fp32(total)
    tl.store(final_out_ptr + c, final_f32, mask=c < C)


# 8f106dba: N=512, C=120, H=W=28.
@oracle_impl(hardware="B200", point="8f106dba", BLOCK_HW=128, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# f063e905: N=512, C=480, H=W=14.
@oracle_impl(hardware="B200", point="f063e905", BLOCK_HW=256, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=4)
# ae67748a: N=512, C=672, H=W=14.
@oracle_impl(hardware="B200", point="ae67748a", BLOCK_HW=256, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=8)
# 2514b9d8: N=512, C=960, H=W=7.
@oracle_impl(hardware="B200", point="2514b9d8", BLOCK_HW=64, BLOCK_C=64, FINAL_BLOCK_C=32, num_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_HW: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1 = inputs
    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w

    gate_out = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    row_out = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    final_out = torch.empty_strided(
        (c,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _spatial_gate_kernel[(n, triton.cdiv(c, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        gate_out,
        row_out,
        arg0_s0=arg0_1.stride(0),
        arg0_s1=arg0_1.stride(1),
        arg0_s2=arg0_1.stride(2),
        arg0_s3=arg0_1.stride(3),
        arg1_s0=arg1_1.stride(0),
        arg1_s1=arg1_1.stride(1),
        arg1_s2=arg1_1.stride(2),
        arg1_s3=arg1_1.stride(3),
        C=c,
        H=h,
        W=w,
        HW=hw,
        BLOCK_HW=BLOCK_HW,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _channel_sum_kernel[(triton.cdiv(c, FINAL_BLOCK_C),)](
        row_out,
        final_out,
        C=c,
        N=n,
        BLOCK_N=512,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=4,
        num_stages=3,
    )

    return gate_out, row_out, final_out
