"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-Neo scalar-fill masked additive-bias bf16 softmax scope, including the [1,1,2048,2048] bool mask slice, [512,128,128] to [32,16,128,128] fp32 view, fp32-min fill for masked elements, [32,1,128,128] bf16 bias promotion, stable natural-exp row softmax, explicit bf16 rounding, expand, and final [512,128,128] view in one Triton kernel, whereas Inductor lowers the decomposed slice/view/where/add/amax/sub/exp/sum/div/cast/expand/view graph through generic reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this scalar-fill masked attention softmax with broadcast bias as one reusable full-scope row template that hoists the shared mask and bias across heads; the fix is NEW_PATTERN: add a guarded K=128 masked scalar-fill additive-bias attention softmax lowering that reuses the sliced mask and bias across head tiles while writing the final bf16 view layout directly."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _masked_scalar_bias_softmax_kernel(
    mask_ptr,
    scores_ptr,
    bias_ptr,
    out_ptr,
    scores_s0: tl.constexpr,
    scores_s1: tl.constexpr,
    scores_s2: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    bias_s0: tl.constexpr,
    bias_s2: tl.constexpr,
    bias_s3: tl.constexpr,
    out_s0: tl.constexpr,
    out_s1: tl.constexpr,
    out_s2: tl.constexpr,
    heads: tl.constexpr,
    q_len: tl.constexpr,
    k_len: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    batch_q = tl.program_id(0)
    head_block = tl.program_id(1)

    batch = batch_q // q_len
    q = batch_q - batch * q_len
    head_offsets = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, BLOCK_K)

    head_mask = head_offsets < heads
    col_mask = cols < k_len
    elem_mask = head_mask[:, None] & col_mask[None, :]

    keep = tl.load(
        mask_ptr + q * mask_s2 + cols * mask_s3,
        mask=col_mask,
        other=0,
    )
    bias = tl.load(
        bias_ptr + batch * bias_s0 + q * bias_s2 + cols * bias_s3,
        mask=col_mask,
        other=0.0,
    ).to(tl.float32)

    flat_bh = batch * heads + head_offsets
    score_offsets = (
        flat_bh[:, None] * scores_s0
        + q * scores_s1
        + cols[None, :] * scores_s2
    )
    score_vals = tl.load(
        scores_ptr + score_offsets,
        mask=elem_mask & keep[None, :],
        other=0.0,
    ).to(tl.float32)
    fill = tl.full((), -3.4028234663852886e38, tl.float32)
    x = tl.where(keep[None, :], score_vals, fill) + bias[None, :]
    x = tl.where(elem_mask, x, -float("inf"))

    row_max = tl.max(x, axis=1)
    numer = libdevice.exp(x - row_max[:, None])
    numer = tl.where(elem_mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = (
        flat_bh[:, None] * out_s0
        + q * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=elem_mask)


# 0b7018c4: (T([1,1,2048,2048], b8), T([512,128,128], f32), T([32,1,128,128], bf16), ...)
@oracle_impl(hardware="B200", point="0b7018c4", BLOCK_H=8, BLOCK_K=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, BLOCK_K: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2

    batch = int(arg2_1.shape[0])
    heads = int(arg1_1.shape[0] // batch)
    q_len = int(arg1_1.shape[1])
    k_len = int(arg1_1.shape[2])
    out = torch.empty_strided(
        (batch * heads, q_len, k_len),
        (q_len * k_len, k_len, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    _masked_scalar_bias_softmax_kernel[
        (batch * q_len, triton.cdiv(heads, BLOCK_H))
    ](
        arg0_1,
        arg1_1,
        arg2_1,
        out,
        scores_s0=arg1_1.stride(0),
        scores_s1=arg1_1.stride(1),
        scores_s2=arg1_1.stride(2),
        mask_s2=arg0_1.stride(2),
        mask_s3=arg0_1.stride(3),
        bias_s0=arg2_1.stride(0),
        bias_s2=arg2_1.stride(2),
        bias_s3=arg2_1.stride(3),
        out_s0=out.stride(0),
        out_s1=out.stride(1),
        out_s2=out.stride(2),
        heads=heads,
        q_len=q_len,
        k_len=k_len,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
