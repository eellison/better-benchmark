"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete BERT bf16 mask-fill scaled attention softmax scope in one Triton row kernel, including the bool `eq(0)` mask, finite bf16 fill value, shape-param view, divide-by-8 score scale, stable fp32 natural-exp softmax, explicit bf16 probability cast, expand, and final `[192,128,128]` view store, whereas Inductor lowers the decomposed eq/full/view/div/where/cast/amax/sub/exp/sum/div/cast/expand/view graph through generic pointwise mask work plus reduction scheduling; Inductor cannot do this today because its pattern library does not recognize this broadcast bool mask plus bf16 scaled scores and finite masked fill as one full-scope row-softmax template that preserves the dtype boundaries; the fix is NEW_PATTERN: add a guarded BERT masked scaled bf16 attention-softmax lowering that fuses mask application, stable row reductions, final bf16 cast, and output-view emission."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _masked_scaled_softmax_kernel(
    mask_ptr,
    scores_ptr,
    out_ptr,
    mask_s0: tl.constexpr,
    mask_s2: tl.constexpr,
    mask_s3: tl.constexpr,
    scores_s0: tl.constexpr,
    scores_s1: tl.constexpr,
    scores_s2: tl.constexpr,
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
        mask_ptr + batch * mask_s0 + q * mask_s2 + cols * mask_s3,
        mask=col_mask,
        other=0,
    )

    flat_heads = batch * heads + head_offsets
    score_offsets = (
        flat_heads[:, None] * scores_s0
        + q * scores_s1
        + cols[None, :] * scores_s2
    )
    scores = tl.load(scores_ptr + score_offsets, mask=elem_mask, other=0.0).to(tl.float32)
    fill = tl.full((), -998244352.0, tl.float32)
    x = tl.where(keep[None, :] & elem_mask, scores * 0.125, fill)

    row_max = tl.max(x, axis=1)
    numer = libdevice.exp(x - row_max[:, None])
    numer = tl.where(elem_mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    out_offsets = (
        flat_heads[:, None] * out_s0
        + q * out_s1
        + cols[None, :] * out_s2
    )
    tl.store(out_ptr + out_offsets, probs.to(tl.bfloat16), mask=elem_mask)


# b762f7d9: (T([16,1,128,128], b8), T([192,128,128], bf16), S([16,12,128,128]), S([16,12,128,128]), S([192,128,128]))
@oracle_impl(hardware="B200", point="b762f7d9", BLOCK_H=8, BLOCK_K=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, BLOCK_K: int, num_warps: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1, _shape_param_2 = inputs
    del _shape_param_0, _shape_param_1

    out_shape = tuple(int(dim) for dim in _shape_param_2)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    batch = int(arg0_1.shape[0])
    heads = int(_shape_param_2[0]) // batch
    q_len = int(arg1_1.shape[1])
    k_len = int(arg1_1.shape[2])

    _masked_scaled_softmax_kernel[(batch * q_len, triton.cdiv(heads, BLOCK_H))](
        arg0_1,
        arg1_1,
        out,
        mask_s0=arg0_1.stride(0),
        mask_s2=arg0_1.stride(2),
        mask_s3=arg0_1.stride(3),
        scores_s0=arg1_1.stride(0),
        scores_s1=arg1_1.stride(1),
        scores_s2=arg1_1.stride(2),
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
