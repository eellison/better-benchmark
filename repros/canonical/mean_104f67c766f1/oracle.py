"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Gemma bf16 embedding-scale plus RMSNorm alias scope in one shape-specialized Triton row kernel, including the token embedding gather, bf16 scalar multiply side output, fp32 mean-square reduction over the bf16-rounded scaled embedding, eps=1e-6 rsqrt, `(weight.float() + 1.0)` affine, final bf16 cast, and three returned alias views from one base buffer, whereas Inductor lowers the embedding gather, visible scaled-embedding producer, generic RMS reduction, affine epilogue, and alias-only views as separate scheduled regions; Inductor cannot do this today because its normalization pattern library does not recognize an indexed embedding gather with a visible bf16 scaled side output feeding a fixed-hidden RMSNorm and sibling alias views as one full-scope template; the fix is NEW_PATTERN: add an embedding-scale-RMSNorm lowering that folds indexed row loads, explicit bf16 rounding boundaries, row reduction, affine epilogue, and alias-view returns into one guarded schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _embedding_scaled_rmsnorm_kernel(
    embedding_ptr,
    input_ids_ptr,
    scale_ptr,
    weight_ptr,
    scaled_out_ptr,
    norm_out_ptr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN

    token = tl.load(input_ids_ptr + row)
    x = tl.load(embedding_ptr + token * HIDDEN + cols, mask=mask, other=0.0)
    scale = tl.load(scale_ptr)
    scaled_bf16 = (x * scale).to(tl.bfloat16)

    offsets = row * HIDDEN + cols
    tl.store(scaled_out_ptr + offsets, scaled_bf16, mask=mask)

    scaled_f32 = scaled_bf16.to(tl.float32)
    sum_sq = tl.sum(tl.where(mask, scaled_f32 * scaled_f32, 0.0), axis=0)
    inv_rms = tl.rsqrt(sum_sq * (1.0 / HIDDEN) + 1.0e-6)
    weight = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = scaled_f32 * inv_rms * (weight + 1.0)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


# 65410254: Gemma-2, embedding bf16[256000,2304], ids i64[1,1000], scale bf16[], weight bf16[2304]
# 402c18a0: Gemma-3, embedding bf16[262208,2560], ids i64[1,1000], scale bf16[], weight bf16[2560]
@oracle_impl(hardware="B200", point="65410254", BLOCK_H=4096, num_warps=4)
@oracle_impl(hardware="B200", point="402c18a0", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    embedding, input_ids, scale, weight, shape0, shape1, shape2 = inputs
    rows = int(input_ids.numel())
    hidden = int(embedding.shape[1])
    base_shape = (1, rows, hidden)

    scaled_out = torch.empty_strided(
        base_shape,
        (rows * hidden, hidden, 1),
        device=embedding.device,
        dtype=torch.bfloat16,
    )
    norm_base = torch.empty_strided(
        base_shape,
        (rows * hidden, hidden, 1),
        device=embedding.device,
        dtype=torch.bfloat16,
    )

    _embedding_scaled_rmsnorm_kernel[(rows,)](
        embedding,
        input_ids,
        scale,
        weight,
        scaled_out,
        norm_base,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return (
        scaled_out,
        norm_base.view(tuple(int(dim) for dim in shape0)),
        norm_base.view(tuple(int(dim) for dim in shape1)),
        norm_base.view(tuple(int(dim) for dim in shape2)),
    )
