"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 BERT residual LayerNorm fragment in one fixed-hidden row-reduction kernel, including the `[2048,768] -> [16,128,768]` view, visible bf16 residual add, bf16 mean/sub/mul/div/add rounding boundaries, unbiased fp32 variance over the rounded residual add, and the final contiguous `[2048,768]` view, whereas Inductor lowers the returned residual add, mean/variance row reductions, and affine epilogue through generic normalization/pointwise scheduling; Inductor cannot do this today because its normalization scheduler does not keep a live returned bf16 residual producer and all dependent row-statistics/affine epilogues in one full-scope template while preserving explicit dtype boundaries; the fix is SCHEDULER_FUSION: teach the row-normalization template to fuse visible residual producers with bf16 rounded epilogues and alias/view output materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _bert_residual_layernorm_kernel(
    arg0_ptr,
    arg1_ptr,
    weight_ptr,
    bias_ptr,
    add_out_ptr,
    norm_out_ptr,
    ROWS: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
):
    row_ids = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    rows = row_ids[:, None]
    cols = tl.arange(0, BLOCK_H)[None, :]
    row_mask = rows < ROWS
    col_mask = cols < HIDDEN
    mask = row_mask & col_mask
    offsets = rows * HIDDEN + cols

    lhs = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    add_bf16 = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    tl.store(add_out_ptr + offsets, add_bf16.to(tl.bfloat16), mask=mask)

    add_for_reduce = tl.where(mask, add_bf16, 0.0)
    sum_x = tl.sum(add_for_reduce, axis=1)[:, None]
    mean_f32 = sum_x / HIDDEN
    mean_bf16 = mean_f32.to(tl.bfloat16).to(tl.float32)

    centered_for_epilogue = (add_bf16 - mean_bf16).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scaled = (weight * centered_for_epilogue).to(tl.bfloat16).to(tl.float32)

    sum_x2 = tl.sum(tl.where(mask, add_bf16 * add_bf16, 0.0), axis=1)[:, None]
    var_sum = sum_x2 - sum_x * mean_f32
    variance = var_sum / (HIDDEN - 1.0)
    denom = (tl.sqrt(tl.maximum(variance, 0.0)).to(tl.bfloat16).to(tl.float32) + 1.0e-6)
    denom = denom.to(tl.bfloat16).to(tl.float32)

    divided = (scaled / denom).to(tl.bfloat16).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    out = (divided + bias).to(tl.bfloat16)
    tl.store(norm_out_ptr + offsets, out, mask=mask)


@oracle_impl(hardware="B200", point="1c404995", BLOCK_H=1024, ROW_BLOCK=2, num_warps=4)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int, num_warps: int):
    arg0, arg1, arg2, arg3, shape0, shape1 = inputs
    rows = int(arg0.shape[0])
    hidden = int(arg0.shape[1])
    add_out = torch.empty_strided(
        tuple(int(dim) for dim in shape0),
        (128 * hidden, hidden, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    norm_out = torch.empty_strided(
        tuple(int(dim) for dim in shape1),
        (hidden, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    _bert_residual_layernorm_kernel[(triton.cdiv(rows, ROW_BLOCK),)](
        arg0,
        arg1,
        arg2,
        arg3,
        add_out,
        norm_out,
        ROWS=rows,
        HIDDEN=hidden,
        BLOCK_H=BLOCK_H,
        ROW_BLOCK=ROW_BLOCK,
        num_warps=num_warps,
    )
    return add_out, norm_out
