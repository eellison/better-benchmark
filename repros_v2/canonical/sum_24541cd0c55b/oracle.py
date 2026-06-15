"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 attention-backward row epilogue in one Triton kernel, including the shape-param view, dropout-mask scaling, bf16-scaled probability reconstruction, row-wise natural-exp denominator divide, fp32 product sum, prims.fma-equivalent broadcast update, explicit bf16 rounding, final scale rounding, and returned aliasing `[256,512,512]` view, whereas Inductor lowers the decomposed view/squeeze/cast/mul/sub/where/exp/div/sum/fma/cast/mul/view/permute/view graph through its generic reduction scheduler; Inductor cannot do this today because the scheduler does not recognize this fixed-width attention-backward reduction with observable bf16 rounding boundaries and a metadata-only sibling output as one full-scope row template; the fix is SCHEDULER_FUSION: add a guarded XLNet attention-backward row-reduction template that sinks the probability reconstruction, reduction epilogue, final cast/scale, and alias-only view return into one scheduled unit."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _fma_rn_f32(a, b, c):
    return tl.inline_asm_elementwise(
        "fma.rn.f32 $0, $1, $2, $3;",
        constraints="=f,f,f,f",
        args=[a, b, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _xlnet_attention_backward_kernel(
    grad_ptr,
    dropout_mask_ptr,
    prob_ptr,
    row_base_a_ptr,
    row_base_b_ptr,
    row_pred_ptr,
    row_denom_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < N_COLS
    mask = row_mask[:, None] & col_mask[None, :]
    offsets = rows[:, None] * N_COLS + cols[None, :]

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.float32)
    prob_bf16 = tl.load(prob_ptr + offsets, mask=mask, other=0.0)
    prob = prob_bf16.to(tl.float32)

    base_a = tl.load(row_base_a_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    base_b = tl.load(row_base_b_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
    pred = tl.load(row_pred_ptr + rows, mask=row_mask, other=0).to(tl.int1)
    denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0).to(tl.float32)

    scaled_prob = (prob_bf16 * 0.125).to(tl.bfloat16).to(tl.float32)
    true_path = (prob - base_a[:, None]) * 0.125
    false_path = scaled_prob - base_b[:, None]
    exponent = tl.where(pred[:, None], true_path, false_path)
    div = libdevice.exp(exponent) / denom[:, None]

    scaled_grad = grad * (keep * 1.1111111111111112)
    product = scaled_grad * div
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
    fma = _fma_rn_f32(-div, row_sum, product)
    rounded = fma.to(tl.bfloat16).to(tl.float32)
    out = (rounded * 0.125).to(tl.bfloat16)

    tl.store(out_ptr + offsets, out, mask=mask)


# 66f209c9: (T([256,512,512], bf16), T([16,16,512,512], b8), ...)
@oracle_impl(hardware="B200", point="66f209c9", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_N, num_warps):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
    ) = inputs
    del _shape_param_0, _shape_param_1

    out = torch.empty_strided(
        tuple(arg2_1.shape),
        tuple(arg2_1.stride()),
        device=arg2_1.device,
        dtype=torch.bfloat16,
    )
    n_cols = int(arg2_1.shape[-1])
    n_rows = arg2_1.numel() // n_cols

    _xlnet_attention_backward_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out,
        N_ROWS=n_rows,
        N_COLS=n_cols,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.view(tuple(int(dim) for dim in _shape_param_2))
