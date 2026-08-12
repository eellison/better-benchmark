"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet bf16 attention-backward row epilogue as a specialized Triton row reduction, including the shape-param view, dropout-mask scaling, probability reconstruction, Inductor's power-of-two scale sinking, `tl_math.exp` formulation, fp32 row sum, `libdevice.fma` update, final bf16 store, and returned aliasing `[256,512,512]` view. Inductor already emits a near-floor persistent reduction for this fixed-width row, but this oracle keeps the full captured scope in one tuned tile while matching the generated exp/cast lowering that matters for B200 numerics."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math

from oracle_harness import oracle_impl


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
    mask = (rows[:, None] < N_ROWS) & (cols[None, :] < N_COLS)
    offsets = rows[:, None] * N_COLS + cols[None, :]

    pred = tl.load(row_pred_ptr + rows, mask=rows < N_ROWS, other=0, eviction_policy="evict_last").to(tl.int1)
    prob = tl.load(prob_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    base_a = tl.load(row_base_a_ptr + rows, mask=rows < N_ROWS, other=0.0, eviction_policy="evict_last")
    base_b = tl.load(row_base_b_ptr + rows, mask=rows < N_ROWS, other=0.0, eviction_policy="evict_last")
    denom = tl.load(row_denom_ptr + rows, mask=rows < N_ROWS, other=1.0, eviction_policy="evict_last")

    true_path = (prob * 1.0 - base_a[:, None]) * 0.125
    false_path = prob * 0.125 - base_b[:, None]
    exponent = tl.where(pred[:, None], true_path, false_path)
    div = tl_math.exp(exponent) / denom[:, None]

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    keep = tl.load(dropout_mask_ptr + offsets, mask=mask, other=0).to(tl.int1).to(tl.float32)
    scaled_grad = grad * (keep * 1.1111111111111112)
    product = scaled_grad * div
    row_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)[:, None]
    out = libdevice.fma(-div, row_sum, product) * 0.125

    tl.store(out_ptr + offsets, out, mask=mask)


# 66f209c9: (T([256,512,512], bf16), T([16,16,512,512], b8), ...)
@oracle_impl(hardware="B200", point="66f209c9", BLOCK_M=2, BLOCK_N=512, num_warps=8)
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
        num_stages=1,
    )
    return out, out.view(tuple(int(dim) for dim in _shape_param_2))
