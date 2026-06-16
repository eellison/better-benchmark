"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete TrOCR bf16 softmax-backward row update in one Triton row-reduction kernel, including the `[1024,256,256] -> [64,16,256,256]` metadata view, fp32 broadcast add from the `[64,1,256,256]` tensor, fp32 subtract/`aten.exp`/divide by the row denominator, fp32 multiplication by the bf16 gradient input, the last-dimension `sum(..., keepdim=True)`, `prims.fma(-div, row_sum, mul)` epilogue, bf16 conversion, and final contiguous `[1024,256,256]` view. Inductor already lowers this as a generic fixed-width persistent row reduction plus epilogue, so the local gap is mostly launch/memory/math floor rather than missing broader scope; the fix is BANDWIDTH_BOUND: record this as a full-scope floor/numerics-flag check unless a broader row-reduction/softmax-backward template changes both sides."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 1024 * 256
K = 256


@triton.jit
def _softmax_backward_kernel(
    grad_ptr,
    logits_bf16_ptr,
    bias_ptr,
    row_shift_ptr,
    row_denom_ptr,
    out_ptr,
    ROW_BLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * ROW_BLOCK + tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, BLOCK_K)
    row_ids = rows[:, None]
    col_ids = cols[None, :]
    row_mask = rows < 262144
    mask = row_mask[:, None] & (col_ids < 256)

    offsets = row_ids * 256 + col_ids
    query = rows - (rows // 256) * 256
    batch = rows // 4096
    bias_offsets = batch[:, None] * 65536 + query[:, None] * 256 + col_ids

    grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    logits = tl.load(logits_bf16_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    shift = tl.load(row_shift_ptr + rows, mask=row_mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    denom = tl.load(row_denom_ptr + rows, mask=row_mask, other=1.0, eviction_policy="evict_last").to(tl.float32)

    shifted = (logits + bias) - shift[:, None]
    prob_for_sum = libdevice.exp(shifted) / denom[:, None]
    product = grad * prob_for_sum
    row_sum = tl.sum(product, axis=1)[:, None].to(tl.float32)

    prob_for_epilogue = libdevice.exp(shifted) / denom[:, None]
    out = tl.fma(-prob_for_epilogue, row_sum, product)
    tl.store(out_ptr + offsets, out, mask=mask)


# b0ff1d38: TrOCR bf16 softmax-backward row update, K=256.
@oracle_impl(hardware="B200", point="b0ff1d38", ROW_BLOCK=2, BLOCK_K=256, num_warps=4, num_stages=4)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK: int,
    BLOCK_K: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, *_shape_params = inputs
    out = torch.empty_strided(
        (1024, 256, 256),
        (K * K, K, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _softmax_backward_kernel[(triton.cdiv(ROWS, ROW_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        out,
        ROW_BLOCK=ROW_BLOCK,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
