"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GPT-Neo bf16 attention softmax-backward row update in one Triton kernel, including the `[512,128,128] -> [32,16,128,128]` view semantics, generated same-segment causal mask reconstruction from the position and segment-index inputs, the sliced external bool mask, bf16 mask-fill logits, natural-exp probability reconstruction from the saved row shift and denominator, fp32 row product sum, prims.fma-equivalent epilogue, explicit bf16 cast, final mask fill, and returned contiguous `[512,128,128]` output. Inductor lowers the mask construction, softmax-backward reduction, fma epilogue, cast, and final where through generic producer/reduction scheduling; it cannot do this today because scheduler/codegen does not recognize this full saved-attention-backward envelope as one row-reduction template with observable bf16 rounding and mask boundaries. The fix is SCHEDULER_FUSION: fuse mask reconstruction, probability reconstruction, row reduction, fma, bf16 rounding, and output store in one shape-specialized row program."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
N_ROWS = BATCH * HEADS * SEQ
N_COLS = SEQ


@triton.jit
def _gptneo_attention_backward_kernel(
    grad_ptr,
    positions_ptr,
    segments_ptr,
    mask_scalar_ptr,
    logits_ptr,
    external_mask_ptr,
    row_shift_ptr,
    row_denom_ptr,
    fill_scalar_ptr,
    out_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_offsets = rows[:, None] * 128 + cols[None, :]

    batch = rows // (16 * 128)
    rem = rows - batch * (16 * 128)
    head = rem // 128
    query = rem - head * 128

    grad = tl.load(grad_ptr + row_offsets, eviction_policy="evict_first").to(tl.float32)
    logits = tl.load(logits_ptr + row_offsets, eviction_policy="evict_first")

    external_keep = tl.load(
        external_mask_ptr + query[:, None] * 2048 + cols[None, :],
        eviction_policy="evict_last",
    )
    neg_bf16 = tl.full((BLOCK_M, BLOCK_N), -3.3895313892515355e38, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    masked_logits = tl.where(external_keep, logits, neg_bf16).to(tl.float32)

    q_pos = tl.load(positions_ptr + query).to(tl.int64)
    k_pos = tl.load(positions_ptr + cols).to(tl.int64)
    q_segment = tl.load(segments_ptr + batch * 128 + q_pos).to(tl.int64)
    k_segment = tl.load(segments_ptr + batch[:, None] * 128 + k_pos[None, :]).to(tl.int64)
    structured_keep = (k_pos[None, :] <= q_pos[:, None]) & (
        k_segment == q_segment[:, None]
    )
    mask_scalar = tl.load(mask_scalar_ptr).to(tl.float32)
    mask_bias = tl.where(
        structured_keep,
        mask_scalar,
        tl.full((BLOCK_M, BLOCK_N), -3.4028234663852886e38, tl.float32),
    )

    row_index = (batch * 16 + head) * 128 + query
    row_shift = tl.load(row_shift_ptr + row_index).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + row_index).to(tl.float32)

    scores = masked_logits + mask_bias
    probs = libdevice.exp(scores - row_shift[:, None]) / row_denom[:, None]
    product = grad * probs
    row_sum = tl.sum(product, axis=1)[:, None]
    value = tl.fma(-probs, row_sum, product).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    fill = tl.load(fill_scalar_ptr)
    out = tl.where(external_keep, value, fill)
    tl.store(out_ptr + row_offsets, out)


@oracle_impl(hardware="B200", point="c5abdf2a", BLOCK_M=8, BLOCK_N=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        grad,
        positions,
        segments,
        mask_scalar,
        logits,
        external_mask,
        row_shift,
        row_denom,
        fill_scalar,
        shape0,
        shape1,
        shape2,
        shape3,
    ) = inputs
    del shape0, shape1, shape2, shape3

    out = torch.empty_strided(
        (BATCH * HEADS, SEQ, SEQ),
        (SEQ * SEQ, SEQ, 1),
        device=grad.device,
        dtype=torch.bfloat16,
    )
    _gptneo_attention_backward_kernel[(triton.cdiv(N_ROWS, BLOCK_M),)](
        grad,
        positions,
        segments,
        mask_scalar,
        logits,
        external_mask,
        row_shift,
        row_denom,
        fill_scalar,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
