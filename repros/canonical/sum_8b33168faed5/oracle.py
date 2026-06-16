"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the
complete GPT-Neo bf16 attention softmax-backward row update in one Triton kernel,
including the `[512,128,128] -> [32,16,128,128]` view semantics, structured
same-segment causal mask reconstruction from the integer inputs, the sliced
causal bool mask, bf16 mask-fill logits, natural-exp probability reconstruction,
fp32 row product sum, prims.fma-equivalent epilogue, explicit bf16 cast, final
mask fill, and returned contiguous `[512,128,128]` output. Inductor lowers the
mask construction, softmax-backward reduction, fma epilogue, cast, and final
where through generic producer/reduction scheduling. It cannot do this today
because scheduler/codegen does not recognize this full saved-attention-backward
envelope as one row-reduction template with observable bf16 rounding and mask
boundaries; the fix is SCHEDULER_FUSION: fuse mask reconstruction, probability
reconstruction, row reduction, fma, bf16 rounding, and output store in one
shape-specialized row program."""

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
ROWS_PER_PROGRAM = 8


@triton.jit
def _gptneo_attention_backward_kernel(
    grad_ptr,
    batch_index_ptr,
    positions_ptr,
    segment_ptr,
    mask_scalar_ptr,
    logits_ptr,
    causal_mask_ptr,
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

    b = rows // (16 * 128)
    rem = rows - b * (16 * 128)
    h = rem // 128
    q = rem - h * 128

    grad = tl.load(grad_ptr + row_offsets, eviction_policy="evict_first").to(tl.float32)
    logits = tl.load(logits_ptr + row_offsets, eviction_policy="evict_first")

    causal_mask = tl.load(
        causal_mask_ptr + q[:, None] * 2048 + cols[None, :],
        eviction_policy="evict_last",
    )
    neg_bf16 = tl.full((BLOCK_M, BLOCK_N), -3.3895313892515355e38, tl.float32).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )
    masked_logits = tl.where(causal_mask, logits, neg_bf16).to(tl.float32)

    q_pos = tl.load(positions_ptr + q).to(tl.int64)
    k_pos = tl.load(positions_ptr + cols).to(tl.int64)
    batch_index = tl.load(batch_index_ptr + b).to(tl.int64)
    q_segment = tl.load(segment_ptr + batch_index * 128 + q_pos).to(tl.int64)
    k_segment = tl.load(
        segment_ptr + batch_index[:, None] * 128 + k_pos[None, :]
    ).to(tl.int64)
    structured_mask = (k_pos[None, :] <= q_pos[:, None]) & (
        k_segment == q_segment[:, None]
    )
    mask_scalar = tl.load(mask_scalar_ptr).to(tl.float32)
    mask_bias = tl.where(
        structured_mask,
        mask_scalar,
        tl.full((BLOCK_M, BLOCK_N), -3.4028234663852886e38, tl.float32),
    )

    row_index = (b * 16 + h) * 128 + q
    row_shift = tl.load(row_shift_ptr + row_index).to(tl.float32)
    row_denom = tl.load(row_denom_ptr + row_index).to(tl.float32)

    scores = masked_logits + mask_bias
    shifted = scores - row_shift[:, None]
    probs = libdevice.exp(shifted) / row_denom[:, None]
    product = grad * probs
    row_sum = tl.sum(product, axis=1)[:, None]
    fma = tl.fma(-probs, row_sum, product)
    rounded = fma.to(tl.bfloat16, fp_downcast_rounding="rtne")
    fill = tl.load(fill_scalar_ptr)
    out = tl.where(causal_mask, rounded, fill)
    tl.store(out_ptr + row_offsets, out)


@oracle_impl(
    hardware="B200",
    point="6f5387ec",
    BLOCK_M=ROWS_PER_PROGRAM,
    BLOCK_N=N_COLS,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    out = torch.empty_strided(
        (BATCH * HEADS, SEQ, SEQ),
        (SEQ * SEQ, SEQ, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    _gptneo_attention_backward_kernel[(triton.cdiv(N_ROWS, BLOCK_M),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        arg9_1,
        out,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
