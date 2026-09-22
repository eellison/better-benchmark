"""cuTile port of sum_02fe2c82b1fd: GPT-Neo attention softmax-backward row update.

For each row of the (BATCH*HEADS*SEQ, SEQ) matrix, reconstruct probs from saved
row_shift/row_denom, apply mask/fill, compute row_sum(grad * probs) and the fma
epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
N_ROWS = BATCH * HEADS * SEQ  # 65536
BLOCK_M = 8
BLOCK_N = SEQ  # 128 (power-of-2)
NEG_MIN_BF16 = -3.3895313892515355e38
NEG_MIN_F32 = -3.4028234663852886e38


@ct.kernel
def _gptneo_attn_backward_kernel(
    grad_flat,        # (512*128*128,) f32 (BATCH*HEADS*SEQ*SEQ)
    positions_flat,   # (128,) i64
    segments_flat,    # (32*128,) i64
    mask_scalar,      # (1,) f32
    logits_flat,      # (512*128*128,) bf16
    external_mask_flat, # (128*2048,) b8 -- slice [0:128, 0:128] logical
    row_shift_flat,   # (BATCH*HEADS*SEQ,) f32
    row_denom_flat,   # (BATCH*HEADS*SEQ,) f32
    fill_scalar,      # (1,) bf16
    out_flat,         # (512*128*128,) bf16
    EXT_STRIDE: ct.Constant[int],  # arg5.stride(2) = 2048
):
    pid = ct.bid(0)
    rows = pid * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    cols = ct.arange(BLOCK_N, dtype=ct.int32)

    batch = rows // (HEADS * SEQ)
    rem = rows - batch * (HEADS * SEQ)
    head = rem // SEQ
    query = rem - head * SEQ

    row_off_2d = ct.reshape(rows, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    row_offs = row_off_2d * SEQ + cols_2d  # (BLOCK_M, BLOCK_N)

    grad = ct.astype(
        ct.gather(grad_flat, row_offs, padding_value=0.0), ct.float32
    )
    logits_bf = ct.gather(logits_flat, row_offs, padding_value=0.0)

    # external_mask[query, cols] with stride 2048
    q_2d = ct.reshape(query, (BLOCK_M, 1))
    ext_off = q_2d * EXT_STRIDE + cols_2d
    external_keep_raw = ct.gather(external_mask_flat, ext_off, padding_value=False)
    external_keep = external_keep_raw != False  # noqa: E712

    neg_min_bf16_tile = ct.full((BLOCK_M, BLOCK_N), NEG_MIN_BF16, dtype=ct.bfloat16)
    masked_logits_bf = ct.where(external_keep, logits_bf, neg_min_bf16_tile)
    masked_logits = ct.astype(masked_logits_bf, ct.float32)

    # q_pos = positions[query]
    q_pos = ct.gather(positions_flat, query, padding_value=0)  # (BLOCK_M,) i64
    # k_pos = positions[cols]
    k_pos = ct.gather(positions_flat, cols, padding_value=0)  # (BLOCK_N,) i64
    # q_segment = segments[batch*128 + q_pos]  (BLOCK_M,)
    batch_i32 = batch
    q_pos_i32 = ct.astype(q_pos, ct.int32)
    q_seg_off = batch_i32 * SEQ + q_pos_i32
    q_segment = ct.gather(segments_flat, q_seg_off, padding_value=0)
    # k_segment = segments[batch[:,None] * 128 + k_pos[None,:]]
    k_pos_i32 = ct.astype(k_pos, ct.int32)
    batch_2d = ct.reshape(batch_i32, (BLOCK_M, 1))
    k_pos_2d = ct.reshape(k_pos_i32, (1, BLOCK_N))
    k_seg_off = batch_2d * SEQ + k_pos_2d
    k_segment = ct.gather(segments_flat, k_seg_off, padding_value=0)

    # structured_keep = (k_pos <= q_pos) & (k_segment == q_segment)
    q_pos_2d = ct.reshape(q_pos_i32, (BLOCK_M, 1))
    causal = k_pos_2d <= q_pos_2d
    same_seg = k_segment == ct.reshape(q_segment, (BLOCK_M, 1))
    structured_keep = causal & same_seg

    ms = ct.gather(mask_scalar, ct.arange(1, dtype=ct.int32), padding_value=0.0)
    ms_scalar = ms  # (1,) tile
    # broadcast ms_scalar to (BLOCK_M, BLOCK_N)
    ms_bcast = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32)
    ms_bcast = ms_bcast + ct.reshape(ms_scalar, (1, 1))
    neg_min_f32_tile = ct.full((BLOCK_M, BLOCK_N), NEG_MIN_F32, dtype=ct.float32)
    mask_bias = ct.where(structured_keep, ms_bcast, neg_min_f32_tile)

    row_index = (batch * HEADS + head) * SEQ + query
    row_shift = ct.gather(row_shift_flat, row_index, padding_value=0.0)
    row_denom = ct.gather(row_denom_flat, row_index, padding_value=0.0)

    scores = masked_logits + mask_bias
    row_shift_2d = ct.reshape(row_shift, (BLOCK_M, 1))
    row_denom_2d = ct.reshape(row_denom, (BLOCK_M, 1))
    probs = ct.exp(scores - row_shift_2d) / row_denom_2d
    product = grad * probs
    row_sum = ct.sum(product, axis=1, keepdims=True)
    value_f = -probs * row_sum + product
    value_bf = ct.astype(value_f, ct.bfloat16)

    fill = ct.gather(fill_scalar, ct.arange(1, dtype=ct.int32), padding_value=0.0)
    fill_bcast = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    fill_bcast = fill_bcast + ct.reshape(fill, (1, 1))
    out_tile = ct.where(external_keep, value_bf, fill_bcast)
    ct.scatter(out_flat, row_offs, out_tile)


@oracle_impl(hardware="B200", point="c5abdf2a", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
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

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(N_ROWS, BLOCK_M), 1, 1),
        _gptneo_attn_backward_kernel,
        (
            grad.reshape(-1),
            positions.reshape(-1),
            segments.reshape(-1),
            mask_scalar.reshape(1),
            logits.reshape(-1),
            external_mask.reshape(-1),
            row_shift.reshape(-1),
            row_denom.reshape(-1),
            fill_scalar.reshape(1),
            out.reshape(-1),
            int(external_mask.stride(2)),
        ),
    )
    return out
