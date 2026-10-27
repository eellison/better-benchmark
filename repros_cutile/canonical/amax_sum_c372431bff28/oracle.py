"""cuTile port of amax_sum_c372431bff28: GPT-Neo masked-attention softmax.

For each row: build a same-segment causal mask, combine with external attn mask,
compute stable f32 softmax, produce bf16 probs and an add_mask side output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 32
HEADS = 16
SEQ = 128
BATCH_HEADS = BATCH * HEADS  # 512
N_ROWS = BATCH_HEADS * SEQ  # 65536
BLOCK_M = 8
BLOCK_N = SEQ  # 128 (power-of-2)
NEG_MIN_BF16 = -3.3895313892515355e38
NEG_MIN_F32 = -3.4028234663852886e38


@ct.kernel
def _gptneo_masked_softmax_kernel(
    attn_mask_flat,   # 1D bool: full 1x1x2048x2048 -> we index [query * stride + cols]
    scores_flat,      # 1D f32: BATCH_HEADS*SEQ*SEQ
    index_flat,       # 1D i64: BATCH*SEQ
    add_mask_flat,    # 1D bf16: BATCH*SEQ*SEQ
    out_flat,         # 1D bf16: BATCH_HEADS*SEQ*SEQ
    N_ROWS_C: ct.Constant[int],
    FULL_MASK_STRIDE: ct.Constant[int],
    ATTN_MASK_NUMEL: ct.Constant[int],
):
    pid = ct.bid(0)
    rows = pid * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    row_mask = rows < N_ROWS_C

    flat_bh = rows // SEQ
    batch = flat_bh // HEADS
    head = flat_bh - batch * HEADS
    query = rows - flat_bh * SEQ

    q_seg_off = batch * SEQ + query
    q_seg = ct.gather(index_flat, q_seg_off, mask=row_mask, padding_value=0)

    batch_2d = ct.reshape(batch, (BLOCK_M, 1))
    cols_2d = ct.reshape(cols, (1, BLOCK_N))
    row_mask_2d = ct.reshape(row_mask, (BLOCK_M, 1))
    k_seg_off = batch_2d * SEQ + cols_2d
    k_seg = ct.gather(index_flat, k_seg_off, mask=row_mask_2d, padding_value=0)

    q_2d = ct.reshape(query, (BLOCK_M, 1))
    causal = cols_2d <= q_2d
    same_segment = k_seg == ct.reshape(q_seg, (BLOCK_M, 1))
    local_keep = causal & same_segment

    neg_min_bf16 = ct.full((BLOCK_M, BLOCK_N), NEG_MIN_BF16, dtype=ct.bfloat16)
    zero_bf16 = ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.bfloat16)
    add_mask = ct.where(local_keep, zero_bf16, neg_min_bf16)

    side_off = ct.reshape(batch, (BLOCK_M, 1)) * (SEQ * SEQ) + q_2d * SEQ + cols_2d
    head_zero_2d = ct.reshape(head == 0, (BLOCK_M, 1))
    side_mask = row_mask_2d & head_zero_2d
    ct.scatter(add_mask_flat, side_off, add_mask, mask=side_mask)

    ext_off = q_2d * FULL_MASK_STRIDE + cols_2d
    external_keep_raw = ct.gather(
        attn_mask_flat, ext_off, mask=row_mask_2d, padding_value=False
    )
    external_keep = external_keep_raw != False  # noqa: E712

    score_off = ct.reshape(flat_bh, (BLOCK_M, 1)) * (SEQ * SEQ) + q_2d * SEQ + cols_2d
    score = ct.gather(scores_flat, score_off, mask=row_mask_2d, padding_value=0.0)

    neg_f32 = ct.full((BLOCK_M, BLOCK_N), NEG_MIN_F32, dtype=ct.float32)
    ninf_f32 = ct.full((BLOCK_M, BLOCK_N), float("-inf"), dtype=ct.float32)
    masked_score = ct.where(external_keep, score, neg_f32)
    logits = ct.where(
        row_mask_2d,
        masked_score + ct.astype(add_mask, ct.float32),
        ninf_f32,
    )

    row_max = ct.max(logits, axis=1, keepdims=True)
    numer = ct.exp(logits - row_max)
    numer = ct.where(row_mask_2d, numer, ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32))
    denom = ct.sum(numer, axis=1, keepdims=True)
    probs = ct.astype(numer / denom, ct.bfloat16)

    ct.scatter(out_flat, score_off, probs, mask=row_mask_2d)


@oracle_impl(hardware="B200", point="5911cf96", BLOCK_M=8, BLOCK_N=128)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int):
    arg0_1, arg1_1, arg2_1, shape0, shape1, shape2, shape3 = inputs
    del shape0, shape1, shape2

    batch = int(arg2_1.shape[0])
    seq_len = int(arg2_1.shape[1])
    heads = int(arg1_1.shape[0] // batch)
    n_rows = int(arg1_1.shape[0] * seq_len)

    add_mask = torch.empty_strided(
        (batch, 1, seq_len, seq_len),
        (seq_len * seq_len, seq_len * seq_len, seq_len, 1),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape3),
        tuple(int(stride) for stride in arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    full_mask_stride = int(arg0_1.stride(2))

    # 1D flat views. attn_mask has shape (1,1,2048,2048); view as flat.
    attn_flat = arg0_1.reshape(-1)
    scores_flat = arg1_1.reshape(-1)
    index_flat = arg2_1.reshape(-1)
    add_mask_flat = add_mask.reshape(-1)
    out_flat = out.reshape(-1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_rows, BLOCK_M), 1, 1),
        _gptneo_masked_softmax_kernel,
        (
            attn_flat,
            scores_flat,
            index_flat,
            add_mask_flat,
            out_flat,
            n_rows,
            full_mask_stride,
            attn_flat.numel(),
        ),
    )
    return add_mask, out
