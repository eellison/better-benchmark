"""cuTile port of pointwise_19870c5a1252: GPT-oss rotary embeddings + grouped-key repeat.

Mirrors the Triton oracle: does the rotary math inside cuTile kernels (Q and K
paths split into two launches because Q covers 64 heads while K only 8 with 8
repeats each). All bf16 rounding boundaries match the Triton `cvt.rn.bf16`
semantics via ct.astype defaults.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 1000
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 64
HALF = 32
GROUPS = Q_HEADS // KV_HEADS
TAIL = 127
TAIL_START = SEQ - TAIL
BLOCK_SEQ = 8  # largest power of 2 dividing SEQ=1000


@ct.kernel
def _q_rotary_kernel(
    q_ptr,        # bf16 [SEQ, QH, D]
    cos_ptr,      # bf16 [SEQ, HALF]  (strided)
    sin_ptr,      # bf16 [SEQ, HALF]
    out_q_ptr,    # bf16 [QH, SEQ, D]
):
    q_head = ct.bid(0)
    seq_block = ct.bid(1)

    q_lo_3d = ct.load(q_ptr, index=(seq_block, q_head, 0), shape=(BLOCK_SEQ, 1, HALF))
    q_hi_3d = ct.load(q_ptr, index=(seq_block, q_head, 1), shape=(BLOCK_SEQ, 1, HALF))
    q_lo = ct.reshape(q_lo_3d, (BLOCK_SEQ, HALF))
    q_hi = ct.reshape(q_hi_3d, (BLOCK_SEQ, HALF))
    q_lo_f = ct.astype(q_lo, ct.float32)
    q_hi_f = ct.astype(q_hi, ct.float32)

    cos = ct.load(cos_ptr, index=(seq_block, 0), shape=(BLOCK_SEQ, HALF))
    sin = ct.load(sin_ptr, index=(seq_block, 0), shape=(BLOCK_SEQ, HALF))
    cos_f = ct.astype(cos, ct.float32)
    sin_f = ct.astype(sin, ct.float32)

    q_mul_lo = ct.astype(ct.astype(q_lo_f * cos_f, ct.bfloat16), ct.float32)
    q_rot_lo = ct.astype(ct.astype(q_hi_f * sin_f, ct.bfloat16), ct.float32)
    q_mul_hi = ct.astype(ct.astype(q_hi_f * cos_f, ct.bfloat16), ct.float32)
    q_rot_hi = ct.astype(ct.astype(q_lo_f * sin_f, ct.bfloat16), ct.float32)
    out_lo = ct.astype(q_mul_lo - q_rot_lo, ct.bfloat16)
    out_hi = ct.astype(q_mul_hi + q_rot_hi, ct.bfloat16)

    out_lo_3d = ct.reshape(out_lo, (1, BLOCK_SEQ, HALF))
    out_hi_3d = ct.reshape(out_hi, (1, BLOCK_SEQ, HALF))
    ct.store(out_q_ptr, index=(q_head, seq_block, 0), tile=out_lo_3d)
    ct.store(out_q_ptr, index=(q_head, seq_block, 1), tile=out_hi_3d)


@ct.kernel
def _k_rotary_repeat_kernel(
    k_ptr,               # bf16 [SEQ, KVH, D]
    cos_ptr,             # bf16 [SEQ, HALF]
    sin_ptr,             # bf16 [SEQ, HALF]
    out_k_base_ptr,      # bf16 [KVH, SEQ, D]
    out_repeat_ptr,      # bf16 [QH, D, SEQ] strides (SEQ*D, 1, D)
    GROUPS_C: ct.Constant[int],
):
    kv_head = ct.bid(0)
    seq_block = ct.bid(1)

    k_lo_3d = ct.load(k_ptr, index=(seq_block, kv_head, 0), shape=(BLOCK_SEQ, 1, HALF))
    k_hi_3d = ct.load(k_ptr, index=(seq_block, kv_head, 1), shape=(BLOCK_SEQ, 1, HALF))
    k_lo = ct.reshape(k_lo_3d, (BLOCK_SEQ, HALF))
    k_hi = ct.reshape(k_hi_3d, (BLOCK_SEQ, HALF))
    k_lo_f = ct.astype(k_lo, ct.float32)
    k_hi_f = ct.astype(k_hi, ct.float32)

    cos = ct.load(cos_ptr, index=(seq_block, 0), shape=(BLOCK_SEQ, HALF))
    sin = ct.load(sin_ptr, index=(seq_block, 0), shape=(BLOCK_SEQ, HALF))
    cos_f = ct.astype(cos, ct.float32)
    sin_f = ct.astype(sin, ct.float32)

    k_mul_lo = ct.astype(ct.astype(k_lo_f * cos_f, ct.bfloat16), ct.float32)
    k_rot_lo = ct.astype(ct.astype(k_hi_f * sin_f, ct.bfloat16), ct.float32)
    k_mul_hi = ct.astype(ct.astype(k_hi_f * cos_f, ct.bfloat16), ct.float32)
    k_rot_hi = ct.astype(ct.astype(k_lo_f * sin_f, ct.bfloat16), ct.float32)
    out_lo = ct.astype(k_mul_lo - k_rot_lo, ct.bfloat16)
    out_hi = ct.astype(k_mul_hi + k_rot_hi, ct.bfloat16)

    out_lo_3d = ct.reshape(out_lo, (1, BLOCK_SEQ, HALF))
    out_hi_3d = ct.reshape(out_hi, (1, BLOCK_SEQ, HALF))
    # K base output (whole SEQ; only the tail slice is returned to the user).
    ct.store(out_k_base_ptr, index=(kv_head, seq_block, 0), tile=out_lo_3d)
    ct.store(out_k_base_ptr, index=(kv_head, seq_block, 1), tile=out_hi_3d)

    # Repeated output has layout [QH, D, SEQ] — transpose the (BLOCK_SEQ, HALF)
    # tile to (HALF, BLOCK_SEQ) and store at (out_head, halfbin, seq_block).
    out_lo_t = ct.transpose(out_lo)
    out_hi_t = ct.transpose(out_hi)
    out_lo_t_3d = ct.reshape(out_lo_t, (1, HALF, BLOCK_SEQ))
    out_hi_t_3d = ct.reshape(out_hi_t, (1, HALF, BLOCK_SEQ))
    for g in ct.static_iter(range(GROUPS_C)):
        out_head = kv_head * GROUPS_C + g
        ct.store(out_repeat_ptr, index=(out_head, 0, seq_block), tile=out_lo_t_3d)
        ct.store(out_repeat_ptr, index=(out_head, 1, seq_block), tile=out_hi_t_3d)


@oracle_impl(hardware="B200", point="fa3cc74f", BLOCK_PAIRS=1024)
def oracle_forward(inputs, *, BLOCK_PAIRS: int):
    q_mm, k_mm, cos, sin, *_shape_params = inputs
    device = q_mm.device

    q_3d = q_mm.view(SEQ, Q_HEADS, HEAD_DIM)
    k_3d = k_mm.view(SEQ, KV_HEADS, HEAD_DIM)
    cos_2d = cos.squeeze(0)  # (SEQ, HALF) with strides (1, SEQ)
    sin_2d = sin.squeeze(0)

    out_q = torch.empty_strided(
        (Q_HEADS, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_k_base = torch.empty_strided(
        (1, KV_HEADS, SEQ, HEAD_DIM),
        (KV_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_k_base_3d = out_k_base.squeeze(0)  # (KV_HEADS, SEQ, D)
    out_repeat = torch.empty_strided(
        (Q_HEADS, HEAD_DIM, SEQ),
        (SEQ * HEAD_DIM, 1, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (Q_HEADS, SEQ // BLOCK_SEQ, 1),
        _q_rotary_kernel,
        (q_3d, cos_2d, sin_2d, out_q),
    )
    ct.launch(
        stream, (KV_HEADS, SEQ // BLOCK_SEQ, 1),
        _k_rotary_repeat_kernel,
        (k_3d, cos_2d, sin_2d, out_k_base_3d, out_repeat, GROUPS),
    )
    out_k_tail = out_k_base[:, :, TAIL_START:, :]
    return out_q, out_repeat, out_k_tail
