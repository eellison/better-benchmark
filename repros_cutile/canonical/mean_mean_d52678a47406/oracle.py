"""cuTile port of mean_mean_d52678a47406: Gemma-3 dynamic RoPE + RMSNorm.

Three cuTile kernels:
  1) _positions_kernel: writes positions[1, 1000] = 0..999 (i64)
  2) _rope_trig_kernel: generates cos/sin[1, 1000, 256] (bf16)
  3) _rmsnorm_rope_kernel: RMSNorm(weight+1.0) + rotate-half rotary
     applied to q (Q_HEADS=8) or k (K_HEADS=4) with output tiles.
     For k, also writes to the repeated_key output at each of the
     GROUPS positions.

Returns (positions, cos, sin, out_q, out_repeat, out_k). Layouts match
the Triton oracle's strided outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 1
SEQ = 1000
Q_HEADS = 8
K_HEADS = 4
HEAD_DIM = 256
HALF = 128
GROUPS = Q_HEADS // K_HEADS
EPS = 1.0e-6


@ct.kernel
def _rope_trig_kernel(
    inv_freq_ptr,     # bf16 [HALF=128]
    cos_ptr,          # bf16 [SEQ, HEAD_DIM] flat 2D
    sin_ptr,          # bf16 [SEQ, HEAD_DIM]
    SEQ_LEN: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    BLOCK_SEQ: ct.Constant[int],
    BLOCK_DIM: ct.Constant[int],
):
    s_block = ct.bid(0)
    d_block = ct.bid(1)

    dims = d_block * BLOCK_DIM + ct.arange(BLOCK_DIM, dtype=ct.int32)
    freq_idx = dims - (dims // HALF_DIM) * HALF_DIM

    seqs = s_block * BLOCK_SEQ + ct.arange(BLOCK_SEQ, dtype=ct.int32)

    inv_freq_bf = ct.gather(inv_freq_ptr, (freq_idx,))
    inv_freq_f = ct.astype(inv_freq_bf, ct.float32)

    seqs_f = ct.astype(seqs, ct.float32)
    seqs_2d = ct.reshape(seqs_f, (BLOCK_SEQ, 1))
    inv_2d = ct.reshape(inv_freq_f, (1, BLOCK_DIM))
    phase = seqs_2d * inv_2d

    cos_val = ct.astype(ct.cos(phase), ct.bfloat16)
    sin_val = ct.astype(ct.sin(phase), ct.bfloat16)

    ct.store(cos_ptr, index=(s_block, d_block), tile=cos_val)
    ct.store(sin_ptr, index=(s_block, d_block), tile=sin_val)


@ct.kernel
def _rmsnorm_rope_q_kernel(
    q_ptr,        # bf16 [SEQ, Q_HEADS, HEAD_DIM]
    weight_ptr,   # bf16 [HEAD_DIM]
    cos_ptr,      # bf16 [SEQ, HEAD_DIM]
    sin_ptr,      # bf16 [SEQ, HEAD_DIM]
    out_q_ptr,    # bf16 [SEQ, Q_HEADS, HEAD_DIM] contiguous
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    Q_HEADS_C: ct.Constant[int],
    EPS_VALUE: ct.Constant[float],
):
    seq = ct.bid(0)
    head = ct.bid(1)

    q_row = ct.load(q_ptr, index=(seq, head, 0), shape=(1, 1, HEAD_DIM_C))
    q_1d = ct.astype(ct.reshape(q_row, (HEAD_DIM_C,)), ct.float32)

    # RMSNorm: rsqrt(mean(x^2) + eps) * x * (weight+1)
    sq_sum = ct.sum(q_1d * q_1d)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HEAD_DIM_C) + EPS_VALUE)

    weight = ct.load(weight_ptr, index=(0,), shape=(HEAD_DIM_C,))
    weight_f = ct.astype(weight, ct.float32) + 1.0

    # Normalized-and-affine, then cast to bf16 (matches Triton's convert_element_type)
    normed_f = q_1d * inv_rms * weight_f
    normed_bf = ct.astype(normed_f, ct.bfloat16)
    normed_after_bf = ct.astype(normed_bf, ct.float32)

    # Rotary: rotate-half
    dims = ct.arange(HEAD_DIM_C, dtype=ct.int32)
    is_lo = dims < HALF_DIM
    partner = ct.where(is_lo, dims + HALF_DIM, dims - HALF_DIM)
    # gather partner values from normed_bf via a rebuild.
    # cuTile can't gather from a tile; we recompute via loading q partner and applying RMSNorm again.
    # However, since RMSNorm gain factor for partner is the same in units of inv_rms
    # (per-row scalar) times a different weight index, we do that:
    partner_weight_bf = ct.gather(weight_ptr, (partner,))
    partner_weight_f = ct.astype(partner_weight_bf, ct.float32) + 1.0
    # partner values in q
    partner_q_bf = ct.gather(q_ptr, (ct.full((HEAD_DIM_C,), seq, dtype=ct.int32),
                                      ct.full((HEAD_DIM_C,), head, dtype=ct.int32),
                                      partner))
    partner_q_f = ct.astype(partner_q_bf, ct.float32)
    partner_normed_f = partner_q_f * inv_rms * partner_weight_f
    partner_normed_bf = ct.astype(partner_normed_f, ct.bfloat16)
    partner_normed_after_f = ct.astype(partner_normed_bf, ct.float32)

    # rotated: sign * partner_normed  (-1 for lo half, +1 for hi half)
    sign = ct.where(
        is_lo,
        ct.full(shape=(HEAD_DIM_C,), fill_value=-1.0, dtype=ct.float32),
        ct.full(shape=(HEAD_DIM_C,), fill_value=1.0, dtype=ct.float32),
    )
    rotated_f = sign * partner_normed_after_f

    cos_row = ct.load(cos_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    sin_row = ct.load(sin_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    cos_f = ct.astype(ct.reshape(cos_row, (HEAD_DIM_C,)), ct.float32)
    sin_f = ct.astype(ct.reshape(sin_row, (HEAD_DIM_C,)), ct.float32)

    # Match Triton: out = round_bf16(value*cos) + round_bf16(rot_value*sin)
    val_cos_bf = ct.astype(normed_after_bf * cos_f, ct.bfloat16)
    val_cos_f = ct.astype(val_cos_bf, ct.float32)
    rot_sin_bf = ct.astype(rotated_f * sin_f, ct.bfloat16)
    rot_sin_f = ct.astype(rot_sin_bf, ct.float32)
    out_f = val_cos_f + rot_sin_f
    out_bf = ct.astype(out_f, ct.bfloat16)
    out_bf_3d = ct.reshape(out_bf, (1, 1, HEAD_DIM_C))
    ct.store(out_q_ptr, index=(seq, head, 0), tile=out_bf_3d)


@ct.kernel
def _rmsnorm_rope_k_kernel(
    k_ptr,        # bf16 [SEQ, K_HEADS, HEAD_DIM]
    weight_ptr,   # bf16 [HEAD_DIM]
    cos_ptr,      # bf16 [SEQ, HEAD_DIM]
    sin_ptr,      # bf16 [SEQ, HEAD_DIM]
    out_k_ptr,    # bf16 [SEQ, K_HEADS, HEAD_DIM] contiguous
    out_repeat_ptr,   # bf16 [Q_HEADS, SEQ, HEAD_DIM] contiguous
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    K_HEADS_C: ct.Constant[int],
    GROUPS_C: ct.Constant[int],
    Q_HEADS_C: ct.Constant[int],
    EPS_VALUE: ct.Constant[float],
):
    seq = ct.bid(0)
    k_head = ct.bid(1)

    k_row = ct.load(k_ptr, index=(seq, k_head, 0), shape=(1, 1, HEAD_DIM_C))
    k_1d = ct.astype(ct.reshape(k_row, (HEAD_DIM_C,)), ct.float32)
    sq_sum = ct.sum(k_1d * k_1d)
    inv_rms = ct.rsqrt(sq_sum * (1.0 / HEAD_DIM_C) + EPS_VALUE)

    weight = ct.load(weight_ptr, index=(0,), shape=(HEAD_DIM_C,))
    weight_f = ct.astype(weight, ct.float32) + 1.0

    normed_f = k_1d * inv_rms * weight_f
    normed_bf = ct.astype(normed_f, ct.bfloat16)
    normed_after_bf = ct.astype(normed_bf, ct.float32)

    dims = ct.arange(HEAD_DIM_C, dtype=ct.int32)
    is_lo = dims < HALF_DIM
    partner = ct.where(is_lo, dims + HALF_DIM, dims - HALF_DIM)
    partner_weight_bf = ct.gather(weight_ptr, (partner,))
    partner_weight_f = ct.astype(partner_weight_bf, ct.float32) + 1.0
    partner_k_bf = ct.gather(k_ptr, (ct.full((HEAD_DIM_C,), seq, dtype=ct.int32),
                                      ct.full((HEAD_DIM_C,), k_head, dtype=ct.int32),
                                      partner))
    partner_k_f = ct.astype(partner_k_bf, ct.float32)
    partner_normed_f = partner_k_f * inv_rms * partner_weight_f
    partner_normed_bf = ct.astype(partner_normed_f, ct.bfloat16)
    partner_normed_after_f = ct.astype(partner_normed_bf, ct.float32)

    sign = ct.where(
        is_lo,
        ct.full(shape=(HEAD_DIM_C,), fill_value=-1.0, dtype=ct.float32),
        ct.full(shape=(HEAD_DIM_C,), fill_value=1.0, dtype=ct.float32),
    )
    rotated_f = sign * partner_normed_after_f

    cos_row = ct.load(cos_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    sin_row = ct.load(sin_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    cos_f = ct.astype(ct.reshape(cos_row, (HEAD_DIM_C,)), ct.float32)
    sin_f = ct.astype(ct.reshape(sin_row, (HEAD_DIM_C,)), ct.float32)

    val_cos_bf = ct.astype(normed_after_bf * cos_f, ct.bfloat16)
    val_cos_f = ct.astype(val_cos_bf, ct.float32)
    rot_sin_bf = ct.astype(rotated_f * sin_f, ct.bfloat16)
    rot_sin_f = ct.astype(rot_sin_bf, ct.float32)
    out_f = val_cos_f + rot_sin_f
    out_bf = ct.astype(out_f, ct.bfloat16)
    out_bf_3d = ct.reshape(out_bf, (1, 1, HEAD_DIM_C))
    ct.store(out_k_ptr, index=(seq, k_head, 0), tile=out_bf_3d)

    # Repeat write for grouped-key output: for each group, store at (out_head, seq, :)
    for group in ct.static_iter(range(GROUPS_C)):
        out_head = k_head * GROUPS_C + group
        ct.store(out_repeat_ptr, index=(out_head, seq, 0), tile=out_bf_3d)


@oracle_impl(hardware="B200", point="767ad01a", TRIG_BLOCK=256, BLOCK_ROWS=2)
def oracle_forward(inputs, *, TRIG_BLOCK: int, BLOCK_ROWS: int):
    del TRIG_BLOCK, BLOCK_ROWS
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = inputs[:5]
    device = arg0_1.device

    # 1) positions: i64[1, 1000] contiguous
    positions = torch.empty_strided(
        (BATCH, SEQ),
        (SEQ, 1),
        device=device,
        dtype=torch.int64,
    )

    # 2) cos/sin: bf16[1, 1000, 256] contiguous
    cos = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    sin = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM),
        (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # 2D flat views for the trig kernel
    cos_2d = cos.view(SEQ, HEAD_DIM)
    sin_2d = sin.view(SEQ, HEAD_DIM)

    stream = torch.cuda.current_stream()

    # Positions: fill with arange (matches iota+unsqueeze in Repro)
    torch.arange(SEQ, dtype=torch.int64, device=device, out=positions.view(SEQ))

    # RoPE trig
    BLOCK_SEQ = 8
    BLOCK_DIM = 64
    ct.launch(
        stream,
        (SEQ // BLOCK_SEQ, HEAD_DIM // BLOCK_DIM, 1),
        _rope_trig_kernel,
        (arg2_1, cos_2d, sin_2d, SEQ, HEAD_DIM, HALF, BLOCK_SEQ, BLOCK_DIM),
    )
    # SEQ=1000, BLOCK_SEQ=8 -> 125 blocks. 8*125 = 1000 exact. HEAD_DIM=256, BLOCK_DIM=64 -> 4 blocks. OK.

    # q out contiguous
    out_q_contig = torch.empty(
        (SEQ, Q_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    q_view = arg0_1.view(SEQ, Q_HEADS, HEAD_DIM)
    ct.launch(
        stream,
        (SEQ, Q_HEADS, 1),
        _rmsnorm_rope_q_kernel,
        (q_view, arg1_1, cos_2d, sin_2d, out_q_contig,
         HEAD_DIM, HALF, Q_HEADS, EPS),
    )

    # out_q strided
    out_q = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_q_view = out_q.permute(0, 2, 1, 3)
    out_q_view.copy_(out_q_contig.view(1, SEQ, Q_HEADS, HEAD_DIM))

    # out_repeat: contiguous [B, Q_HEADS, SEQ, HEAD_DIM]
    out_repeat = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    # k output strided
    out_k = torch.empty_strided(
        (BATCH, K_HEADS, SEQ, HEAD_DIM),
        (K_HEADS * SEQ * HEAD_DIM, HEAD_DIM, K_HEADS * HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_k_contig = torch.empty((SEQ, K_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_repeat_contig = torch.empty(
        (Q_HEADS, SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16
    )

    k_view = arg3_1.view(SEQ, K_HEADS, HEAD_DIM)
    ct.launch(
        stream,
        (SEQ, K_HEADS, 1),
        _rmsnorm_rope_k_kernel,
        (k_view, arg4_1, cos_2d, sin_2d, out_k_contig, out_repeat_contig,
         HEAD_DIM, HALF, K_HEADS, GROUPS, Q_HEADS, EPS),
    )

    out_k_view = out_k.permute(0, 2, 1, 3)
    out_k_view.copy_(out_k_contig.view(1, SEQ, K_HEADS, HEAD_DIM))
    out_repeat.copy_(out_repeat_contig.view(1, Q_HEADS, SEQ, HEAD_DIM))

    return positions, cos, sin, out_q, out_repeat, out_k
