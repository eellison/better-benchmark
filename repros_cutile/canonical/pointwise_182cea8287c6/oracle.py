"""cuTile port of pointwise_182cea8287c6: Gemma dynamic RoPE trig + rotary q/k repeat.

Two cuTile kernels: (1) generate cos/sin table (bf16) from inv_freq * seq; (2)
apply rotate-half + multiply for q (8 heads) and k (4 heads), plus the grouped-key
repeat (each of the 4 k-heads is replicated into 2 output q-heads).

Returns: cos_bf16, sin_bf16, out_q, out_repeat, out_k (slice of out_repeat)
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
TRIG_NUMEL = BATCH * SEQ * HEAD_DIM  # 256000


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _rope_trig_kernel(
    inv_freq_ptr,     # bf16 [HALF=128]
    cos_ptr,          # bf16 [SEQ*HEAD_DIM]  (flat)
    sin_ptr,          # bf16 [SEQ*HEAD_DIM]  (flat)
    SEQ_LEN: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    BLOCK_SEQ: ct.Constant[int],
    BLOCK_DIM: ct.Constant[int],
):
    """Grid: (seq_blocks, dim_blocks, 1). Each block writes a (BLOCK_SEQ, BLOCK_DIM) tile."""
    s_block = ct.bid(0)
    d_block = ct.bid(1)

    dims = d_block * BLOCK_DIM + ct.arange(BLOCK_DIM, dtype=ct.int32)
    # freq index = dim % HALF_DIM  (HEAD_DIM = 2*HALF, so first half + second half same freqs)
    freq_idx = dims - (dims // HALF_DIM) * HALF_DIM

    seqs = s_block * BLOCK_SEQ + ct.arange(BLOCK_SEQ, dtype=ct.int32)

    # Load inv_freq via gather since freq_idx uses %.
    inv_freq_bf = ct.gather(inv_freq_ptr, (freq_idx,))
    inv_freq_f = ct.astype(inv_freq_bf, ct.float32)  # (BLOCK_DIM,)

    seqs_f = ct.astype(seqs, ct.float32)  # (BLOCK_SEQ,)
    # phase[i, j] = seq_i * inv_freq_j
    seqs_2d = ct.reshape(seqs_f, (BLOCK_SEQ, 1))
    inv_2d = ct.reshape(inv_freq_f, (1, BLOCK_DIM))
    phase = seqs_2d * inv_2d  # (BLOCK_SEQ, BLOCK_DIM)

    cos_val = ct.astype(ct.cos(phase), ct.bfloat16)
    sin_val = ct.astype(ct.sin(phase), ct.bfloat16)

    ct.store(cos_ptr, index=(s_block, d_block), tile=cos_val)
    ct.store(sin_ptr, index=(s_block, d_block), tile=sin_val)


@ct.kernel
def _rotary_q_kernel(
    q_ptr,        # bf16 [SEQ, Q_HEADS, HEAD_DIM]  (viewed as [BATCH*SEQ, Q_HEADS, HEAD_DIM])
    cos_ptr,      # bf16 [SEQ, HEAD_DIM]
    sin_ptr,      # bf16 [SEQ, HEAD_DIM]
    out_q_ptr,    # bf16 [SEQ, Q_HEADS, HEAD_DIM]  contiguous
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    Q_HEADS_C: ct.Constant[int],
):
    """Grid: (SEQ, Q_HEADS, 1). One tile per (seq, head)."""
    seq = ct.bid(0)
    head = ct.bid(1)

    q_row = ct.load(q_ptr, index=(seq, head, 0), shape=(1, 1, HEAD_DIM_C))
    q_row_f = ct.astype(ct.reshape(q_row, (HEAD_DIM_C,)), ct.float32)  # (HEAD_DIM,)

    cos_row = ct.load(cos_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    sin_row = ct.load(sin_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    cos_f = ct.astype(ct.reshape(cos_row, (HEAD_DIM_C,)), ct.float32)
    sin_f = ct.astype(ct.reshape(sin_row, (HEAD_DIM_C,)), ct.float32)

    # rotate-half: [x_hi, x_lo] -> [-x_hi, x_lo] where lo=first half, hi=second half.
    # For dim d in [0, HALF): partner is q[d+HALF], sign=-1 (i.e. rot_val = -q[d+HALF])
    # For dim d in [HALF, HEAD): partner is q[d-HALF], sign=+1 (rot_val = +q[d-HALF])
    # We can compute this by gathering across dims via a rotate.
    # Simpler: use gather.
    dims = ct.arange(HEAD_DIM_C, dtype=ct.int32)
    # partner_dim: if d < HALF: d + HALF, else d - HALF
    is_lo = dims < HALF_DIM
    partner = ct.where(is_lo, dims + HALF_DIM, dims - HALF_DIM)
    # sign: -1 for lo half, +1 for hi half
    sign = ct.where(is_lo, ct.full((HEAD_DIM_C,), -1.0, dtype=ct.float32),
                    ct.full((HEAD_DIM_C,), 1.0, dtype=ct.float32))
    partner_val = ct.gather(q_ptr, (ct.full((HEAD_DIM_C,), seq, dtype=ct.int32),
                                    ct.full((HEAD_DIM_C,), head, dtype=ct.int32),
                                    partner))
    partner_f = ct.astype(partner_val, ct.float32)
    rotated = sign * partner_f

    # Round q*cos and rotated*sin each to bf16 (matches inline_ptx cvt.rn.bf16.f32).
    q_cos = ct.astype(ct.astype(q_row_f * cos_f, ct.bfloat16), ct.float32)
    rot_sin = ct.astype(ct.astype(rotated * sin_f, ct.bfloat16), ct.float32)
    out_f = q_cos + rot_sin
    out_bf = ct.astype(out_f, ct.bfloat16)

    out_bf_3d = ct.reshape(out_bf, (1, 1, HEAD_DIM_C))
    ct.store(out_q_ptr, index=(seq, head, 0), tile=out_bf_3d)


@ct.kernel
def _rotary_k_repeat_kernel(
    k_ptr,        # bf16 [SEQ, K_HEADS, HEAD_DIM]
    cos_ptr,      # bf16 [SEQ, HEAD_DIM]
    sin_ptr,      # bf16 [SEQ, HEAD_DIM]
    out_k_ptr,    # bf16 [SEQ, K_HEADS, HEAD_DIM]  contiguous stride (K_HEADS*HEAD_DIM, HEAD_DIM, 1)
    out_repeat_ptr,   # bf16 [Q_HEADS, SEQ, HEAD_DIM]  contiguous
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM: ct.Constant[int],
    K_HEADS_C: ct.Constant[int],
    GROUPS_C: ct.Constant[int],
    Q_HEADS_C: ct.Constant[int],
):
    """Grid: (SEQ, K_HEADS, 1). One tile per (seq, k_head)."""
    seq = ct.bid(0)
    k_head = ct.bid(1)

    k_row = ct.load(k_ptr, index=(seq, k_head, 0), shape=(1, 1, HEAD_DIM_C))
    k_row_f = ct.astype(ct.reshape(k_row, (HEAD_DIM_C,)), ct.float32)

    cos_row = ct.load(cos_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    sin_row = ct.load(sin_ptr, index=(seq, 0), shape=(1, HEAD_DIM_C))
    cos_f = ct.astype(ct.reshape(cos_row, (HEAD_DIM_C,)), ct.float32)
    sin_f = ct.astype(ct.reshape(sin_row, (HEAD_DIM_C,)), ct.float32)

    dims = ct.arange(HEAD_DIM_C, dtype=ct.int32)
    is_lo = dims < HALF_DIM
    partner = ct.where(is_lo, dims + HALF_DIM, dims - HALF_DIM)
    sign = ct.where(is_lo, ct.full((HEAD_DIM_C,), -1.0, dtype=ct.float32),
                    ct.full((HEAD_DIM_C,), 1.0, dtype=ct.float32))
    partner_val = ct.gather(k_ptr, (ct.full((HEAD_DIM_C,), seq, dtype=ct.int32),
                                    ct.full((HEAD_DIM_C,), k_head, dtype=ct.int32),
                                    partner))
    partner_f = ct.astype(partner_val, ct.float32)
    rotated = sign * partner_f

    k_cos = ct.astype(ct.astype(k_row_f * cos_f, ct.bfloat16), ct.float32)
    rot_sin = ct.astype(ct.astype(rotated * sin_f, ct.bfloat16), ct.float32)
    out_f = k_cos + rot_sin
    out_bf = ct.astype(out_f, ct.bfloat16)

    out_bf_3d = ct.reshape(out_bf, (1, 1, HEAD_DIM_C))
    ct.store(out_k_ptr, index=(seq, k_head, 0), tile=out_bf_3d)

    # Write to out_repeat at (group0_head, seq, :) and (group1_head, seq, :)
    for group in ct.static_iter(range(GROUPS_C)):
        out_head = k_head * GROUPS_C + group
        out_repeat_3d = ct.reshape(out_bf, (1, 1, HEAD_DIM_C))
        ct.store(out_repeat_ptr, index=(out_head, seq, 0), tile=out_repeat_3d)


@oracle_impl(hardware="B200", point="5da3d9cf", TRIG_BLOCK=256, BLOCK_PAIRS=256)
def oracle_forward(inputs, *, TRIG_BLOCK: int, BLOCK_PAIRS: int):
    q_mm, inv_freq, k_mm, *_shape_params = inputs
    device = q_mm.device

    # Trig table: bf16 [1, SEQ, HEAD_DIM] contiguous.
    cos = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM), (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    sin = torch.empty_strided(
        (BATCH, SEQ, HEAD_DIM), (SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    # For the trig kernel we view [SEQ, HEAD_DIM] (BATCH=1).
    cos_2d = cos.view(SEQ, HEAD_DIM)
    sin_2d = sin.view(SEQ, HEAD_DIM)

    BLOCK_SEQ = 8
    BLOCK_DIM = 64
    if SEQ % BLOCK_SEQ != 0:
        # Fall back to safer blocks if needed.
        BLOCK_SEQ = 1
    if HEAD_DIM % BLOCK_DIM != 0:
        BLOCK_DIM = HEAD_DIM
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (SEQ // BLOCK_SEQ, HEAD_DIM // BLOCK_DIM, 1),
        _rope_trig_kernel,
        (inv_freq, cos_2d, sin_2d, SEQ, HEAD_DIM, HALF, BLOCK_SEQ, BLOCK_DIM),
    )

    # out_q has strided layout (Q_HEADS*SEQ*HEAD_DIM, HEAD_DIM, Q_HEADS*HEAD_DIM, 1)
    # i.e. shape [BATCH, Q_HEADS, SEQ, HEAD_DIM] but stride 1 dim last, HEAD_DIM in second-to-last.
    # Stride order shows this is actually [BATCH, SEQ, Q_HEADS, HEAD_DIM] permuted -> [BATCH, Q_HEADS, SEQ, HEAD_DIM].
    # Simplest: allocate a contiguous [SEQ, Q_HEADS, HEAD_DIM] tensor, then produce
    # the strided view via permute.
    out_q_contig = torch.empty(
        (SEQ, Q_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    q_view = q_mm.view(SEQ, Q_HEADS, HEAD_DIM)  # bf16 [1000, 8, 256]
    ct.launch(
        stream,
        (SEQ, Q_HEADS, 1),
        _rotary_q_kernel,
        (q_view, cos_2d, sin_2d, out_q_contig, HEAD_DIM, HALF, Q_HEADS),
    )
    # Reshape (SEQ, Q_HEADS, HEAD_DIM) -> (BATCH, Q_HEADS, SEQ, HEAD_DIM) with proper stride.
    # out_q permute (1, 0, 2) gives [Q_HEADS, SEQ, HEAD_DIM]; unsqueeze(0) gives [1, ...].
    out_q_result = out_q_contig.permute(1, 0, 2).unsqueeze(0).contiguous(
        memory_format=torch.contiguous_format
    )
    # But target strides are (Q*S*H, H, Q*H, 1). Just build strided empty and copy.
    out_q = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    # Stride layout of out_q: [B, Q, S, H] with stride (Q*S*H, H, Q*H, 1)
    # This is equivalent to [B, S, Q, H] contiguous permuted to [B, Q, S, H].
    # So we can copy from out_q_contig (viewed as [1, S, Q, H]) permuted:
    out_q_view = out_q.permute(0, 2, 1, 3)  # [B, S, Q, H] view
    out_q_view.copy_(out_q_contig.view(1, SEQ, Q_HEADS, HEAD_DIM))

    # out_repeat has stride (Q*S*H, S*H, H, 1) -> shape [B, Q, S, H] contiguous.
    out_repeat = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ, HEAD_DIM),
        (Q_HEADS * SEQ * HEAD_DIM, SEQ * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    # out_k has stride (K*S*H, H, K*H, 1) -> like out_q, [B, S, K, H] permuted.
    out_k = torch.empty_strided(
        (BATCH, K_HEADS, SEQ, HEAD_DIM),
        (K_HEADS * SEQ * HEAD_DIM, HEAD_DIM, K_HEADS * HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    # Allocate contiguous scratch tensors for the kernel.
    out_k_contig = torch.empty((SEQ, K_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_repeat_contig = torch.empty((Q_HEADS, SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)

    k_view = k_mm.view(SEQ, K_HEADS, HEAD_DIM)
    ct.launch(
        stream,
        (SEQ, K_HEADS, 1),
        _rotary_k_repeat_kernel,
        (k_view, cos_2d, sin_2d, out_k_contig, out_repeat_contig,
         HEAD_DIM, HALF, K_HEADS, GROUPS, Q_HEADS),
    )

    # Copy scratch -> strided outputs.
    out_k_view = out_k.permute(0, 2, 1, 3)  # [B, S, K, H]
    out_k_view.copy_(out_k_contig.view(1, SEQ, K_HEADS, HEAD_DIM))

    # out_repeat: [B, Q, S, H] contiguous.  Our scratch is [Q, S, H] which matches directly.
    out_repeat.copy_(out_repeat_contig.view(1, Q_HEADS, SEQ, HEAD_DIM))

    # Triton returns cos, sin, out_q, out_repeat, slice(out_k, dim=2, -4095:)
    # slice with -4095: on a size-1000 dim gives the full range (start clamped to 0).
    return cos, sin, out_q, out_repeat, out_k
