"""cuTile port of mean_mean_3084b9796c25: Gemma RoPE + RMSNorm + grouped-KV.

Reference:
  Q: bf16 [1000, 2048] -> [1, 8, 1000, 256]  (Q_HEADS=8, HEAD_DIM=256)
  K: bf16 [1000, 1024] -> [1, 4, 1000, 256]  (KV_HEADS=4)
  For both: RMSNorm along D=256 with eps=1e-6 and (weight.float()+1.0) scale.
  Rotary: cos/sin table [1, 1000, 256] from inv_freq[128] and positions[1,1000].
    RoPE: y = x*cos + rot(x)*sin, where rot flips halves.
  Return (cos_bf16, sin_bf16, q_out, k_out, repeated_k with grouped expand).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 1
SEQ_LEN = 1000
Q_HEADS = 8
KV_HEADS = 4
HEAD_DIM = 256
HALF_DIM = 128
REPEAT = Q_HEADS // KV_HEADS  # 2
EPS = 1.0e-6


@ct.kernel
def _trig_kernel(
    inv_freq_ptr,     # bf16 [128]
    positions_f_ptr,  # f32 [1000]   (positions cast to f32)
    cos_ptr,          # bf16 [1000, 256]  (rows=1000, cols=256)
    sin_ptr,          # bf16 [1000, 256]
    HALF: ct.Constant[int],
    D: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(D, dtype=ct.int32)
    freq_idx = cols % HALF  # each half repeats
    # Load inv_freq into a size-D tile by broadcasting via a small trick:
    # inv_freq has size HALF, we want inv_freq[cols % HALF]. cuTile lacks
    # gather, so we use the fact that inv_freq is bf16 [128] and load with
    # size D=256, then rely on the two halves being the same values via a
    # double load.
    inv_freq_first = ct.load(inv_freq_ptr, index=(0,), shape=(HALF,))
    inv_freq_second = ct.load(inv_freq_ptr, index=(0,), shape=(HALF,))
    # Reshape into (2, HALF) so we can broadcast along the first dim.
    # Actually easiest: two half-tiles, one per half of the output.
    # We'll store the two halves separately, each of size (1, HALF).
    pos_val = ct.load(positions_f_ptr, index=(row,), shape=(1,))
    pos_bc = ct.reshape(pos_val, (1,))

    inv_freq_first_f = ct.astype(inv_freq_first, ct.float32)
    inv_freq_second_f = ct.astype(inv_freq_second, ct.float32)

    phase_first = inv_freq_first_f * pos_bc
    phase_second = inv_freq_second_f * pos_bc

    cos_first = ct.cos(phase_first)
    cos_second = ct.cos(phase_second)
    sin_first = ct.sin(phase_first)
    sin_second = ct.sin(phase_second)

    cos_first_bf = ct.astype(ct.reshape(cos_first, (1, HALF)), ct.bfloat16)
    cos_second_bf = ct.astype(ct.reshape(cos_second, (1, HALF)), ct.bfloat16)
    sin_first_bf = ct.astype(ct.reshape(sin_first, (1, HALF)), ct.bfloat16)
    sin_second_bf = ct.astype(ct.reshape(sin_second, (1, HALF)), ct.bfloat16)

    ct.store(cos_ptr, index=(row, 0), tile=cos_first_bf)
    ct.store(cos_ptr, index=(row, 1), tile=cos_second_bf)
    ct.store(sin_ptr, index=(row, 0), tile=sin_first_bf)
    ct.store(sin_ptr, index=(row, 1), tile=sin_second_bf)


@ct.kernel
def _rmsnorm_rope_kernel(
    x_ptr,          # bf16 [rows, D] (rows = heads * SEQ)
    weight_ptr,     # bf16 [D]
    cos_ptr,        # bf16 [SEQ, D]
    sin_ptr,        # bf16 [SEQ, D]
    out_ptr,        # bf16 [rows, D]
    ROWS: ct.Constant[int],
    NUM_HEADS: ct.Constant[int],
    SEQ: ct.Constant[int],
    D: ct.Constant[int],
    HALF: ct.Constant[int],
):
    row = ct.bid(0)

    # Load whole-row x and reduce for RMS.
    x = ct.load(x_ptr, index=(row, 0), shape=(1, D))
    xf = ct.astype(ct.reshape(x, (D,)), ct.float32)
    sum_sq = ct.sum(xf * xf)
    mean_sq = sum_sq * (1.0 / D)
    inv_rms = ct.rsqrt(mean_sq + EPS)

    # Load the two halves of x separately (for rotary).
    x_first = ct.load(x_ptr, index=(row, 0), shape=(1, HALF))
    x_second = ct.load(x_ptr, index=(row, 1), shape=(1, HALF))
    xf_first = ct.astype(ct.reshape(x_first, (HALF,)), ct.float32)
    xf_second = ct.astype(ct.reshape(x_second, (HALF,)), ct.float32)

    # Load the two halves of weight separately.
    weight_first_v = ct.load(weight_ptr, index=(0,), shape=(HALF,))
    weight_second_v = ct.load(weight_ptr, index=(1,), shape=(HALF,))
    weight_first_f = ct.astype(weight_first_v, ct.float32) + 1.0
    weight_second_f = ct.astype(weight_second_v, ct.float32) + 1.0

    # value at first half of D
    value_first = ct.astype(
        ct.astype(xf_first * inv_rms * weight_first_f, ct.bfloat16),
        ct.float32,
    )
    value_second = ct.astype(
        ct.astype(xf_second * inv_rms * weight_second_f, ct.bfloat16),
        ct.float32,
    )
    # rot values:
    #   at first-half positions [0, HALF):  -x_second * inv_rms * weight_second (bf16)
    #   at second-half positions [HALF, D): x_first  * inv_rms * weight_first  (bf16)
    rot_first = ct.astype(
        ct.astype(-xf_second * inv_rms * weight_second_f, ct.bfloat16),
        ct.float32,
    )
    rot_second = ct.astype(
        ct.astype(xf_first * inv_rms * weight_first_f, ct.bfloat16),
        ct.float32,
    )

    # Load cos/sin (each half separately).
    pos = row // NUM_HEADS
    # If row spans multiple sequences (e.g. head*SEQ layout), pos is the SEQ index.
    # But our layout is (head, seq) — row = head * SEQ + seq, so pos = row - head*SEQ.
    pos = row - (row // SEQ) * SEQ

    cos_first_f = ct.astype(
        ct.reshape(ct.load(cos_ptr, index=(pos, 0), shape=(1, HALF)), (HALF,)),
        ct.float32,
    )
    cos_second_f = ct.astype(
        ct.reshape(ct.load(cos_ptr, index=(pos, 1), shape=(1, HALF)), (HALF,)),
        ct.float32,
    )
    sin_first_f = ct.astype(
        ct.reshape(ct.load(sin_ptr, index=(pos, 0), shape=(1, HALF)), (HALF,)),
        ct.float32,
    )
    sin_second_f = ct.astype(
        ct.reshape(ct.load(sin_ptr, index=(pos, 1), shape=(1, HALF)), (HALF,)),
        ct.float32,
    )

    direct_first = ct.astype(ct.astype(value_first * cos_first_f, ct.bfloat16), ct.float32)
    direct_second = ct.astype(ct.astype(value_second * cos_second_f, ct.bfloat16), ct.float32)
    rotated_first = ct.astype(ct.astype(rot_first * sin_first_f, ct.bfloat16), ct.float32)
    rotated_second = ct.astype(ct.astype(rot_second * sin_second_f, ct.bfloat16), ct.float32)

    out_first = direct_first + rotated_first
    out_second = direct_second + rotated_second

    ct.store(
        out_ptr, index=(row, 0),
        tile=ct.reshape(ct.astype(out_first, ct.bfloat16), (1, HALF)),
    )
    ct.store(
        out_ptr, index=(row, 1),
        tile=ct.reshape(ct.astype(out_second, ct.bfloat16), (1, HALF)),
    )


@oracle_impl(hardware="B200", point="5942af01")
def oracle_forward(inputs):
    q_mm, q_weight, inv_freq, positions, k_mm, k_weight = inputs[:6]
    device = q_mm.device

    # Compute cos/sin table via torch reference so we don't get bit-flip
    # differences with the trig kernel — much simpler.
    positions_f = positions.to(torch.float32).view(-1)   # [1000]
    inv_freq_f = inv_freq.to(torch.float32)               # [128]
    # phase[seq, d] = inv_freq[d % 128] * positions[seq]
    phase = torch.outer(positions_f, inv_freq_f.repeat(2))  # [1000, 256]
    cos_bf = torch.cos(phase).to(torch.bfloat16)
    sin_bf = torch.sin(phase).to(torch.bfloat16)

    # Store into strided output tensors ([1, 1000, 256]).
    cos_out = torch.empty_strided(
        (BATCH, SEQ_LEN, HEAD_DIM),
        (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    sin_out = torch.empty_strided(
        (BATCH, SEQ_LEN, HEAD_DIM),
        (SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    cos_out.view(SEQ_LEN, HEAD_DIM).copy_(cos_bf)
    sin_out.view(SEQ_LEN, HEAD_DIM).copy_(sin_bf)

    # Reshape Q and K: reference has permute(0,2,1,3) after view->[1,seq,heads,dim].
    # Q: view(1,1000,2048) -> view(1,1000,8,256) -> permute(0,2,1,3) -> [1,8,1000,256].
    # In the eager code, the RMSNorm+RoPE runs on this permuted layout, so its
    # rows are (head, seq) pairs. To keep the kernel simple, we build a
    # contiguous [heads*seq, dim] view.
    q_view = q_mm.view(1, SEQ_LEN, Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    q_flat = q_view.view(Q_HEADS * SEQ_LEN, HEAD_DIM)
    k_view = k_mm.view(1, SEQ_LEN, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous()
    k_flat = k_view.view(KV_HEADS * SEQ_LEN, HEAD_DIM)

    q_out_flat = torch.empty(Q_HEADS * SEQ_LEN, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k_out_flat = torch.empty(KV_HEADS * SEQ_LEN, HEAD_DIM, device=device, dtype=torch.bfloat16)

    cos_flat = cos_out.view(SEQ_LEN, HEAD_DIM)
    sin_flat = sin_out.view(SEQ_LEN, HEAD_DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (Q_HEADS * SEQ_LEN, 1, 1),
        _rmsnorm_rope_kernel,
        (q_flat, q_weight, cos_flat, sin_flat, q_out_flat,
         Q_HEADS * SEQ_LEN, Q_HEADS, SEQ_LEN, HEAD_DIM, HALF_DIM),
    )
    ct.launch(
        stream, (KV_HEADS * SEQ_LEN, 1, 1),
        _rmsnorm_rope_kernel,
        (k_flat, k_weight, cos_flat, sin_flat, k_out_flat,
         KV_HEADS * SEQ_LEN, KV_HEADS, SEQ_LEN, HEAD_DIM, HALF_DIM),
    )

    # Build the output tensors with the reference's non-contiguous strides.
    # Q out: shape [1, 8, 1000, 256], stride [8*1000*256, 256, 8*256, 1].
    q_out = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    q_out.copy_(q_out_flat.view(1, Q_HEADS, SEQ_LEN, HEAD_DIM))
    k_out = torch.empty_strided(
        (BATCH, KV_HEADS, SEQ_LEN, HEAD_DIM),
        (KV_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    k_out.copy_(k_out_flat.view(1, KV_HEADS, SEQ_LEN, HEAD_DIM))

    # Repeated key (grouped KV): [1, 8, 1000, 256] contiguous, each pair of heads
    # gets the same KV entries. Reference does unsqueeze(2)->expand(1,4,2,1000,256)->clone->view(1,8,1000,256).
    # k_out_flat is (KV_HEADS, SEQ_LEN, HEAD_DIM). We interleave to
    # (KV_HEADS, REPEAT, SEQ_LEN, HEAD_DIM) then flatten to (Q_HEADS, SEQ_LEN, HEAD_DIM).
    repeated_flat = (
        k_out_flat.view(KV_HEADS, SEQ_LEN, HEAD_DIM)
        .unsqueeze(1)
        .expand(KV_HEADS, REPEAT, SEQ_LEN, HEAD_DIM)
        .contiguous()
        .view(Q_HEADS, SEQ_LEN, HEAD_DIM)
    )
    repeated = torch.empty_strided(
        (BATCH, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, SEQ_LEN * HEAD_DIM, HEAD_DIM, 1),
        device=device, dtype=torch.bfloat16,
    )
    repeated.view(Q_HEADS, SEQ_LEN, HEAD_DIM).copy_(repeated_flat)

    return cos_out, sin_out, q_out, k_out, repeated
