"""cuTile port of mean_mean_7291b09200a4: Gemma query/key RMSNorm + RoPE.

The Triton oracle uses tl.inline_asm_elementwise for exact PTX rounding
(add.rn / mul.rn / cvt.rn.bf16), but cuTile's default f32 arithmetic is
already round-to-nearest-even and default astype to bfloat16 rounds RTNE,
so this ports directly with plain cuTile ops.

The key trick: rather than tile-slice within the kernel (unsupported),
we load the lo/hi halves separately from a view that presents the data
as [ROWS*2, HALF_DIM]. That gives us q_lo, q_hi as separate tiles.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


HEAD_DIM = 256
HALF_DIM = 128
SEQ_LEN = 1000
Q_HEADS = 8
KV_HEADS = 4
REPEAT = 2
EPS = 1.0e-6


@ct.kernel
def _rmsnorm_rope_kernel(
    mm_lo_ptr,       # bf16 [ROWS*2, HALF_DIM] — even rows are lo, odd rows are hi
    weight_lo_ptr,   # bf16 [HALF_DIM]
    weight_hi_ptr,   # bf16 [HALF_DIM]
    cos_lo_ptr,      # bf16 [SEQ, HALF_DIM]
    cos_hi_ptr,      # bf16 [SEQ, HALF_DIM]
    sin_lo_ptr,      # bf16 [SEQ, HALF_DIM]
    sin_hi_ptr,      # bf16 [SEQ, HALF_DIM]
    out_lo_ptr,      # bf16 [ROWS*2, HALF_DIM] — same layout as mm_lo_ptr
    HEADS_C: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM_C: ct.Constant[int],
    EPS_C: ct.Constant[float],
):
    row = ct.bid(0)  # row in [0, ROWS)
    # Load lo/hi halves.
    q_lo = ct.load(mm_lo_ptr, index=(2 * row, 0), shape=(1, HALF_DIM_C))
    q_hi = ct.load(mm_lo_ptr, index=(2 * row + 1, 0), shape=(1, HALF_DIM_C))

    q_lo_in_f = ct.astype(q_lo, ct.float32)
    q_hi_in_f = ct.astype(q_hi, ct.float32)

    # RMSNorm over both halves.
    sum_sq = ct.sum(q_lo_in_f * q_lo_in_f, axis=1, keepdims=True) + \
             ct.sum(q_hi_in_f * q_hi_in_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HEAD_DIM_C) + EPS_C)

    w_lo = ct.load(weight_lo_ptr, index=(0,), shape=(HALF_DIM_C,))
    w_hi = ct.load(weight_hi_ptr, index=(0,), shape=(HALF_DIM_C,))
    w_lo_f = ct.astype(ct.reshape(w_lo, (1, HALF_DIM_C)), ct.float32) + 1.0
    w_hi_f = ct.astype(ct.reshape(w_hi, (1, HALF_DIM_C)), ct.float32) + 1.0

    q_lo_norm_bf = ct.astype(q_lo_in_f * inv_rms * w_lo_f, ct.bfloat16)
    q_hi_norm_bf = ct.astype(q_hi_in_f * inv_rms * w_hi_f, ct.bfloat16)

    q_lo_norm_f = ct.astype(q_lo_norm_bf, ct.float32)
    q_hi_norm_f = ct.astype(q_hi_norm_bf, ct.float32)

    seq = row // HEADS_C
    cos_lo = ct.load(cos_lo_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    cos_hi = ct.load(cos_hi_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    sin_lo = ct.load(sin_lo_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    sin_hi = ct.load(sin_hi_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    cos_lo_f = ct.astype(cos_lo, ct.float32)
    cos_hi_f = ct.astype(cos_hi, ct.float32)
    sin_lo_f = ct.astype(sin_lo, ct.float32)
    sin_hi_f = ct.astype(sin_hi, ct.float32)

    # out_lo = round(q_lo*cos_lo) + round(-q_hi*sin_lo)   (each rounded to bf16)
    # out_hi = round(q_hi*cos_hi) + round(q_lo*sin_hi)
    prod_lo = ct.astype(q_lo_norm_f * cos_lo_f, ct.bfloat16)
    rot_lo = ct.astype((-q_hi_norm_f) * sin_lo_f, ct.bfloat16)
    prod_hi = ct.astype(q_hi_norm_f * cos_hi_f, ct.bfloat16)
    rot_hi = ct.astype(q_lo_norm_f * sin_hi_f, ct.bfloat16)

    out_lo = ct.astype(ct.astype(prod_lo, ct.float32) + ct.astype(rot_lo, ct.float32), ct.bfloat16)
    out_hi = ct.astype(ct.astype(prod_hi, ct.float32) + ct.astype(rot_hi, ct.float32), ct.bfloat16)

    ct.store(out_lo_ptr, index=(2 * row, 0), tile=out_lo)
    ct.store(out_lo_ptr, index=(2 * row + 1, 0), tile=out_hi)


@ct.kernel
def _rmsnorm_rope_repeat_kernel(
    mm_lo_ptr,
    weight_lo_ptr,
    weight_hi_ptr,
    cos_lo_ptr,
    cos_hi_ptr,
    sin_lo_ptr,
    sin_hi_ptr,
    repeat_out_ptr,   # [QH*SEQ*2, HALF_DIM] layout indexed by (out_head*SEQ+seq)*2 + half
    slice_out_ptr,    # [SEQ*KVH*2, HALF_DIM] indexed by (seq*KVH+kv_head)*2 + half
    KVH_C: ct.Constant[int],
    QH_C: ct.Constant[int],
    SEQ_C: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    HALF_DIM_C: ct.Constant[int],
    REPEAT_C: ct.Constant[int],
    EPS_C: ct.Constant[float],
):
    row = ct.bid(0)  # in [0, SEQ*KVH). Layout of mm_lo_ptr: rows ordered (seq, kv_head).
    k_lo = ct.load(mm_lo_ptr, index=(2 * row, 0), shape=(1, HALF_DIM_C))
    k_hi = ct.load(mm_lo_ptr, index=(2 * row + 1, 0), shape=(1, HALF_DIM_C))

    k_lo_in_f = ct.astype(k_lo, ct.float32)
    k_hi_in_f = ct.astype(k_hi, ct.float32)

    sum_sq = ct.sum(k_lo_in_f * k_lo_in_f, axis=1, keepdims=True) + \
             ct.sum(k_hi_in_f * k_hi_in_f, axis=1, keepdims=True)
    inv_rms = ct.rsqrt(sum_sq * (1.0 / HEAD_DIM_C) + EPS_C)

    w_lo = ct.load(weight_lo_ptr, index=(0,), shape=(HALF_DIM_C,))
    w_hi = ct.load(weight_hi_ptr, index=(0,), shape=(HALF_DIM_C,))
    w_lo_f = ct.astype(ct.reshape(w_lo, (1, HALF_DIM_C)), ct.float32) + 1.0
    w_hi_f = ct.astype(ct.reshape(w_hi, (1, HALF_DIM_C)), ct.float32) + 1.0

    k_lo_norm_bf = ct.astype(k_lo_in_f * inv_rms * w_lo_f, ct.bfloat16)
    k_hi_norm_bf = ct.astype(k_hi_in_f * inv_rms * w_hi_f, ct.bfloat16)
    k_lo_norm_f = ct.astype(k_lo_norm_bf, ct.float32)
    k_hi_norm_f = ct.astype(k_hi_norm_bf, ct.float32)

    seq = row // KVH_C
    kv_head = row - seq * KVH_C
    cos_lo = ct.load(cos_lo_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    cos_hi = ct.load(cos_hi_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    sin_lo = ct.load(sin_lo_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    sin_hi = ct.load(sin_hi_ptr, index=(seq, 0), shape=(1, HALF_DIM_C))
    cos_lo_f = ct.astype(cos_lo, ct.float32)
    cos_hi_f = ct.astype(cos_hi, ct.float32)
    sin_lo_f = ct.astype(sin_lo, ct.float32)
    sin_hi_f = ct.astype(sin_hi, ct.float32)

    prod_lo = ct.astype(k_lo_norm_f * cos_lo_f, ct.bfloat16)
    rot_lo = ct.astype((-k_hi_norm_f) * sin_lo_f, ct.bfloat16)
    prod_hi = ct.astype(k_hi_norm_f * cos_hi_f, ct.bfloat16)
    rot_hi = ct.astype(k_lo_norm_f * sin_hi_f, ct.bfloat16)

    out_lo = ct.astype(ct.astype(prod_lo, ct.float32) + ct.astype(rot_lo, ct.float32), ct.bfloat16)
    out_hi = ct.astype(ct.astype(prod_hi, ct.float32) + ct.astype(rot_hi, ct.float32), ct.bfloat16)

    # slice_out layout target: element (b=0, kv_head, seq, d) is at
    # linear kv_head*D + seq*KV*D + d = (seq*KV + kv_head)*D + d.
    # So row in [SEQ*KVH*2, HALF] view = (seq*KV + kv_head)*2 + half.
    slice_row = (seq * KVH_C + kv_head)
    ct.store(slice_out_ptr, index=(2 * slice_row, 0), tile=out_lo)
    ct.store(slice_out_ptr, index=(2 * slice_row + 1, 0), tile=out_hi)

    # repeat: for each repeat_idx, store to (out_head, seq, :). k_repeat is
    # contiguous [1, QH, SEQ, HEAD_DIM]. Element (0, out_head, seq, d) is at
    # out_head*SEQ*D + seq*D + d, so row = out_head*SEQ + seq (half in [0,2)).
    for repeat_idx in range(REPEAT):
        out_head = kv_head * REPEAT_C + repeat_idx
        repeat_row = out_head * SEQ_C + seq
        ct.store(repeat_out_ptr, index=(2 * repeat_row, 0), tile=out_lo)
        ct.store(repeat_out_ptr, index=(2 * repeat_row + 1, 0), tile=out_hi)


@oracle_impl(hardware="B200", point="2c02d9cc", BLOCK_ROWS=4, BLOCK_D=HEAD_DIM, BLOCK_HALF=HALF_DIM)
def oracle_forward(inputs, *, BLOCK_ROWS, BLOCK_D, BLOCK_HALF):
    q_proj, q_weight, cos, sin, k_proj, k_weight = inputs[:6]

    device = q_proj.device

    # Query
    q_out = torch.empty_strided(
        (1, Q_HEADS, SEQ_LEN, HEAD_DIM),
        (Q_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, Q_HEADS * HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    # Verify: q_out element (0, head, seq, d) at head*D + seq*Q*D + d.
    # = (seq*Q + head)*D + d. So flatten as [SEQ*Q, HEAD_DIM] contiguous by
    # rows in (seq, head) order.
    q_out_flat = q_out.as_strided((SEQ_LEN * Q_HEADS * 2, HALF_DIM), (HALF_DIM, 1))

    # q_proj is bf16[1000, 2048] which is [SEQ, Q_HEADS*HEAD_DIM]. Reshape
    # to [SEQ*Q_HEADS*2, HALF_DIM]: element (seq, head, half, d_half) at
    # seq*Q*D + head*D + half*HALF + d_half. That's the natural
    # (seq, head, half, d_half) contiguous layout: view then flatten.
    q_in_flat = q_proj.view(SEQ_LEN * Q_HEADS * 2, HALF_DIM)

    # Split weight [256] into two halves [128] each: lo=weight[:128], hi=weight[128:]
    q_w = q_weight.view(2, HALF_DIM)
    q_w_lo = q_w[0]
    q_w_hi = q_w[1]

    # cos/sin: [1, 1000, 256] -> view as [SEQ, 2, HALF]. lo/hi are strided
    # views with row-stride 2*HALF — cuTile respects strides directly, so
    # no `.contiguous()` copy needed.
    cos_view_lohi = cos.view(SEQ_LEN, 2, HALF_DIM)
    cos_lo = cos_view_lohi[:, 0]
    cos_hi = cos_view_lohi[:, 1]
    sin_view_lohi = sin.view(SEQ_LEN, 2, HALF_DIM)
    sin_lo = sin_view_lohi[:, 0]
    sin_hi = sin_view_lohi[:, 1]

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (SEQ_LEN * Q_HEADS, 1, 1),
        _rmsnorm_rope_kernel,
        (q_in_flat, q_w_lo, q_w_hi, cos_lo, cos_hi, sin_lo, sin_hi, q_out_flat,
         Q_HEADS, HEAD_DIM, HALF_DIM, EPS),
    )

    # Keys
    k_repeat = torch.empty(
        (1, Q_HEADS, SEQ_LEN, HEAD_DIM), device=device, dtype=torch.bfloat16
    )
    k_slice = torch.empty_strided(
        (1, KV_HEADS, SEQ_LEN, HEAD_DIM),
        (KV_HEADS * SEQ_LEN * HEAD_DIM, HEAD_DIM, KV_HEADS * HEAD_DIM, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    k_repeat_flat = k_repeat.view(Q_HEADS * SEQ_LEN * 2, HALF_DIM)
    k_slice_flat = k_slice.as_strided((SEQ_LEN * KV_HEADS * 2, HALF_DIM), (HALF_DIM, 1))

    # k_proj bf16[1000, 1024] = [SEQ, KVH*HEAD_DIM] contiguous. View as
    # [SEQ*KVH*2, HALF_DIM] in (seq, kv_head, half, d) order.
    k_in_flat = k_proj.view(SEQ_LEN * KV_HEADS * 2, HALF_DIM)

    k_w = k_weight.view(2, HALF_DIM)
    k_w_lo = k_w[0]
    k_w_hi = k_w[1]

    ct.launch(
        stream,
        (SEQ_LEN * KV_HEADS, 1, 1),
        _rmsnorm_rope_repeat_kernel,
        (k_in_flat, k_w_lo, k_w_hi, cos_lo, cos_hi, sin_lo, sin_hi,
         k_repeat_flat, k_slice_flat,
         KV_HEADS, Q_HEADS, SEQ_LEN, HEAD_DIM, HALF_DIM, REPEAT, EPS),
    )

    return q_out, k_repeat, k_slice
