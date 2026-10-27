"""cuTile port of pointwise_0c33f088df8a: GPT-OSS rotary Q/K.

Ports the Triton `_gptoss_rotary_repeat_kernel`. For each (seq, head, half):
- Load x_lo, x_hi
- Multiply by cos, sin (each rounded to bf16)
- q_out_lo = x_lo*cos - x_hi*sin; q_out_hi = x_hi*cos + x_lo*sin
Then produce the repeat_out tensor via torch expand+permute.
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


@ct.kernel
def _rotary_kernel(
    x_ptr,          # bf16 [SEQ*HEADS*2, HALF]
    cos_ptr,        # bf16 [SEQ, HALF]
    sin_ptr,        # bf16 [SEQ, HALF]
    out_ptr,        # bf16 [HEADS*SEQ*2, HALF]
    HEADS_C: ct.Constant[int],
    SEQ_C: ct.Constant[int],
    HEAD_DIM_C: ct.Constant[int],
    HALF_C: ct.Constant[int],
):
    row = ct.bid(0)  # in [0, HEADS*SEQ)
    head = row // SEQ_C
    seq = row - head * SEQ_C

    x_row = seq * HEADS_C + head
    x_lo = ct.load(x_ptr, index=(2 * x_row, 0), shape=(1, HALF_C))
    x_hi = ct.load(x_ptr, index=(2 * x_row + 1, 0), shape=(1, HALF_C))

    x_lo_f = ct.astype(x_lo, ct.float32)
    x_hi_f = ct.astype(x_hi, ct.float32)

    cos_tile = ct.load(cos_ptr, index=(seq, 0), shape=(1, HALF_C))
    sin_tile = ct.load(sin_ptr, index=(seq, 0), shape=(1, HALF_C))
    cos_f = ct.astype(cos_tile, ct.float32)
    sin_f = ct.astype(sin_tile, ct.float32)

    q_mul_lo = ct.astype(x_lo_f * cos_f, ct.bfloat16)
    q_rot_lo = ct.astype(x_hi_f * sin_f, ct.bfloat16)
    q_mul_hi = ct.astype(x_hi_f * cos_f, ct.bfloat16)
    q_rot_hi = ct.astype(x_lo_f * sin_f, ct.bfloat16)

    q_out_lo = ct.astype(q_mul_lo, ct.float32) - ct.astype(q_rot_lo, ct.float32)
    q_out_hi = ct.astype(q_mul_hi, ct.float32) + ct.astype(q_rot_hi, ct.float32)

    out_lo = ct.astype(q_out_lo, ct.bfloat16)
    out_hi = ct.astype(q_out_hi, ct.bfloat16)

    out_row = head * SEQ_C + seq
    ct.store(out_ptr, index=(2 * out_row, 0), tile=out_lo)
    ct.store(out_ptr, index=(2 * out_row + 1, 0), tile=out_hi)


@oracle_impl(hardware="B200", point="fa3cc74f", BLOCK_PAIRS=1024)
def oracle_forward(inputs, *, BLOCK_PAIRS):
    q_mm, k_mm, cos, sin, *_shape_params = inputs
    device = q_mm.device

    out_q = torch.empty((Q_HEADS, SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_k = torch.empty((1, KV_HEADS, SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_repeat = torch.empty_strided(
        (Q_HEADS, HEAD_DIM, SEQ),
        (SEQ * HEAD_DIM, 1, HEAD_DIM),
        device=device,
        dtype=torch.bfloat16,
    )

    q_in_view = q_mm.contiguous().view(SEQ * Q_HEADS * 2, HALF)
    k_in_view = k_mm.contiguous().view(SEQ * KV_HEADS * 2, HALF)
    cos_view = cos.squeeze(0).contiguous().view(SEQ, HALF)
    sin_view = sin.squeeze(0).contiguous().view(SEQ, HALF)

    out_q_view = out_q.view(Q_HEADS * SEQ * 2, HALF)
    out_k_view = out_k.view(1 * KV_HEADS * SEQ * 2, HALF)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (Q_HEADS * SEQ, 1, 1),
        _rotary_kernel,
        (q_in_view, cos_view, sin_view, out_q_view,
         Q_HEADS, SEQ, HEAD_DIM, HALF),
    )
    ct.launch(
        stream,
        (KV_HEADS * SEQ, 1, 1),
        _rotary_kernel,
        (k_in_view, cos_view, sin_view, out_k_view,
         KV_HEADS, SEQ, HEAD_DIM, HALF),
    )

    # Build out_repeat by expanding out_k over the GROUPS axis and copying
    # into the strided layout.
    expanded = out_k.squeeze(0).unsqueeze(1).expand(KV_HEADS, GROUPS, SEQ, HEAD_DIM)
    contig = expanded.contiguous().view(Q_HEADS, SEQ, HEAD_DIM)
    out_repeat.copy_(contig.permute(0, 2, 1))

    return out_q, out_k, out_repeat
