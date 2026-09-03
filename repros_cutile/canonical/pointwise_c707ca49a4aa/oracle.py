"""cuTile port of pointwise_c707ca49a4aa: Gemma rotary embedding pointwise scope.

Applies rotary rotation to Q and K via a cuTile kernel over (batch*heads*seq,
HEAD_DIM) rows with per-row cos/sin tables broadcast. Returns two returned
rotated tensors plus the repeated-K clone.

Uses rotate_half semantics:
  out = cat([lo * cos + (-hi) * sin, hi * cos + lo * sin])
Values come rotated back through the cat.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 1
SEQ = 1000
Q_HEADS = 8
KV_HEADS = 4
HEAD_DIM = 256
HALF = 128
GROUPS = Q_HEADS // KV_HEADS


@ct.kernel
def _rotary_apply_kernel(
    x_ptr,          # bf16 [n_rows, HEAD_DIM]
    cos_ptr,        # bf16 [n_rows, HEAD_DIM] broadcast per seq
    sin_ptr,        # bf16 [n_rows, HEAD_DIM]
    out_ptr,        # bf16 [n_rows, HEAD_DIM]
    BLOCK_HALF: ct.Constant[int],
):
    row = ct.bid(0)

    x_lo = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    x_hi = ct.load(x_ptr, index=(row, 1), shape=(1, BLOCK_HALF))
    cos_lo = ct.load(cos_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    cos_hi = ct.load(cos_ptr, index=(row, 1), shape=(1, BLOCK_HALF))
    sin_lo = ct.load(sin_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    sin_hi = ct.load(sin_ptr, index=(row, 1), shape=(1, BLOCK_HALF))

    # Per-mul bf16 rounding boundary matches Inductor.
    mul_lo_bf = ct.astype(
        ct.astype(x_lo, ct.float32) * ct.astype(cos_lo, ct.float32),
        ct.bfloat16)
    mul_hi_bf = ct.astype(
        ct.astype(x_hi, ct.float32) * ct.astype(cos_hi, ct.float32),
        ct.bfloat16)

    # rotate_half: [-hi, lo] then multiply by sin.
    neg_hi_bf = ct.astype(0.0 - ct.astype(x_hi, ct.float32), ct.bfloat16)
    rot_lo_bf = ct.astype(
        ct.astype(neg_hi_bf, ct.float32) * ct.astype(sin_lo, ct.float32),
        ct.bfloat16)
    rot_hi_bf = ct.astype(
        ct.astype(x_lo, ct.float32) * ct.astype(sin_hi, ct.float32),
        ct.bfloat16)

    # add: bf16 + bf16
    out_lo = ct.astype(
        ct.astype(mul_lo_bf, ct.float32) + ct.astype(rot_lo_bf, ct.float32),
        ct.bfloat16)
    out_hi = ct.astype(
        ct.astype(mul_hi_bf, ct.float32) + ct.astype(rot_hi_bf, ct.float32),
        ct.bfloat16)

    ct.store(out_ptr, index=(row, 0), tile=out_lo)
    ct.store(out_ptr, index=(row, 1), tile=out_hi)


@oracle_impl(hardware="B200", point="9e133e29", BLOCK_HALF=128)
def oracle_forward(inputs, *, BLOCK_HALF: int):
    q_mm, cos_arg, sin_arg, k_mm, *_shape_params = inputs
    device = q_mm.device

    # Cos / Sin: (1, SEQ, 256). Row broadcast: same for all heads.
    # Broadcast to (heads, SEQ, HEAD_DIM) contiguous.
    # We'll materialize per-row broadcast tables for Q and K.
    cos_flat = cos_arg.contiguous().view(SEQ, HEAD_DIM)  # (SEQ, HEAD_DIM)
    sin_flat = sin_arg.contiguous().view(SEQ, HEAD_DIM)

    # Q: (1, 1000, 2048) -> (1, 1000, 8, 256) -> permute -> (1, 8, 1000, 256)
    q_hs = q_mm.view(1, SEQ, Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().view(Q_HEADS * SEQ, HEAD_DIM)
    k_hs = k_mm.view(1, SEQ, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().view(KV_HEADS * SEQ, HEAD_DIM)

    # Per-row cos/sin: for row = head * SEQ + seq, use cos_flat[seq].
    cos_per_row_q = cos_flat.unsqueeze(0).expand(Q_HEADS, SEQ, HEAD_DIM).contiguous().view(Q_HEADS * SEQ, HEAD_DIM)
    sin_per_row_q = sin_flat.unsqueeze(0).expand(Q_HEADS, SEQ, HEAD_DIM).contiguous().view(Q_HEADS * SEQ, HEAD_DIM)
    cos_per_row_k = cos_flat.unsqueeze(0).expand(KV_HEADS, SEQ, HEAD_DIM).contiguous().view(KV_HEADS * SEQ, HEAD_DIM)
    sin_per_row_k = sin_flat.unsqueeze(0).expand(KV_HEADS, SEQ, HEAD_DIM).contiguous().view(KV_HEADS * SEQ, HEAD_DIM)

    out_q_flat = torch.empty((Q_HEADS * SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_k_flat = torch.empty((KV_HEADS * SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (Q_HEADS * SEQ, 1, 1), _rotary_apply_kernel,
        (q_hs, cos_per_row_q, sin_per_row_q, out_q_flat, BLOCK_HALF),
    )
    ct.launch(
        stream, (KV_HEADS * SEQ, 1, 1), _rotary_apply_kernel,
        (k_hs, cos_per_row_k, sin_per_row_k, out_k_flat, BLOCK_HALF),
    )

    # add: bf16[1, 8, 1000, 256]
    add = out_q_flat.view(1, Q_HEADS, SEQ, HEAD_DIM)
    add_1 = out_k_flat.view(1, KV_HEADS, SEQ, HEAD_DIM)

    # view_4: bf16[1, 8, 1000, 256] = repeated K clone from add_1.
    # unsqueeze at dim 2 -> [1, 4, 1, 1000, 256], expand [1, 4, 2, 1000, 256] clone
    # -> view [1, 8, 1000, 256]
    view_4 = (
        add_1
        .unsqueeze(2)
        .expand(1, KV_HEADS, GROUPS, SEQ, HEAD_DIM)
        .contiguous()
        .view(1, Q_HEADS, SEQ, HEAD_DIM)
    )
    return add, add_1, view_4
