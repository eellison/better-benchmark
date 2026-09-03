"""cuTile port of pointwise_416792e04a87: GPT-OSS rotary pointwise.

Uses two cuTile kernels: one to build the (cos, sin) trig table (bf16 with
rtne rounding — cuTile's default), and one to apply rotary to Q and K,
also emitting the repeated-K clone in its target layout plus the tail
slice view.
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
SCALE = 1.3465735902799727
TRIG_NUMEL = SEQ * HALF  # 32000 = 500 * 64


@ct.kernel
def _rope_trig_table_kernel(
    inv_freq_ptr,      # bf16 [HALF]
    out_cos_ptr,       # bf16 [TRIG_NUMEL] laid out as (HALF_D, S)
    out_sin_ptr,       # bf16 [TRIG_NUMEL]
    S_C: ct.Constant[int],
    HALF_C: ct.Constant[int],
    BLOCK_S: ct.Constant[int],
):
    half_id = ct.bid(0)
    inv_freq_val = ct.load(inv_freq_ptr, index=(half_id,), shape=(1,))
    inv_freq_f = ct.astype(inv_freq_val, ct.float32)

    seq_ids = ct.arange(BLOCK_S, dtype=ct.int32)
    phase = ct.astype(seq_ids, ct.float32) * ct.reshape(inv_freq_f, (1,))
    cos_val = ct.astype(ct.cos(phase) * SCALE, ct.bfloat16)
    sin_val = ct.astype(ct.sin(phase) * SCALE, ct.bfloat16)

    ct.store(out_cos_ptr, index=(half_id, 0), tile=ct.reshape(cos_val, (1, BLOCK_S)))
    ct.store(out_sin_ptr, index=(half_id, 0), tile=ct.reshape(sin_val, (1, BLOCK_S)))


@ct.kernel
def _rope_qk_kernel(
    q_ptr,              # bf16 [Q_ROWS, HEAD_DIM] where Q_ROWS = Q_HEADS * SEQ
    k_ptr,              # bf16 [K_ROWS, HEAD_DIM] where K_ROWS = KV_HEADS * SEQ
    cos_table_ptr,      # bf16 [SEQ, HALF]
    sin_table_ptr,      # bf16 [SEQ, HALF]
    out_q_ptr,          # bf16 [Q_ROWS, HEAD_DIM]
    out_k_ptr,          # bf16 [K_ROWS, HEAD_DIM]
    BLOCK_HALF: ct.Constant[int],
):
    """One thread per (head_id, seq_id). Rotary rotates in halves."""
    row = ct.bid(0)  # row = head_id * SEQ + seq_id (for Q)
    is_q = ct.bid(1) == 0  # placeholder — not used; two launches

    # Not used, since we launch this kernel for Q and K separately with same shape.
    q_lo = ct.load(q_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    q_hi = ct.load(q_ptr, index=(row, 1), shape=(1, BLOCK_HALF))

    # seq_id = row % SEQ
    # We can compute from row & modulo, but simpler: table indexed by row modulo SEQ.
    # Since the caller feeds contiguous rows, we index by row % SEQ.
    # But we cannot do arbitrary indexing here — use precomputed per-row tables.
    ct.store(out_q_ptr, index=(row, 0), tile=q_lo)  # placeholder
    ct.store(out_q_ptr, index=(row, 1), tile=q_hi)


@ct.kernel
def _rotary_apply_kernel(
    x_ptr,              # bf16 [n_rows, HEAD_DIM]
    cos_ptr,            # bf16 [n_rows, HALF]
    sin_ptr,            # bf16 [n_rows, HALF]
    out_ptr,            # bf16 [n_rows, HEAD_DIM]
    BLOCK_HALF: ct.Constant[int],
):
    row = ct.bid(0)
    x_lo = ct.load(x_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    x_hi = ct.load(x_ptr, index=(row, 1), shape=(1, BLOCK_HALF))
    cos_val = ct.load(cos_ptr, index=(row, 0), shape=(1, BLOCK_HALF))
    sin_val = ct.load(sin_ptr, index=(row, 0), shape=(1, BLOCK_HALF))

    x_lo_f = ct.astype(x_lo, ct.float32)
    x_hi_f = ct.astype(x_hi, ct.float32)
    cos_f = ct.astype(cos_val, ct.float32)
    sin_f = ct.astype(sin_val, ct.float32)

    # Compute mul chunks separately with intermediate bf16 rounding to mimic
    # Inductor's per-mul bf16 boundary.
    lo_cos_bf = ct.astype(x_lo_f * cos_f, ct.bfloat16)
    hi_sin_bf = ct.astype(x_hi_f * sin_f, ct.bfloat16)
    hi_cos_bf = ct.astype(x_hi_f * cos_f, ct.bfloat16)
    lo_sin_bf = ct.astype(x_lo_f * sin_f, ct.bfloat16)

    out_lo = ct.astype(
        ct.astype(lo_cos_bf, ct.float32) - ct.astype(hi_sin_bf, ct.float32),
        ct.bfloat16)
    out_hi = ct.astype(
        ct.astype(hi_cos_bf, ct.float32) + ct.astype(lo_sin_bf, ct.float32),
        ct.bfloat16)

    ct.store(out_ptr, index=(row, 0), tile=out_lo)
    ct.store(out_ptr, index=(row, 1), tile=out_hi)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="deb9da58", BLOCK_HALF=32, BLOCK_S=1024)
def oracle_forward(inputs, *, BLOCK_HALF: int, BLOCK_S: int):
    q_mm, k_mm, inv_freq, *_shape_params = inputs
    device = q_mm.device

    # 1. Build the trig table via cuTile. Layout matches reference:
    # `unsqueeze/permute` gives contiguous stride (1000*32, 1, 1000) meaning
    # sin/cos stored as (HALF, SEQ) in memory but exposed as (1, SEQ, HALF).
    # Backing tensor is (HALF, S_PAD), padded to power of 2 in the S dim.
    S_PAD = BLOCK_S  # 1024, must be >= SEQ (1000), padded with zeros
    cos_pad = torch.zeros((HALF, S_PAD), device=device, dtype=torch.bfloat16)
    sin_pad = torch.zeros((HALF, S_PAD), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (HALF, 1, 1), _rope_trig_table_kernel,
        (inv_freq, cos_pad, sin_pad, S_PAD, HALF, BLOCK_S),
    )
    # Clip padding to real SEQ.
    cos_table = cos_pad[:, :SEQ]  # shape (HALF, SEQ)
    sin_table = sin_pad[:, :SEQ]

    # Return-shaped tables: (1, SEQ, HALF) with contiguous stride (S*HALF, 1, S).
    # That means the memory layout is (HALF, SEQ) contiguous but viewed as
    # (1, SEQ, HALF) via a permute (0, 2, 1) -> stride (S*HALF, 1, S).
    out_cos = cos_table.contiguous().unsqueeze(0).permute(0, 2, 1)  # (1, SEQ, HALF)
    out_sin = sin_table.contiguous().unsqueeze(0).permute(0, 2, 1)

    # 2. Apply rotary to Q and K.
    # Q: bf16[1, 1000, 4096] -> [1, 1000, 64, 64] -> permute(0, 2, 1, 3) -> [1, 64, 1000, 64]
    # split along -1 -> two [1, 64, 1000, 32] halves.
    # For simplicity, permute Q and K to (heads, seq, HEAD_DIM) contiguous.
    q_hs = q_mm.view(1, SEQ, Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().view(Q_HEADS * SEQ, HEAD_DIM)
    k_hs = k_mm.view(1, SEQ, KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3).contiguous().view(KV_HEADS * SEQ, HEAD_DIM)

    # cos/sin per row: for row = head * SEQ + seq_id, use cos_table[:, seq_id]
    # cos_table is (HALF, SEQ). We need (n_rows, HALF).
    seq_idx = torch.arange(SEQ, device=device)
    # For all heads (Q_HEADS or KV_HEADS), take the same trig table per seq.
    cos_per_row_q = cos_table[:, seq_idx].t().unsqueeze(0).expand(Q_HEADS, SEQ, HALF).contiguous().view(Q_HEADS * SEQ, HALF)
    sin_per_row_q = sin_table[:, seq_idx].t().unsqueeze(0).expand(Q_HEADS, SEQ, HALF).contiguous().view(Q_HEADS * SEQ, HALF)
    cos_per_row_k = cos_table[:, seq_idx].t().unsqueeze(0).expand(KV_HEADS, SEQ, HALF).contiguous().view(KV_HEADS * SEQ, HALF)
    sin_per_row_k = sin_table[:, seq_idx].t().unsqueeze(0).expand(KV_HEADS, SEQ, HALF).contiguous().view(KV_HEADS * SEQ, HALF)

    out_q_flat = torch.empty((Q_HEADS * SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)
    out_k_flat = torch.empty((KV_HEADS * SEQ, HEAD_DIM), device=device, dtype=torch.bfloat16)

    ct.launch(
        stream, (Q_HEADS * SEQ, 1, 1), _rotary_apply_kernel,
        (q_hs, cos_per_row_q, sin_per_row_q, out_q_flat, BLOCK_HALF),
    )
    ct.launch(
        stream, (KV_HEADS * SEQ, 1, 1), _rotary_apply_kernel,
        (k_hs, cos_per_row_k, sin_per_row_k, out_k_flat, BLOCK_HALF),
    )

    out_q_kd = out_q_flat.view(Q_HEADS, SEQ, HEAD_DIM)
    out_k_kd = out_k_flat.view(KV_HEADS, SEQ, HEAD_DIM)

    # view_4: bf16[64, 1000, 64] — same as out_q_kd (contiguous [Q, S, D])
    view_4 = out_q_kd

    # view_6: bf16[64, 64, 1000] = repeat-expanded K permuted to (H*G, D, S).
    # Reference: K [1,8,1000,64] -> unsqueeze -> expand [1,8,8,1000,64] clone
    # -> view [1,64,1000,64] -> permute (0,1,3,2) -> [1,64,64,1000] -> view [64,64,1000]
    k_repeat = (
        out_k_kd
        .unsqueeze(1)
        .expand(KV_HEADS, GROUPS, SEQ, HEAD_DIM)
        .contiguous()
        .view(KV_HEADS * GROUPS, SEQ, HEAD_DIM)
        .permute(0, 2, 1)
        .contiguous()
    )
    view_6 = k_repeat  # (64, 64, 1000)

    # slice_1: bf16[1, 8, 127, 64] — last 127 rows of K along seq.
    slice_1 = out_k_kd[:, TAIL_START:, :].unsqueeze(0)  # (1, 8, 127, 64)

    return out_cos, out_sin, view_4, view_6, slice_1
