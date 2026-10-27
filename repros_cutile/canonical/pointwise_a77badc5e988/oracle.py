"""cuTile port of pointwise_a77badc5e988: GPT-J bf16 RoPE pair-scatter kernel.

Per-row kernel: one program per (seq, head). Loads a tile of HEAD_DIM=256
elements, applies the RoPE cos/sin/scatter/full-base pattern (rotary for
dim<64, tail passes through arg5_1 + q_in for dim>=64), and stores back
to (seq, head, dim) in a (SEQ, HEADS*HEAD_DIM) output buffer.

inline_asm add.rn.f32 / mul.rn.f32 and bf16 rtne rounding are cuTile
defaults so they drop to plain arithmetic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
HIDDEN = HEADS * HEAD_DIM
ROWS = SEQ * HEADS


@ct.kernel
def _rope_kernel(
    q_in,          # bf16 (HEADS, HEAD_DIM, SEQ)  — arg0_1
    k_in,          # bf16 (HEADS, SEQ, HEAD_DIM)  — arg1_1
    coeff_cos,     # bf16 (SEQ, ROTARY_DIM)       — arg2_1 flat view
    scatter_base,  # bf16 (SEQ, HEADS, ROTARY_DIM) — arg3_1 flat view
    coeff_sin,     # bf16 (SEQ, ROTARY_DIM)       — arg4_1 flat view (aliased with arg2)
    full_base,     # bf16 (SEQ, HEADS, HEAD_DIM)  — arg5_1 flat view
    q_out,         # bf16 (SEQ, HEADS, HEAD_DIM)
    k_out,         # bf16 (SEQ, HEADS, HEAD_DIM)
    BLOCK_D: ct.Constant[int],
):
    row = ct.bid(0)
    seq = row // HEADS
    head = row - seq * HEADS

    dims = ct.arange(BLOCK_D, dtype=ct.int32)
    pair = dims // 2
    paired_dim = dims ^ 1
    is_even = (dims & 1) == 0
    rotary = dims < ROTARY_DIM

    # Q load: q_in[head, dim, seq]
    seq_b = ct.full((BLOCK_D,), seq, dtype=ct.int32)
    head_b = ct.full((BLOCK_D,), head, dtype=ct.int32)
    q_bf = ct.gather(q_in, (head_b, dims, seq_b))
    q_pair_bf = ct.gather(q_in, (head_b, paired_dim, seq_b))
    q_f = ct.astype(q_bf, ct.float32)
    q_pair_f = ct.astype(q_pair_bf, ct.float32)
    # bf16 round q (matches _bf16 wrapper on q input) — Triton does q = _bf16(q_load_to_f32)
    q_f = ct.astype(ct.astype(q_f, ct.bfloat16), ct.float32)
    q_pair_f = ct.astype(ct.astype(q_pair_f, ct.bfloat16), ct.float32)

    # K load: k_in[head, seq, dim]
    k_bf = ct.gather(k_in, (head_b, seq_b, dims))
    k_pair_bf = ct.gather(k_in, (head_b, seq_b, paired_dim))
    k_f = ct.astype(k_bf, ct.float32)
    k_pair_f = ct.astype(k_pair_bf, ct.float32)
    # Note: Triton does NOT round k with _bf16 wrapper (k = tl.load(...).to(f32)).

    # cos/sin loads at (seq, pair) — same pair for even/odd of each pair.
    # pair ranges 0..127 for dim 0..255, but only pair<32 (rotary dims) are used.
    cos_bf = ct.gather(coeff_cos, (seq_b, pair), mask=rotary, padding_value=0)
    sin_bf = ct.gather(coeff_sin, (seq_b, pair), mask=rotary, padding_value=0)
    cos_f = ct.astype(cos_bf, ct.float32)
    sin_f = ct.astype(sin_bf, ct.float32)

    # scatter_base at (seq, head, dim) — Triton uses mask rotary so we
    # substitute 0 for tail dims via ct.where after load. But we can also
    # gather at the position freely with rotary mask.
    scatter_bf = ct.gather(scatter_base, (seq_b, head_b, dims),
                           mask=rotary, padding_value=0)
    scatter_f = ct.astype(scatter_bf, ct.float32)

    # full_base at (seq, head, dim)
    full_bf = ct.gather(full_base, (seq_b, head_b, dims))
    full_f = ct.astype(full_bf, ct.float32)

    # bf16-rounded arithmetic (default in cuTile).
    def bf16(x):
        return ct.astype(ct.astype(x, ct.bfloat16), ct.float32)

    q_pair_cos = bf16(q_pair_f * cos_f)
    q_scatter_even = bf16(scatter_f + q_pair_cos)
    q_scatter_odd = bf16(-q_pair_cos + scatter_f)
    q_scatter = ct.where(is_even, q_scatter_even, q_scatter_odd)
    q_direct = bf16(q_f * sin_f)
    q_rotary_val = bf16(full_f + bf16(q_scatter + q_direct))
    q_tail = bf16(q_f + full_f)
    q_out_val = ct.astype(ct.where(rotary, q_rotary_val, q_tail),
                          ct.bfloat16)
    ct.scatter(q_out, (seq_b, head_b, dims), q_out_val)

    k_pair_cos = bf16(k_pair_f * cos_f)
    k_scatter_even = bf16(scatter_f + k_pair_cos)
    k_scatter_odd = bf16(-k_pair_cos + scatter_f)
    k_scatter = ct.where(is_even, k_scatter_even, k_scatter_odd)
    k_direct = bf16(k_f * sin_f)
    k_rotary_val = bf16(full_f + bf16(k_scatter + k_direct))
    k_tail = bf16(k_f + full_f)
    k_out_val = ct.astype(ct.where(rotary, k_rotary_val, k_tail),
                          ct.bfloat16)
    ct.scatter(k_out, (seq_b, head_b, dims), k_out_val)


@oracle_impl(hardware="B200", point="0231a2f7", BLOCK_M=1, BLOCK_D=256)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_D: int):
    del BLOCK_M  # unused: one row per program.
    q_in, k_in, coeff_cos_in, scatter_base_in, coeff_sin_in, full_base_in = inputs[:6]

    # Coefficient tensors have shape (1,128,1,32,1) with stride (8192,64,32,1,1).
    # Their flat elements at (s, p) are seq*64+p in the underlying [128,64] storage.
    # arg2_1 has off=0 → view as (SEQ, ROTARY_DIM) via as_strided.
    # arg4_1 has off=32 in shared storage — we produce a proper (SEQ, ROTARY_DIM)
    # view respecting the storage_offset.
    #
    # The reference `arg2_1[0, s, 0, p, 0]` = the underlying storage at (s, p).
    # We build cos_view and sin_view accordingly:
    device = q_in.device
    # coeff_cos_in.stride() is (8192, 64, 32, 1, 1), storage_offset = 0.
    # coeff_sin_in.stride() is (8192, 64, 32, 1, 1), storage_offset = 32.
    # Each seq has ROTARY_DIM // 2 = 32 pair coefficients (pair = dim // 2).
    # An indexing at (0, s, 0, p, 0) uses s * 64 + p (relative to
    # storage_offset). We materialize each as a (SEQ, ROTARY_DIM // 2) view.
    PAIRS = ROTARY_DIM // 2
    cos_view = torch.as_strided(coeff_cos_in, (SEQ, PAIRS), (64, 1),
                                coeff_cos_in.storage_offset())
    sin_view = torch.as_strided(coeff_sin_in, (SEQ, PAIRS), (64, 1),
                                coeff_sin_in.storage_offset())

    # scatter_base is arg3_1 shape (1,128,16,64). View as (SEQ, HEADS, ROTARY_DIM).
    scatter_view = scatter_base_in.view(SEQ, HEADS, ROTARY_DIM)
    # full_base is arg5_1 shape (1,128,16,256). View as (SEQ, HEADS, HEAD_DIM).
    full_view = full_base_in.view(SEQ, HEADS, HEAD_DIM)

    q_out = torch.empty_strided(
        (SEQ, HIDDEN), (HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    k_out = torch.empty_strided(
        (SEQ, HIDDEN), (HIDDEN, 1),
        device=device, dtype=torch.bfloat16,
    )
    q_out_3d = q_out.view(SEQ, HEADS, HEAD_DIM)
    k_out_3d = k_out.view(SEQ, HEADS, HEAD_DIM)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _rope_kernel,
        (
            q_in, k_in,
            cos_view, scatter_view, sin_view, full_view,
            q_out_3d, k_out_3d,
            BLOCK_D,
        ),
    )
    return q_out, q_out.permute(1, 0), k_out, k_out.permute(1, 0)
