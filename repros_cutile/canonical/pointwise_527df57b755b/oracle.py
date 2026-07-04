"""cuTile port of pointwise_527df57b755b: Mistral rotary Q/K with grouped-key repeat.

The Triton oracle fuses rotary application on Q (32 heads) and K (8 heads),
plus a repeat-heads clone of the rotated K (8→32) with a different stride
layout. All of the "inline PTX" ops are RTNE add/mul/fma which are the
default rounding in cuTile — we can just use `+`, `*`, and ct.astype(...).

We split the work into three passes for cleanliness:
  1. rotary Q  — output layout permuted (b, qh, s, d) with stride (s*qh*d, d, qh*d, 1)
  2. rotary K  — output layout permuted (b, kh, s, d) with stride (s*kh*d, d, kh*d, 1)
  3. repeat K  — expand K to (b, qh, s, d) contiguous, along groups axis.

We use torch to prepare permutation/expand outside the kernel and run
cuTile pointwise kernels over the contiguous representations, then let
torch reformat if needed. Since we need specific output strides though,
we write directly to a preallocated tensor via cuTile.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _rotary_out_kernel(
    x_ptr,        # bf16 [N_PAIRS * 2] (flat, contiguous B*S*H*D)
    cos_ptr,      # bf16 [S*D]
    sin_ptr,      # bf16 [S*D]
    out_ptr,      # bf16 [N_PAIRS * 2] (flat, contiguous output = torch will permute)
    HALF: ct.Constant[int],
    D: ct.Constant[int],
    H: ct.Constant[int],
    S: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """Element-wise rotary in flat [B*S*H*D] layout (no permute here).
    We process BLOCK pairs at a time — a "pair" is one (lo, hi) element pair.
    Grid: cdiv(N_PAIRS, BLOCK).
    """
    pid = ct.bid(0)
    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)

    # Decode: idx is a pair index; pair position = (b, s, h, half_d)
    half_d = idx % HALF
    h_idx = (idx // HALF) % H
    s_idx = (idx // (HALF * H)) % S
    b_idx = idx // (HALF * H * S)

    # x is flat [B, S, H, D] contig
    x_stride_b = S * H * D
    x_stride_s = H * D
    x_stride_h = D
    x_lo = b_idx * x_stride_b + s_idx * x_stride_s + h_idx * x_stride_h + half_d
    x_hi = x_lo + HALF

    cos_lo_off = s_idx * D + half_d
    cos_hi_off = cos_lo_off + HALF

    x_lo_v = ct.astype(ct.gather(x_ptr, x_lo), ct.float32)
    x_hi_v = ct.astype(ct.gather(x_ptr, x_hi), ct.float32)
    cos_lo_v = ct.astype(ct.gather(cos_ptr, cos_lo_off), ct.float32)
    cos_hi_v = ct.astype(ct.gather(cos_ptr, cos_hi_off), ct.float32)
    sin_lo_v = ct.astype(ct.gather(sin_ptr, cos_lo_off), ct.float32)
    sin_hi_v = ct.astype(ct.gather(sin_ptr, cos_hi_off), ct.float32)

    # bf16-rounded intermediates for each multiplicand (matches PTX cvt.rn.bf16.f32)
    q_mul_lo = ct.astype(ct.astype(x_lo_v * cos_lo_v, ct.bfloat16), ct.float32)
    q_rot_lo = ct.astype(ct.astype((0.0 - x_hi_v) * sin_lo_v, ct.bfloat16), ct.float32)
    q_mul_hi = ct.astype(ct.astype(x_hi_v * cos_hi_v, ct.bfloat16), ct.float32)
    q_rot_hi = ct.astype(ct.astype(x_lo_v * sin_hi_v, ct.bfloat16), ct.float32)

    out_lo = ct.astype(q_mul_lo + q_rot_lo, ct.bfloat16)
    out_hi = ct.astype(q_mul_hi + q_rot_hi, ct.bfloat16)

    ct.scatter(out_ptr, x_lo, out_lo)
    ct.scatter(out_ptr, x_hi, out_hi)


@oracle_impl(hardware="B200", point="8d7cf075", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    q_mm, cos, sin, k_mm = inputs[:4]
    device = q_mm.device

    seq = int(cos.shape[1])
    head_dim = int(cos.shape[2])
    half = head_dim // 2
    batch = int(q_mm.shape[0]) // seq
    q_heads = int(q_mm.shape[1]) // head_dim
    kv_heads = int(k_mm.shape[1]) // head_dim
    groups = q_heads // kv_heads

    # Reshape source tensors to flat contiguous [B, S, H, D]
    q_flat = q_mm.contiguous().view(batch, seq, q_heads, head_dim)
    k_flat = k_mm.contiguous().view(batch, seq, kv_heads, head_dim)
    cos_flat = cos.contiguous().view(seq * head_dim)
    sin_flat = sin.contiguous().view(seq * head_dim)

    # Run rotary in flat contiguous layout, then torch permutes for output stride.
    q_rot = torch.empty_like(q_flat)
    k_rot = torch.empty_like(k_flat)

    stream = torch.cuda.current_stream()

    n_pairs_q = batch * seq * q_heads * half
    ct.launch(
        stream, (ct.cdiv(n_pairs_q, BLOCK), 1, 1),
        _rotary_out_kernel,
        (q_flat.view(-1), cos_flat, sin_flat, q_rot.view(-1),
         half, head_dim, q_heads, seq, BLOCK),
    )

    n_pairs_k = batch * seq * kv_heads * half
    ct.launch(
        stream, (ct.cdiv(n_pairs_k, BLOCK), 1, 1),
        _rotary_out_kernel,
        (k_flat.view(-1), cos_flat, sin_flat, k_rot.view(-1),
         half, head_dim, kv_heads, seq, BLOCK),
    )

    # Now assemble outputs with the required strides.
    # Q output: shape (batch, q_heads, seq, head_dim), stride (seq*q_heads*head_dim, head_dim, q_heads*head_dim, 1)
    #   That's the same as q_flat.permute(0, 2, 1, 3), i.e., (b, h, s, d) from (b, s, h, d).
    out_q = q_rot.permute(0, 2, 1, 3)  # contiguous only via clone; but stride matches request
    # Verify strides: q_flat is (b, s, h, d) contig with stride (s*h*d, h*d, d, 1).
    # After permute(0,2,1,3) → shape (b, h, s, d), stride (s*h*d, d, h*d, 1) — matches expected.
    # No clone needed (view of same storage).

    out_k = k_rot.permute(0, 2, 1, 3)  # (b, kh, s, d) with stride (s*kh*d, d, kh*d, 1)

    # Repeat-heads clone: shape (b, q_heads, s, d), contiguous.
    # k_rot shape (b, s, kh, d) — we want (b, kh, s, d) expanded to (b, kh, groups, s, d) → (b, q_heads, s, d).
    out_k_contig = k_rot.permute(0, 2, 1, 3).contiguous()  # (b, kh, s, d) contig
    out_repeat = out_k_contig.unsqueeze(2).expand(batch, kv_heads, groups, seq, head_dim).contiguous().view(batch, q_heads, seq, head_dim)

    return out_q, out_k, out_repeat
