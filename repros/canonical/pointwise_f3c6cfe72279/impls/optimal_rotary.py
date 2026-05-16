"""
Optimal Triton kernel for Mistral rotary position embedding.

Fuses the full pattern:
  q_embed = q * cos + rotate_half(q) * sin
  k_embed = k * cos + rotate_half(k) * sin
  k_repeated = repeat_kv(k_embed, n_rep=4)

Key insight: rotate_half(x)[..., i] for i < half_dim is -x[..., i + half_dim]
             rotate_half(x)[..., i] for i >= half_dim is  x[..., i - half_dim]

So: out[i] = x[i]*cos[i] + (-x[i + half_dim])*sin[i]     for i < half_dim
    out[i] = x[i]*cos[i] + ( x[i - half_dim])*sin[i]     for i >= half_dim

No cat, no intermediate tensors, single kernel per embed.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.utils import do_bench_using_profiling as do_bench


# ============================================================
# Triton kernel: fused rotary embedding
# ============================================================
@triton.jit
def _rotary_embed_kernel(
    # Input tensor x: [batch, heads, seq, head_dim] contiguous
    X_ptr,
    # cos/sin: [1, 1, seq, head_dim] broadcast over batch and heads
    COS_ptr,
    SIN_ptr,
    # Output tensor
    OUT_ptr,
    # Strides for X (and OUT, same layout)
    stride_xb, stride_xh, stride_xs, stride_xd,
    # Strides for cos/sin: [1, 1, seq, head_dim]
    stride_cs, stride_cd,
    # Dimensions
    batch: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Each program handles one (batch, head, seq_block) tile."""
    pid_bh = tl.program_id(0)  # flattened batch*heads
    pid_s = tl.program_id(1)   # seq block

    batch_idx = pid_bh // heads
    head_idx = pid_bh % heads

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    # Process full head_dim at once (BLOCK_D == head_dim)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    # Compute partner index for rotate_half
    # For i < half_dim: partner = i + half_dim (and negate)
    # For i >= half_dim: partner = i - half_dim (no negate)
    partner_offs = tl.where(d_offs < half_dim, d_offs + half_dim, d_offs - half_dim)
    negate = tl.where(d_offs < half_dim, -1.0, 1.0)

    # Load x values: shape [BLOCK_S, BLOCK_D]
    x_base = X_ptr + batch_idx * stride_xb + head_idx * stride_xh
    x_ptrs = x_base + s_offs[:, None] * stride_xs + d_offs[None, :] * stride_xd
    x_partner_ptrs = x_base + s_offs[:, None] * stride_xs + partner_offs[None, :] * stride_xd

    mask = s_mask[:, None] & d_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_partner = tl.load(x_partner_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin: [seq, head_dim] broadcast
    cs_ptrs = COS_ptr + s_offs[:, None] * stride_cs + d_offs[None, :] * stride_cd
    sn_ptrs = SIN_ptr + s_offs[:, None] * stride_cs + d_offs[None, :] * stride_cd
    cos_vals = tl.load(cs_ptrs, mask=mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sn_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute: out = x * cos + rotate_half(x) * sin
    rotated = x_partner * negate[None, :]
    out_vals = x_vals * cos_vals + rotated * sin_vals

    # Store
    out_base = OUT_ptr + batch_idx * stride_xb + head_idx * stride_xh
    out_ptrs = out_base + s_offs[:, None] * stride_xs + d_offs[None, :] * stride_xd
    tl.store(out_ptrs, out_vals.to(tl.bfloat16), mask=mask)


@triton.jit
def _rotary_embed_repeat_kv_kernel(
    # Input k: [batch, kv_heads, seq, head_dim]
    X_ptr,
    COS_ptr,
    SIN_ptr,
    # Output: [batch, kv_heads * n_rep, seq, head_dim]
    OUT_ptr,
    # Strides for X
    stride_xb, stride_xh, stride_xs, stride_xd,
    # Strides for OUT (may differ in head dim due to n_rep expansion)
    stride_ob, stride_oh, stride_os, stride_od,
    # Strides for cos/sin
    stride_cs, stride_cd,
    # Dims
    batch: tl.constexpr,
    kv_heads: tl.constexpr,
    n_rep: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused rotary embed + repeat_kv: each KV head is written to n_rep output heads."""
    pid_bh = tl.program_id(0)  # flattened batch*kv_heads
    pid_s = tl.program_id(1)

    batch_idx = pid_bh // kv_heads
    kv_head_idx = pid_bh % kv_heads

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < seq_len

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_dim

    partner_offs = tl.where(d_offs < half_dim, d_offs + half_dim, d_offs - half_dim)
    negate = tl.where(d_offs < half_dim, -1.0, 1.0)

    # Load x
    x_base = X_ptr + batch_idx * stride_xb + kv_head_idx * stride_xh
    x_ptrs = x_base + s_offs[:, None] * stride_xs + d_offs[None, :] * stride_xd
    x_partner_ptrs = x_base + s_offs[:, None] * stride_xs + partner_offs[None, :] * stride_xd

    mask = s_mask[:, None] & d_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    x_partner = tl.load(x_partner_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load cos/sin
    cs_ptrs = COS_ptr + s_offs[:, None] * stride_cs + d_offs[None, :] * stride_cd
    sn_ptrs = SIN_ptr + s_offs[:, None] * stride_cs + d_offs[None, :] * stride_cd
    cos_vals = tl.load(cs_ptrs, mask=mask, other=0.0).to(tl.float32)
    sin_vals = tl.load(sn_ptrs, mask=mask, other=0.0).to(tl.float32)

    rotated = x_partner * negate[None, :]
    out_vals = x_vals * cos_vals + rotated * sin_vals
    out_bf16 = out_vals.to(tl.bfloat16)

    # Write to all n_rep output heads
    out_base = OUT_ptr + batch_idx * stride_ob
    for rep in range(n_rep):
        out_head = kv_head_idx * n_rep + rep
        out_ptrs = out_base + out_head * stride_oh + s_offs[:, None] * stride_os + d_offs[None, :] * stride_od
        tl.store(out_ptrs, out_bf16, mask=mask)


# ============================================================
# Python wrapper
# ============================================================
def fused_rotary_embed(q, k, cos, sin, n_rep=4):
    """
    q: [batch, q_heads, seq, head_dim] bf16
    k: [batch, kv_heads, seq, head_dim] bf16
    cos: [1, 1, seq, head_dim] bf16 (or broadcastable)
    sin: [1, 1, seq, head_dim] bf16 (or broadcastable)
    n_rep: repeat factor for kv heads

    Returns:
      q_embed: [batch, q_heads, seq, head_dim] bf16
      k_repeated: [batch, q_heads, seq, head_dim] bf16
    """
    batch, q_heads, seq_len, head_dim = q.shape
    _, kv_heads, _, _ = k.shape
    half_dim = head_dim // 2

    # Ensure cos/sin are [seq, head_dim] contiguous for simple striding
    cos_2d = cos.reshape(seq_len, head_dim).contiguous()
    sin_2d = sin.reshape(seq_len, head_dim).contiguous()

    q_out = torch.empty_like(q)
    k_out = torch.empty(batch, q_heads, seq_len, head_dim, dtype=q.dtype, device=q.device)

    # Choose BLOCK_S to balance occupancy
    BLOCK_S = 32 if seq_len >= 32 else seq_len
    BLOCK_D = triton.next_power_of_2(head_dim)

    # Launch q kernel
    grid_q = (batch * q_heads, triton.cdiv(seq_len, BLOCK_S))
    _rotary_embed_kernel[grid_q](
        q, cos_2d, sin_2d, q_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos_2d.stride(0), cos_2d.stride(1),
        batch, q_heads, seq_len, head_dim, half_dim,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )

    # Launch k kernel (fused with repeat_kv)
    grid_k = (batch * kv_heads, triton.cdiv(seq_len, BLOCK_S))
    _rotary_embed_repeat_kv_kernel[grid_k](
        k, cos_2d, sin_2d, k_out,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        cos_2d.stride(0), cos_2d.stride(1),
        batch, kv_heads, n_rep, seq_len, head_dim, half_dim,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D,
    )

    return q_out, k_out


# ============================================================
# Reference implementation (eager PyTorch, matching the repro)
# ============================================================
def reference_rotary(q, k, inv_freq, n_rep=4):
    """Full reference matching the repro's computation."""
    batch, q_heads, seq_len, head_dim = q.shape
    kv_heads = k.shape[1]
    half_dim = head_dim // 2

    # Compute freqs
    position_ids = torch.arange(seq_len, device=q.device, dtype=torch.float32)
    freqs = torch.outer(position_ids, inv_freq)  # [seq, half_dim]
    emb = torch.cat([freqs, freqs], dim=-1)  # [seq, head_dim]
    cos = emb.cos().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)  # [1,1,seq,head_dim]
    sin = emb.sin().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        return torch.cat([-x2, x1], dim=-1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    # repeat_kv
    k_repeated = k_embed.unsqueeze(2).expand(batch, kv_heads, n_rep, seq_len, head_dim)
    k_repeated = k_repeated.reshape(batch, q_heads, seq_len, head_dim).contiguous()

    return q_embed, k_repeated, cos, sin


# ============================================================
# Inductor compiled version (what the repro does)
# ============================================================
def make_inductor_fn():
    """Create the inductor-compiled rotary embed matching the repro."""

    def rotary_fn(q, k, cos, sin, n_rep):
        batch, q_heads, seq_len, head_dim = q.shape
        kv_heads = k.shape[1]
        half_dim = head_dim // 2

        def rotate_half(x):
            x1 = x[..., :half_dim]
            x2 = x[..., half_dim:]
            return torch.cat([-x2, x1], dim=-1)

        q_embed = q * cos + rotate_half(q) * sin
        k_embed = k * cos + rotate_half(k) * sin

        # repeat_kv
        k_repeated = k_embed.unsqueeze(2).expand(batch, kv_heads, n_rep, seq_len, head_dim)
        k_repeated = k_repeated.reshape(batch, q_heads, seq_len, head_dim).contiguous()
        return q_embed, k_repeated

    return torch.compile(rotary_fn, mode="max-autotune")


# ============================================================
# Benchmark
# ============================================================
def main():
    torch.manual_seed(42)
    device = "cuda"

    # Shapes from repro: batch=4, q_heads=32, kv_heads=8, seq=512, head_dim=128
    batch, q_heads, kv_heads, seq_len, head_dim = 4, 32, 8, 512, 128
    n_rep = q_heads // kv_heads  # 4

    q = torch.randn(batch, q_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    inv_freq = torch.randn(64, dtype=torch.float32, device=device)

    # Compute cos/sin for both implementations
    position_ids = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(position_ids, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos = emb.cos().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)  # [1,1,seq,head_dim]
    sin = emb.sin().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

    # ---- Correctness check ----
    q_ref, k_ref, _, _ = reference_rotary(q, k, inv_freq, n_rep)
    q_tri, k_tri = fused_rotary_embed(q, k, cos, sin, n_rep)

    q_err = (q_ref.float() - q_tri.float()).abs().max().item()
    k_err = (k_ref.float() - k_tri.float()).abs().max().item()
    print(f"Correctness check:")
    print(f"  q_embed max error: {q_err:.6f}")
    print(f"  k_repeated max error: {k_err:.6f}")
    assert q_err < 0.1, f"q error too large: {q_err}"
    assert k_err < 0.1, f"k error too large: {k_err}"
    print("  PASSED\n")

    # ---- Benchmark Inductor (compiled) ----
    print("Compiling with torch.compile (max-autotune)...")
    compiled_fn = make_inductor_fn()
    # Warmup
    for _ in range(3):
        compiled_fn(q, k, cos, sin, n_rep)
    torch.cuda.synchronize()

    inductor_ms = do_bench(lambda: compiled_fn(q, k, cos, sin, n_rep))
    print(f"Inductor (torch.compile): {inductor_ms:.4f} ms")

    # ---- Benchmark Triton (ours) ----
    # Warmup
    for _ in range(3):
        fused_rotary_embed(q, k, cos, sin, n_rep)
    torch.cuda.synchronize()

    triton_ms = do_bench(lambda: fused_rotary_embed(q, k, cos, sin, n_rep))
    print(f"Triton (fused):           {triton_ms:.4f} ms")

    # ---- Report ----
    speedup = inductor_ms / triton_ms
    print(f"\nSpeedup: {speedup:.2f}x")
    print(f"  Inductor: {inductor_ms:.4f} ms")
    print(f"  Triton:   {triton_ms:.4f} ms")

    if speedup > 1.0:
        print(f"\n  >>> Triton kernel is {speedup:.2f}x FASTER than Inductor <<<")
    else:
        print(f"\n  >>> Inductor is {1/speedup:.2f}x faster (Triton kernel needs optimization) <<<")


if __name__ == "__main__":
    main()
