"""
Ideal hand-written Triton kernel: K RoPE + GQA expand fused.

Instead of:
  K0: compute K RoPE -> buf1[4,8,512,128]  (2M elements, 5 loads + 1 store)
  K1: copy buf1 -> buf2[4,8,4,512,128]     (8M elements, 1 load + 1 store)

We want ONE kernel that iterates over the producer's domain (2M elements)
and writes directly to the 4 output locations per element:
  K_fused: compute K RoPE, store to 4 locations in buf2  (2M elements, 5 loads + 4 stores)

Savings: eliminate buf1 entirely (4MB write + 4MB read = 8MB),
  at cost of 3 extra stores per element (3 * 2M * 2B = 12MB).
  Net: 8MB saved - 12MB extra stores = -4MB worse? No -- the expand copy
  kernel was doing 8M loads + 8M stores = 32MB. The fused kernel does
  2M * (5 loads + 4 stores) = 10M + 8M = 36MB... hmm.

Actually let's think about bytes:
  Baseline:
    K0: read k_proj(4MB) + cos(128KB) + sin(128KB), write buf1(4MB) = ~8.3MB
    K1: read buf1(4MB), write buf2(16MB) = 20MB   [8M elements * 2B each for r+w, but buf2 is 8M*2B=16MB]
    Wait -- buf1 is [4,8,512,128] = 2M elements * 2B = 4MB
    buf2 is [4,8,4,512,128] = 8M elements * 2B = 16MB
    K1 reads 4MB (buf1 with broadcast), writes 16MB = 20MB
    Total K0+K1: 8.3 + 20 = 28.3 MB

  Ideal fused (iterate at producer domain, multi-store):
    Read k_proj(4MB) + cos(128KB) + sin(128KB), write buf2(16MB)
    = 4MB + 0.25MB + 16MB = 20.25MB
    Savings: 8MB (no buf1 round-trip)

  Bad fused (iterate at consumer domain, recompute):
    Read k_proj(4MB * 4 = 16MB broadcast) + cos(128KB*4) + sin(128KB*4), write buf2(16MB)
    = 16MB + 1MB + 16MB = 33MB  -- WORSE than baseline!

So the multi-store approach is clearly better.

The ideal kernel iterates 2M elements (producer domain) and does 4 stores each.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import triton
import triton.language as tl


@triton.jit
def k_rope_gqa_fused(
    k_proj_ptr,   # [2048, 1024] bf16 = [B*S, H_kv*D]
    cos_ptr,      # [1, 512, 128] bf16
    sin_ptr,      # [1, 512, 128] bf16
    out_ptr,      # [4, 8, 4, 512, 128] bf16 = [B, H_kv, N_rep, S, D]
    B: tl.constexpr,
    S: tl.constexpr,
    H_kv: tl.constexpr,
    D: tl.constexpr,
    N_rep: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Iterate over producer domain: B * H_kv * S * D = 4*8*512*128 = 2M
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < B * H_kv * S * D

    # Decompose flat index into (batch, kv_head, seq, head_dim)
    head_dim = xindex % D                           # [0, 128)
    seq = (xindex // D) % S                          # [0, 512)
    kv_head = (xindex // (D * S)) % H_kv             # [0, 8)
    batch = xindex // (D * S * H_kv)                 # [0, 4)

    HALF_D: tl.constexpr = D // 2

    # Load k_proj: layout is [B*S, H_kv*D], stride = (H_kv*D, 1)
    # index = (batch * S + seq) * (H_kv * D) + kv_head * D + head_dim
    k_proj_idx = (batch * S + seq) * (H_kv * D) + kv_head * D + head_dim
    k_val = tl.load(k_proj_ptr + k_proj_idx, mask=xmask).to(tl.float32)

    # Load cos/sin: layout is [1, S, D], index = seq * D + head_dim
    cs_idx = seq * D + head_dim
    cos_val = tl.load(cos_ptr + cs_idx, mask=xmask).to(tl.float32)
    sin_val = tl.load(sin_ptr + cs_idx, mask=xmask).to(tl.float32)

    # Build rotated value: [-k[D/2:], k[:D/2]]
    is_first_half = head_dim < HALF_D
    # For first half: load k[head_dim + D/2] and negate
    # For second half: load k[head_dim - D/2]
    rot_idx = tl.where(is_first_half,
                       k_proj_idx + HALF_D,
                       k_proj_idx - HALF_D)
    k_rot_val = tl.load(k_proj_ptr + rot_idx, mask=xmask).to(tl.float32)
    k_rotated = tl.where(is_first_half, -k_rot_val, k_rot_val)

    # RoPE: k * cos + k_rotated * sin
    result = k_val * cos_val + k_rotated * sin_val
    result_bf16 = result.to(tl.bfloat16)

    # Store to 4 locations in output (GQA expand: repeat across N_rep dim)
    # out layout: [B, H_kv, N_rep, S, D]
    # strides: (H_kv*N_rep*S*D, N_rep*S*D, S*D, D, 1)
    base_out = batch * (H_kv * N_rep * S * D) + kv_head * (N_rep * S * D) + seq * D + head_dim
    stride_rep = S * D  # stride for the N_rep dimension

    # 4 stores — one per repeat
    tl.store(out_ptr + base_out + 0 * stride_rep, result_bf16, mask=xmask)
    tl.store(out_ptr + base_out + 1 * stride_rep, result_bf16, mask=xmask)
    tl.store(out_ptr + base_out + 2 * stride_rep, result_bf16, mask=xmask)
    tl.store(out_ptr + base_out + 3 * stride_rep, result_bf16, mask=xmask)


def main():
    torch.manual_seed(42)
    B, S, H_kv, D, N_rep = 4, 512, 8, 128, 4

    k_proj = torch.randn(B * S, H_kv * D, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda')
    sin = torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda')

    # Reference: eager K RoPE + GQA expand
    k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)  # [4,8,512,128]
    cos_q = cos.unsqueeze(1)  # [1,1,512,128]
    sin_q = sin.unsqueeze(1)
    k1, k2 = k[..., :D//2], k[..., D//2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    k_embed = k * cos_q + k_rotated * sin_q
    k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D)
    ref = k_expanded.clone().reshape(B, H_kv * N_rep, S, D)

    # Fused kernel
    out_5d = torch.empty(B, H_kv, N_rep, S, D, dtype=torch.bfloat16, device='cuda')
    numel = B * H_kv * S * D
    XBLOCK = 1024
    grid = ((numel + XBLOCK - 1) // XBLOCK,)
    k_rope_gqa_fused[grid](k_proj, cos, sin, out_5d, B, S, H_kv, D, N_rep, XBLOCK)
    out = out_5d.reshape(B, H_kv * N_rep, S, D)

    # Compare
    max_diff = (ref - out).abs().max().item()
    print(f"Max diff: {max_diff}")
    print(f"PASS" if max_diff < 0.1 else "FAIL")

    # Benchmark
    from triton.testing import do_bench

    fused_ms = do_bench(
        lambda: k_rope_gqa_fused[grid](k_proj, cos, sin, out_5d, B, S, H_kv, D, N_rep, XBLOCK)
    )

    def baseline():
        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
        k1, k2 = k[..., :D//2], k[..., D//2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = k * cos_q + k_rotated * sin_q
        return k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone()

    baseline_ms = do_bench(baseline)

    # Compiled baseline (inductor's 2 kernels for just the K part)
    class KModel(torch.nn.Module):
        def forward(self, k_proj, cos, sin):
            B, S, H_kv, D, N_rep = 4, 512, 8, 128, 4
            k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
            cos_q = cos.unsqueeze(1)
            sin_q = sin.unsqueeze(1)
            k1, k2 = k[..., :D//2], k[..., D//2:]
            k_rotated = torch.cat((-k2, k1), dim=-1)
            k_embed = k * cos_q + k_rotated * sin_q
            k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D)
            return k_expanded.clone().reshape(B, H_kv * N_rep, S, D)

    compiled_k = torch.compile(KModel().cuda())
    with torch.no_grad():
        compiled_k(k_proj, cos, sin)  # warmup
    compiled_ms = do_bench(lambda: compiled_k(k_proj, cos, sin))

    print(f"\nBaseline (eager K RoPE + expand): {baseline_ms:.3f} ms")
    print(f"Compiled (inductor, 2 kernels):   {compiled_ms:.3f} ms")
    print(f"Fused (1 kernel, multi-store):    {fused_ms:.3f} ms")
    print(f"Speedup vs compiled: {compiled_ms / fused_ms:.2f}x")


if __name__ == "__main__":
    main()
