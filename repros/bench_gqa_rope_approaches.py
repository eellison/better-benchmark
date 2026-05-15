"""Benchmark: baseline (3 kernels) vs combo (2 kernels) vs custom fused (1 kernel)"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
import torch
import time
import triton
import triton.language as tl

B, S, H_q, H_kv, D = 4, 512, 32, 8, 128
N_rep = H_q // H_kv  # 4

# ============ Custom fused kernel ============

@triton.jit
def fused_qk_rope_expand_kernel(
    q_ptr, k_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
    stride_q_bs, stride_k_bs,
    B: tl.constexpr, S: tl.constexpr, H_q: tl.constexpr, H_kv: tl.constexpr,
    D: tl.constexpr, HALF_D: tl.constexpr, N_rep: tl.constexpr,
):
    pid = tl.program_id(0)
    h_q = pid % H_q
    tmp = pid // H_q
    s = tmp % S
    b = tmp // S

    d_range = tl.arange(0, D)
    half_range = tl.arange(0, HALF_D)

    # Q RoPE
    q_offset = b * stride_q_bs + s * H_q * D + h_q * D
    q_lo = tl.load(q_ptr + q_offset + half_range).to(tl.float32)
    q_hi = tl.load(q_ptr + q_offset + HALF_D + half_range).to(tl.float32)

    cs_offset = s * D
    cos_lo = tl.load(cos_ptr + cs_offset + half_range).to(tl.float32)
    cos_hi = tl.load(cos_ptr + cs_offset + HALF_D + half_range).to(tl.float32)
    sin_lo = tl.load(sin_ptr + cs_offset + half_range).to(tl.float32)
    sin_hi = tl.load(sin_ptr + cs_offset + HALF_D + half_range).to(tl.float32)

    # q_embed = q * cos + cat(-q_hi, q_lo) * sin
    q_embed_lo = q_lo * cos_lo + (-q_hi) * sin_lo
    q_embed_hi = q_hi * cos_hi + q_lo * sin_hi

    q_out_offset = (b * H_q + h_q) * S * D + s * D
    tl.store(q_out_ptr + q_out_offset + half_range, q_embed_lo.to(tl.bfloat16))
    tl.store(q_out_ptr + q_out_offset + HALF_D + half_range, q_embed_hi.to(tl.bfloat16))

    # K RoPE (same cos/sin, different input)
    h_kv = h_q // N_rep
    k_offset = b * stride_k_bs + s * H_kv * D + h_kv * D
    k_lo = tl.load(k_ptr + k_offset + half_range).to(tl.float32)
    k_hi = tl.load(k_ptr + k_offset + HALF_D + half_range).to(tl.float32)

    k_embed_lo = k_lo * cos_lo + (-k_hi) * sin_lo
    k_embed_hi = k_hi * cos_hi + k_lo * sin_hi

    # Write K at expanded position (same as Q output layout)
    k_out_offset = (b * H_q + h_q) * S * D + s * D
    tl.store(k_out_ptr + k_out_offset + half_range, k_embed_lo.to(tl.bfloat16))
    tl.store(k_out_ptr + k_out_offset + HALF_D + half_range, k_embed_hi.to(tl.bfloat16))


# ============ PyTorch model ============

class Repro(torch.nn.Module):
    def forward(self, q_proj, cos, sin, k_proj):
        q = q_proj.view(B, S, H_q, D).permute(0, 2, 1, 3)
        cos_q = cos.unsqueeze(1)
        sin_q = sin.unsqueeze(1)
        q1, q2 = q[..., :D//2], q[..., D//2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = q * cos_q + q_rotated * sin_q

        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
        k1, k2 = k[..., :D//2], k[..., D//2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = k * cos_q + k_rotated * sin_q

        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone().reshape(B, H_q, S, D)
        return q_embed, k_expanded


def bench(fn, *args, warmup=20, rep=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1e6


if __name__ == "__main__":
    import torch._inductor.config as cfg
    import torch._inductor.metrics as m

    cfg.force_disable_caches = True

    q_proj = torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda')
    sin = torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda')
    k_proj = torch.randn(2048, 1024, dtype=torch.bfloat16, device='cuda')

    model = Repro().cuda()
    inputs = [q_proj, cos, sin, k_proj]

    # --- Baseline (3 kernels) ---
    m.reset()
    compiled_base = torch.compile(model)
    with torch.no_grad():
        out_base = compiled_base(*inputs)
    print(f"Baseline: {m.generated_kernel_count} kernels")
    base_us = bench(lambda: compiled_base(*inputs))
    print(f"  Time: {base_us:.1f} us")

    # --- Combo kernels ---
    cfg.combo_kernels = True
    m.reset()
    compiled_combo = torch.compile(Repro().cuda())
    with torch.no_grad():
        out_combo = compiled_combo(*inputs)
    print(f"\nCombo: {m.generated_kernel_count} kernels")
    combo_us = bench(lambda: compiled_combo(*inputs))
    print(f"  Time: {combo_us:.1f} us")
    cfg.combo_kernels = False

    # --- Custom fused (1 kernel) ---
    q_out = torch.empty(B, H_q, S, D, dtype=torch.bfloat16, device='cuda')
    k_out = torch.empty(B, H_q, S, D, dtype=torch.bfloat16, device='cuda')

    grid = (B * S * H_q,)
    stride_q_bs = S * H_q * D  # q_proj is [B*S, H_q*D]
    stride_k_bs = S * H_kv * D

    def run_custom():
        fused_qk_rope_expand_kernel[grid](
            q_proj, k_proj, cos, sin, q_out, k_out,
            stride_q_bs, stride_k_bs,
            B=B, S=S, H_q=H_q, H_kv=H_kv, D=D, HALF_D=D//2, N_rep=N_rep,
        )

    run_custom()  # compile
    torch.cuda.synchronize()
    custom_us = bench(run_custom)
    print(f"\nCustom fused: 1 kernel")
    print(f"  Time: {custom_us:.1f} us")

    # --- Correctness check ---
    # Compare custom vs baseline
    q_base, k_base = out_base
    print(f"\nCorrectness:")
    print(f"  Q max diff: {(q_out - q_base).abs().max().item():.6f}")
    print(f"  K max diff: {(k_out - k_base).abs().max().item():.6f}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Baseline (3 kernels): {base_us:.1f} us")
    print(f"  Combo    (2 kernels): {combo_us:.1f} us  ({base_us/combo_us:.2f}x vs base)")
    print(f"  Custom   (1 kernel):  {custom_us:.1f} us  ({base_us/custom_us:.2f}x vs base)")
    total_bytes = (16 + 4 + 0.25 + 32) * 1e6  # Q read + K read(1x) + cos/sin + writes
    sol_us = total_bytes / 8e12 * 1e6
    print(f"  SOL (memcopy):        {sol_us:.1f} us  ({base_us/sol_us:.2f}x vs base)")
