"""
Standalone repro: Q/K RoPE + GQA expand — 3 kernels that should ideally be fewer.
"""
import sys
sys.path.insert(0, "/tmp/pytorch-work")
"""

Pattern from every GQA LLM (Mistral, Llama, Qwen, etc.) at every attention layer.
After QKV projection matmuls, inductor generates:
  - Kernel 0: Q RoPE (apply rotary embeddings to queries, 32 heads)
  - Kernel 1: K RoPE (apply rotary embeddings to keys, 8 heads)  
  - Kernel 2: GQA expand (repeat K from 8 -> 32 heads via expand+clone)

Kernels 0 and 1 both read the same cos/sin embeddings but can't fuse because:
  - Different total elements: Q=4*32*512*128=8M, K=4*8*512*128=2M
  - MemoryDep for cos/sin has unused broadcast dim with different ranges:
    Q: MemoryDep('cos', 128*d1 + d3, {d0:4, d1:512, d2:32, d3:128})
    K: MemoryDep('cos', 128*d1 + d3, {d0:4, d1:512, d2:8,  d3:128})
    d0 and d2 don't appear in the index expression but differ in range.

Kernel 1 -> 2 can't fuse because the write and read have different loop
structures (K writes [4,512,8,128], GQA reads [4,8,4,512,128]).

Extracted from: vllm Mistral-7B-Instruct-v0.3 inference, region_011
"""
import torch

class Repro(torch.nn.Module):
    def forward(
        self,
        q_proj: "bf16[2048, 4096]",    # output of Q linear: [B*S, num_heads*head_dim]
        cos: "bf16[1, 512, 128]",       # RoPE cosine embeddings
        sin: "bf16[1, 512, 128]",       # RoPE sine embeddings
        k_proj: "bf16[2048, 1024]",     # output of K linear: [B*S, num_kv_heads*head_dim]
    ):
        B, S, H_q, H_kv, D = 4, 512, 32, 8, 128
        N_rep = H_q // H_kv  # 4x GQA expansion

        # === Q RoPE (kernel 0) ===
        q = q_proj.view(B, S, H_q, D).permute(0, 2, 1, 3)  # [4, 32, 512, 128]
        cos_q = cos.unsqueeze(1)  # [1, 1, 512, 128] — broadcast over batch and heads
        sin_q = sin.unsqueeze(1)
        q1, q2 = q[..., :D//2], q[..., D//2:]
        q_rotated = torch.cat((-q2, q1), dim=-1)
        q_embed = q * cos_q + q_rotated * sin_q  # [4, 32, 512, 128]

        # === K RoPE (kernel 1) — reads SAME cos/sin ===
        k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)  # [4, 8, 512, 128]
        k1, k2 = k[..., :D//2], k[..., D//2:]
        k_rotated = torch.cat((-k2, k1), dim=-1)
        k_embed = k * cos_q + k_rotated * sin_q  # [4, 8, 512, 128]

        # === GQA expand (kernel 2) — repeats K heads ===
        k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D)
        k_expanded = k_expanded.clone().reshape(B, H_q, S, D)  # [4, 32, 512, 128]

        return q_embed, k_expanded


def make_inputs():
    return [
        torch.randn(2048, 4096, dtype=torch.bfloat16, device='cuda'),
        torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda'),
        torch.randn(1, 512, 128, dtype=torch.bfloat16, device='cuda'),
        torch.randn(2048, 1024, dtype=torch.bfloat16, device='cuda'),
    ]


if __name__ == "__main__":
    import torch._inductor.config as cfg
    import torch._inductor.metrics as m
    import os

    # os.environ["TORCH_LOGS"] = "+inductor_metrics,+fusion"
    torch._logging._init_logs()
    cfg.force_disable_caches = True

    model = Repro().cuda()
    inputs = make_inputs()

    # Eager
    with torch.no_grad():
        eager = model(*inputs)

    # Compiled
    m.reset()
    compiled = torch.compile(model)
    with torch.no_grad():
        out = compiled(*inputs)

    print(f"\nKernels generated: {m.generated_kernel_count}")
    print(f"Bytes accessed: {m.num_bytes_accessed / 1e6:.1f} MB")

    # Correctness
    for i, (e, o) in enumerate(zip(eager, out)):
        ok = torch.allclose(e, o, atol=0.1, rtol=0.1)
        print(f"Output {i}: {'PASS' if ok else 'FAIL'} (max diff: {(e-o).abs().max().item():.6f})")
