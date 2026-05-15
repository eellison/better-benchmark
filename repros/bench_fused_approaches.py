"""Compare inductor's recompute fusion vs ideal multi-store kernel."""
import sys
sys.path.insert(0, "/tmp/pytorch-work")

import torch
import triton
import triton.language as tl
from triton.testing import do_bench
import torch._inductor.config as cfg

cfg.force_disable_caches = True

B, S, H_kv, D, N_rep = 4, 512, 8, 128, 4


# === Ideal multi-store kernel: iterate 2M, store 4x ===
@triton.jit
def k_rope_gqa_multistore(
    k_proj_ptr, cos_ptr, sin_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H_kv: tl.constexpr,
    D: tl.constexpr, N_rep: tl.constexpr, XBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < B * H_kv * S * D

    head_dim = xindex % D
    seq = (xindex // D) % S
    kv_head = (xindex // (D * S)) % H_kv
    batch = xindex // (D * S * H_kv)

    HALF_D: tl.constexpr = D // 2

    k_proj_idx = (batch * S + seq) * (H_kv * D) + kv_head * D + head_dim
    k_val = tl.load(k_proj_ptr + k_proj_idx, mask=xmask).to(tl.float32)

    cs_idx = seq * D + head_dim
    cos_val = tl.load(cos_ptr + cs_idx, mask=xmask).to(tl.float32)
    sin_val = tl.load(sin_ptr + cs_idx, mask=xmask).to(tl.float32)

    is_first_half = head_dim < HALF_D
    rot_idx = tl.where(is_first_half, k_proj_idx + HALF_D, k_proj_idx - HALF_D)
    k_rot_val = tl.load(k_proj_ptr + rot_idx, mask=xmask).to(tl.float32)
    k_rotated = tl.where(is_first_half, -k_rot_val, k_rot_val)

    result = (k_val * cos_val + k_rotated * sin_val).to(tl.bfloat16)

    # 4 stores: one per GQA repeat
    base_out = batch * (H_kv * N_rep * S * D) + kv_head * (N_rep * S * D) + seq * D + head_dim
    stride_rep = S * D
    tl.store(out_ptr + base_out + 0 * stride_rep, result, mask=xmask)
    tl.store(out_ptr + base_out + 1 * stride_rep, result, mask=xmask)
    tl.store(out_ptr + base_out + 2 * stride_rep, result, mask=xmask)
    tl.store(out_ptr + base_out + 3 * stride_rep, result, mask=xmask)


# === Recompute kernel: iterate 8M, same logic as inductor generates ===
@triton.jit
def k_rope_gqa_recompute(
    k_proj_ptr, cos_ptr, sin_ptr, out_ptr,
    B: tl.constexpr, S: tl.constexpr, H_kv: tl.constexpr,
    D: tl.constexpr, N_rep: tl.constexpr, XBLOCK: tl.constexpr,
):
    # Iterate over consumer domain: B * H_kv * N_rep * S * D = 8M
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < B * H_kv * N_rep * S * D

    head_dim = xindex % D
    seq = (xindex // D) % S
    rep = (xindex // (D * S)) % N_rep  # noqa: not used in loads
    kv_head = (xindex // (D * S * N_rep)) % H_kv
    batch = xindex // (D * S * N_rep * H_kv)

    HALF_D: tl.constexpr = D // 2

    # Same loads as multi-store but hit 4x due to broadcast over rep
    k_proj_idx = (batch * S + seq) * (H_kv * D) + kv_head * D + head_dim
    k_val = tl.load(k_proj_ptr + k_proj_idx, mask=xmask).to(tl.float32)

    cs_idx = seq * D + head_dim
    cos_val = tl.load(cos_ptr + cs_idx, mask=xmask).to(tl.float32)
    sin_val = tl.load(sin_ptr + cs_idx, mask=xmask).to(tl.float32)

    is_first_half = head_dim < HALF_D
    rot_idx = tl.where(is_first_half, k_proj_idx + HALF_D, k_proj_idx - HALF_D)
    k_rot_val = tl.load(k_proj_ptr + rot_idx, mask=xmask).to(tl.float32)
    k_rotated = tl.where(is_first_half, -k_rot_val, k_rot_val)

    result = (k_val * cos_val + k_rotated * sin_val).to(tl.bfloat16)

    # 1 store per element
    tl.store(out_ptr + xindex, result, mask=xmask)


def main():
    torch.manual_seed(42)

    k_proj = torch.randn(B * S, H_kv * D, dtype=torch.bfloat16, device='cuda')
    cos = torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda')
    sin = torch.randn(1, S, D, dtype=torch.bfloat16, device='cuda')

    # Reference
    k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
    cos_q = cos.unsqueeze(1)
    sin_q = sin.unsqueeze(1)
    k1, k2 = k[..., :D // 2], k[..., D // 2:]
    k_rotated = torch.cat((-k2, k1), dim=-1)
    k_embed = k * cos_q + k_rotated * sin_q
    ref = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D).clone()

    XBLOCK = 1024

    # Multi-store: 2M iteration
    numel_producer = B * H_kv * S * D
    grid_ms = ((numel_producer + XBLOCK - 1) // XBLOCK,)
    out_ms = torch.empty(B, H_kv, N_rep, S, D, dtype=torch.bfloat16, device='cuda')
    k_rope_gqa_multistore[grid_ms](k_proj, cos, sin, out_ms, B, S, H_kv, D, N_rep, XBLOCK)
    diff_ms = (ref - out_ms).abs().max().item()

    # Recompute: 8M iteration
    numel_consumer = B * H_kv * N_rep * S * D
    grid_rc = ((numel_consumer + XBLOCK - 1) // XBLOCK,)
    out_rc = torch.empty(B, H_kv, N_rep, S, D, dtype=torch.bfloat16, device='cuda')
    k_rope_gqa_recompute[grid_rc](k_proj, cos, sin, out_rc, B, S, H_kv, D, N_rep, XBLOCK)
    diff_rc = (ref - out_rc).abs().max().item()

    # Inductor compiled (with our fix)
    class KModel(torch.nn.Module):
        def forward(self, k_proj, cos, sin):
            k = k_proj.view(B, S, H_kv, D).permute(0, 2, 1, 3)
            cos_q = cos.unsqueeze(1)
            sin_q = sin.unsqueeze(1)
            k1, k2 = k[..., :D // 2], k[..., D // 2:]
            k_rotated = torch.cat((-k2, k1), dim=-1)
            k_embed = k * cos_q + k_rotated * sin_q
            k_expanded = k_embed.unsqueeze(2).expand(B, H_kv, N_rep, S, D)
            return k_expanded.clone().reshape(B, H_kv * N_rep, S, D)

    compiled = torch.compile(KModel().cuda())
    with torch.no_grad():
        compiled(k_proj, cos, sin)  # warmup

    # Baseline: inductor WITHOUT our fix (2 kernels) — stash changes first
    # We can't easily test this here, so compare against the 2-kernel cost
    # K RoPE kernel: 2M elements. GQA copy kernel: 8M elements.

    # Benchmark
    ms_multistore = do_bench(lambda: k_rope_gqa_multistore[grid_ms](k_proj, cos, sin, out_ms, B, S, H_kv, D, N_rep, XBLOCK))
    ms_recompute = do_bench(lambda: k_rope_gqa_recompute[grid_rc](k_proj, cos, sin, out_rc, B, S, H_kv, D, N_rep, XBLOCK))
    ms_inductor = do_bench(lambda: compiled(k_proj, cos, sin))

    print(f"Multi-store (2M iter, 4 stores):  {ms_multistore:.4f} ms  diff={diff_ms:.4f}")
    print(f"Recompute  (8M iter, 1 store):    {ms_recompute:.4f} ms  diff={diff_rc:.4f}")
    print(f"Inductor fused (our fix):         {ms_inductor:.4f} ms")
    print()
    print(f"Multi-store vs recompute: {ms_recompute/ms_multistore:.2f}x")
    print(f"Multi-store vs inductor:  {ms_inductor/ms_multistore:.2f}x")


if __name__ == "__main__":
    main()
