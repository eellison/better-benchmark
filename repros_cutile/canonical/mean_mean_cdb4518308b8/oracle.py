"""cuTile port of mean_mean_cdb4518308b8: Qwen3 RMSNorm + RoPE + grouped-KV.

Two RMSNorm kernels (Q with 16 heads, K with 8 heads, per-head hidden=128) plus
sin/cos RoPE tables computed via torch. cuTile does the fp32 mean-square /
rsqrt / affine-multiply / bf16 cast for each token's per-head vector.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
PER_HEAD = 128


@ct.kernel
def _rmsnorm_perhead_kernel(
    x_ptr,        # bf16 [tokens*heads, 128]
    scale_ptr,    # bf16 [128]
    out_ptr,      # bf16 [tokens*heads, 128]
    EPS_C: ct.Constant[float],
    HEAD_DIM: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(x_ptr, index=(row, 0), shape=(1, HEAD_DIM))
    x_f = ct.astype(x, ct.float32)
    inv_h = 1.0 / HEAD_DIM
    ms = ct.sum(x_f * x_f) * inv_h
    inv_rms = ct.rsqrt(ms + EPS_C)
    normalized = x_f * inv_rms
    normalized_bf = ct.astype(normalized, ct.bfloat16)

    scale = ct.load(scale_ptr, index=(0,), shape=(HEAD_DIM,))
    scale_2d = ct.reshape(scale, (1, HEAD_DIM))
    out = scale_2d * normalized_bf
    ct.store(out_ptr, index=(row, 0), tile=out)


def _rmsnorm_perhead(x_bf16, scale_bf16, tokens_heads):
    """Per-head RMSNorm on a [1, T, H, 128] tensor. Returns [1, T, H, 128]."""
    T, H, D = x_bf16.shape[1], x_bf16.shape[2], x_bf16.shape[3]
    assert D == PER_HEAD
    flat = x_bf16.reshape(T * H, D)
    out = torch.empty_like(flat)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (T * H, 1, 1),
        _rmsnorm_perhead_kernel,
        (flat, scale_bf16, out, EPS, PER_HEAD),
    )
    return out.view(1, T, H, D)


@oracle_impl(hardware="B200", point="b0137972")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     s0, s1, s2, s3, s4, s5, s6, s7, s8, s9) = inputs
    device = arg0_1.device
    T = 1000

    # Q RMSNorm ([1, 1000, 16, 128])
    q_view = arg0_1.view(1, T, 16, PER_HEAD)
    q_norm = _rmsnorm_perhead(q_view, arg1_1, T * 16)  # [1, T, 16, 128]

    # RoPE table (torch)
    inv_freq = arg2_1.view(1, 64, 1).float().expand(1, 64, 1)  # arg2 is [64] bf16
    positions = torch.arange(T, device=device, dtype=torch.float32).view(1, 1, T)
    freqs = (inv_freq * positions).permute(0, 2, 1)  # [1, T, 64]
    # Expand to [1, T, 2, 64] then reshape to [1, T, 128]
    freqs_exp = freqs.unsqueeze(2).expand(1, T, 2, 64).contiguous().view(1, T, 2 * 64)
    cos = torch.cos(freqs_exp) * 1.0
    cos_bf16 = cos.to(torch.bfloat16)
    sin = torch.sin(freqs_exp) * 1.0
    sin_bf16 = sin.to(torch.bfloat16)

    # Q rotation
    q_perm = q_norm.permute(0, 2, 1, 3)  # [1, 16, T, 128]
    cos_u = cos_bf16.unsqueeze(1)  # [1, 1, T, 128]
    sin_u = sin_bf16.unsqueeze(1)
    q_mul_cos = q_perm * cos_u
    q_slice_hi = q_perm[..., 64:]
    q_slice_lo = q_perm[..., :64]
    q_rot = torch.cat([-q_slice_hi, q_slice_lo], dim=-1)
    q_out = q_mul_cos + q_rot * sin_u  # add_2, bf16 [1, 16, T, 128]

    # K RMSNorm ([1, T, 8, 128])
    k_view = arg3_1.view(1, T, 8, PER_HEAD)
    k_norm = _rmsnorm_perhead(k_view, arg4_1, T * 8)
    k_perm = k_norm.permute(0, 2, 1, 3)  # [1, 8, T, 128]
    k_mul_cos = k_perm * cos_u
    k_slice_hi = k_perm[..., 64:]
    k_slice_lo = k_perm[..., :64]
    k_rot = torch.cat([-k_slice_hi, k_slice_lo], dim=-1)
    k_out = k_mul_cos + k_rot * sin_u  # add_4, bf16 [1, 8, T, 128]

    # Repeated K view: [1, 8, T, 128] -> unsqueeze(2) -> [1, 8, 1, T, 128] -> expand [1,8,2,T,128] -> view [1,16,T,128]
    k_expand = k_out.unsqueeze(2).expand(1, 8, 2, T, PER_HEAD).contiguous()
    view_5 = k_expand.view(1, 16, T, PER_HEAD)

    return cos_bf16, sin_bf16, q_out, k_out, view_5
