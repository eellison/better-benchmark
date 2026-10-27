"""cuTile port of var_mean_cc73cf06da4e: MobileViT BN-training + SiLU + concat.

Multi-kernel plan:
1. BN stats reduction (cuTile partial + finalize).
2. BN affine + bf16 round + SiLU (cuTile).
3. Concat with arg5_1 (torch, since it's a straightforward memcpy along C).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5
MOMENTUM = 0.1
ONE_MINUS_MOM = 0.9
UNBIAS_CORRECTION = 1.0001220852154804  # captured literal (varies at capture; kept as trace-time value)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@ct.kernel
def _bn_partial_stats_kernel(
    x_ptr,       # bf16 [E, C]
    psum_ptr,    # f32  [num_chunks, C]
    psq_ptr,     # f32  [num_chunks, C]
    E: ct.Constant[int],
    BLOCK_E: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
):
    chunk = ct.bid(0)
    cblk = ct.bid(1)
    x = ct.load(x_ptr, index=(chunk, cblk), shape=(BLOCK_E, C_BLOCK),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    e_arange = ct.arange(BLOCK_E, dtype=ct.int32)
    e_global = chunk * BLOCK_E + e_arange
    e_valid = e_global < E
    e_valid_2d = ct.reshape(e_valid, (BLOCK_E, 1))
    zero = ct.full((BLOCK_E, C_BLOCK), 0.0, dtype=ct.float32)
    x_masked = ct.where(e_valid_2d, x_f, zero)
    sums = ct.sum(x_masked, axis=0, keepdims=True)
    sums_sq = ct.sum(x_masked * x_masked, axis=0, keepdims=True)
    ct.store(psum_ptr, index=(chunk, cblk), tile=sums)
    ct.store(psq_ptr, index=(chunk, cblk), tile=sums_sq)


@ct.kernel
def _bn_finalize_stats_kernel(
    psum_ptr, psq_ptr,
    running_mean_ptr, running_var_ptr,
    saved_mean_ptr, invstd_ptr,
    NUM_CHUNKS: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    C_BLOCK: ct.Constant[int],
    INV_E: ct.Constant[float],
    EPS_: ct.Constant[float],
    UNBIAS: ct.Constant[float],
    MOM: ct.Constant[float],
    ONE_MINUS_MOM_: ct.Constant[float],
):
    cblk = ct.bid(0)
    psum = ct.load(psum_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
                   padding_mode=ct.PaddingMode.ZERO)
    psq = ct.load(psq_ptr, index=(0, cblk), shape=(BLOCK_CHUNKS, C_BLOCK),
                  padding_mode=ct.PaddingMode.ZERO)
    chunks = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    chunk_mask = chunks < NUM_CHUNKS
    chunk_mask_2d = ct.reshape(chunk_mask, (BLOCK_CHUNKS, 1))
    zero = ct.full((BLOCK_CHUNKS, C_BLOCK), 0.0, dtype=ct.float32)
    total = ct.sum(ct.where(chunk_mask_2d, psum, zero), axis=0)
    total_sq = ct.sum(ct.where(chunk_mask_2d, psq, zero), axis=0)
    mean = total * INV_E
    ex2 = total_sq * INV_E
    diff = ex2 - mean * mean
    zero_c = ct.full((C_BLOCK,), 0.0, dtype=ct.float32)
    var = ct.where(diff > zero_c, diff, zero_c)
    invstd = ct.rsqrt(var + EPS_)
    old_mean = ct.load(running_mean_ptr, index=(cblk,), shape=(C_BLOCK,))
    old_var = ct.load(running_var_ptr, index=(cblk,), shape=(C_BLOCK,))
    new_mean = mean * MOM + old_mean * ONE_MINUS_MOM_
    new_var = (var * UNBIAS) * MOM + old_var * ONE_MINUS_MOM_
    ct.store(saved_mean_ptr, index=(cblk,), tile=mean)
    ct.store(invstd_ptr, index=(cblk,), tile=invstd)
    ct.store(running_mean_ptr, index=(cblk,), tile=new_mean)
    ct.store(running_var_ptr, index=(cblk,), tile=new_var)


@ct.kernel
def _bn_affine_silu_kernel(
    x_ptr,           # bf16 [B, C, HW]  (contig NCHW flattened)
    saved_mean_ptr,  # f32 [C]
    invstd_ptr,      # f32 [C]
    weight_ptr,      # f32 [C]
    bias_ptr,        # f32 [C]
    out_ptr,         # bf16 [B, C, HW]
    HW: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)
    x = ct.load(x_ptr, index=(b, c, 0), shape=(1, 1, HW))
    mean = ct.load(saved_mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))
    x_f = ct.astype(x, ct.float32)
    centered = x_f - ct.reshape(mean, (1, 1, 1))
    normalized = centered * ct.reshape(invstd, (1, 1, 1))
    scaled = normalized * ct.reshape(weight, (1, 1, 1))
    affine = scaled + ct.reshape(bias, (1, 1, 1))
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)
    silu = affine_f / (1.0 + ct.exp(-affine_f))
    ct.store(out_ptr, index=(b, c, 0), tile=ct.astype(silu, ct.bfloat16))


@oracle_impl(hardware="B200", point="0be627fe")
@oracle_impl(hardware="B200", point="813b1e31")
@oracle_impl(hardware="B200", point="fe6f268d")
def oracle_forward(inputs, **kwargs):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    device = arg0_1.device
    B, C, H, W = arg0_1.shape
    B, C, H, W = int(B), int(C), int(H), int(W)
    HW = H * W
    E = B * H * W

    BLOCK_E = 512
    C_BLOCK = 8
    while C % C_BLOCK != 0:
        C_BLOCK //= 2
    num_chunks = (E + BLOCK_E - 1) // BLOCK_E
    block_chunks = _next_power_of_2(num_chunks)
    UNBIAS = float(E) / max(1.0, float(E) - 1.0)

    x_flat = arg0_1.permute(0, 2, 3, 1).reshape(E, C).contiguous()

    partial_sum = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    partial_sq = torch.empty((num_chunks, C), device=device, dtype=torch.float32)
    saved_mean = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=device, dtype=torch.float32)
    invstd = torch.empty_strided((1, C, 1, 1), (C, 1, 1, 1), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_chunks, C // C_BLOCK, 1),
        _bn_partial_stats_kernel,
        (x_flat, partial_sum, partial_sq, E, BLOCK_E, C_BLOCK),
    )
    saved_mean_1d = saved_mean.view(C)
    invstd_1d = invstd.view(C)
    ct.launch(
        stream, (C // C_BLOCK, 1, 1),
        _bn_finalize_stats_kernel,
        (partial_sum, partial_sq, arg1_1, arg2_1, saved_mean_1d, invstd_1d,
         num_chunks, block_chunks, C_BLOCK,
         1.0 / float(E), EPS, UNBIAS, MOMENTUM, ONE_MINUS_MOM),
    )

    # SiLU output [B, C, H, W]
    x_2d = arg0_1.contiguous().view(B, C, HW)
    silu_flat = torch.empty((B, C, HW), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (B, C, 1), _bn_affine_silu_kernel,
        (x_2d, saved_mean_1d, invstd_1d, arg3_1, arg4_1, silu_flat, HW),
    )
    silu_bf = silu_flat.view(B, C, H, W)

    # Concat with arg5_1
    cat = torch.cat([arg5_1, silu_bf], dim=1)

    return saved_mean, invstd, cat, arg1_1, arg2_1
