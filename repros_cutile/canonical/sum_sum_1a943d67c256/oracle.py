"""cuTile port of sum_sum_1a943d67c256: GhostNet BN backward.

Three-kernel structure mirrors the Triton reference:
1. `_partial_dual_reduce_kernel` — split-K partial per-channel sums.
2. `_finalize_kernel` — reduces chunks; computes mean_term/prod_coeff/
   output_scale/scale_grad in-kernel.
3. `_epilogue_kernel` — per-element BN-backward writeback.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C = 120
IN_C = 240
H = 28
W = 28
HW = H * W
R = N * HW              # 401,408
NUMEL = N * C * HW      # 48,168,960
REDUCE_SCALE = 2.4912308673469386e-06

# Padded channel count (next pow2 >= 120).
C_PAD = 128

# Reduction chunking (mirrors Triton REDUCE_CHUNK_R=256, but we run one
# chunk per program with BLOCK_R rows loaded at a time).
BLOCK_R = 256               # rows per reduce program
NUM_R_CHUNKS = R // BLOCK_R  # 1568 (since R=401408, BLOCK_R=256)
NUM_R_CHUNKS_PAD = 2048      # next pow2 for finalize load


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


assert R == BLOCK_R * NUM_R_CHUNKS, (R, BLOCK_R, NUM_R_CHUNKS)


@ct.kernel
def _partial_dual_reduce_kernel(
    wide_ptr,          # bf16 [R, IN_C]  (nhwc-flattened)
    rhs_ptr,           # bf16 [R, C_PAD]
    mask_ptr,          # bf16 [R, C_PAD]
    fill_ptr,          # bf16 [1]
    centered_src_ptr,  # bf16 [R, C_PAD]
    mean_ptr,          # f32  [C_PAD]
    partial_where_ptr, # f32  [NUM_R_CHUNKS, C_PAD]
    partial_prod_ptr,  # f32  [NUM_R_CHUNKS, C_PAD]
    C_PAD_: ct.Constant[int],
    BLOCK_R_: ct.Constant[int],
):
    chunk = ct.bid(0)
    # Load a (BLOCK_R, C_PAD) tile
    wide = ct.load(
        wide_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rhs = ct.load(
        rhs_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
    )
    mask_v = ct.load(
        mask_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
    )
    src = ct.load(
        centered_src_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
    )
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))
    mean = ct.load(mean_ptr, index=(0,), shape=(C_PAD_,))

    wide_f = ct.astype(wide, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)
    add_value = wide_f + rhs_f
    fill_f = ct.astype(fill, ct.float32)
    fill_bc = ct.broadcast_to(ct.reshape(fill_f, (1, 1)), (BLOCK_R_, C_PAD_))
    where_value = ct.where(mask_v <= ct.astype(0.0, ct.bfloat16), fill_bc, add_value)

    src_f = ct.astype(src, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_PAD_))
    centered = src_f - mean_2d
    prod = where_value * centered

    p_where = ct.sum(where_value, axis=0)   # (C_PAD,)
    p_prod = ct.sum(prod, axis=0)
    ct.store(
        partial_where_ptr, index=(chunk, 0),
        tile=ct.reshape(p_where, (1, C_PAD_)),
    )
    ct.store(
        partial_prod_ptr, index=(chunk, 0),
        tile=ct.reshape(p_prod, (1, C_PAD_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_where_ptr,   # f32 [NUM_R_CHUNKS, C_PAD]
    partial_prod_ptr,    # f32 [NUM_R_CHUNKS, C_PAD]
    invstd_ptr,          # f32 [C_PAD]
    weight_ptr,          # f32 [C_PAD]
    sum_where_ptr,       # f32 [C_PAD]
    out_vec_ptr,         # f32 [C_PAD]
    mean_term_ptr,       # f32 [C_PAD]
    prod_coeff_ptr,      # f32 [C_PAD]
    output_scale_ptr,    # f32 [C_PAD]
    C_PAD_: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    SCALE_: ct.Constant[float],
):
    where_v = ct.load(
        partial_where_ptr, index=(0, 0), shape=(BLOCK_CHUNKS, C_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    prod_v = ct.load(
        partial_prod_ptr, index=(0, 0), shape=(BLOCK_CHUNKS, C_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    sum_where = ct.sum(where_v, axis=0)   # (C_PAD,)
    sum_prod = ct.sum(prod_v, axis=0)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_PAD_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C_PAD_,))

    mean_term = sum_where * SCALE_
    prod_scaled = sum_prod * SCALE_
    prod_coeff = prod_scaled * (invstd * invstd)
    scale = invstd * weight

    ct.store(sum_where_ptr, index=(0,), tile=sum_where)
    ct.store(out_vec_ptr, index=(0,), tile=sum_prod * invstd)
    ct.store(mean_term_ptr, index=(0,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(0,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(0,), tile=scale)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,           # bf16 [R, IN_C]
    rhs_ptr,            # bf16 [R, C_PAD]
    mask_ptr,           # bf16 [R, C_PAD]
    fill_ptr,           # bf16 [1]
    centered_src_ptr,   # bf16 [R, C_PAD]
    mean_ptr,           # f32  [C_PAD]
    mean_term_ptr,      # f32  [C_PAD]
    prod_coeff_ptr,     # f32  [C_PAD]
    output_scale_ptr,   # f32  [C_PAD]
    out_ptr,            # bf16 [R, C_PAD]
    C_PAD_: ct.Constant[int],
    BLOCK_R_: ct.Constant[int],
):
    chunk = ct.bid(0)
    wide = ct.load(
        wide_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    rhs = ct.load(rhs_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    mask_v = ct.load(mask_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    src = ct.load(centered_src_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    mean = ct.load(mean_ptr, index=(0,), shape=(C_PAD_,))
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(C_PAD_,))
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(C_PAD_,))
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(C_PAD_,))

    # Compute bf16-rounded add, then select.
    add_bf16 = ct.astype(
        ct.astype(wide, ct.float32) + ct.astype(rhs, ct.float32),
        ct.bfloat16,
    )
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_R_, C_PAD_))
    where_bf = ct.where(mask_v <= ct.astype(0.0, ct.bfloat16), fill_bc, add_bf16)
    where_f = ct.astype(where_bf, ct.float32)

    src_f = ct.astype(src, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_PAD_))
    mean_term_2d = ct.reshape(mean_term, (1, C_PAD_))
    prod_coeff_2d = ct.reshape(prod_coeff, (1, C_PAD_))
    output_scale_2d = ct.reshape(output_scale, (1, C_PAD_))
    centered = src_f - mean_2d
    correction = centered * prod_coeff_2d
    after_variance = where_f - correction
    after_mean = after_variance - mean_term_2d
    grad = after_mean * output_scale_2d
    ct.store(out_ptr, index=(chunk, 0), tile=ct.astype(grad, ct.bfloat16))


@oracle_impl(hardware="B200", point="11c59fce")
def oracle_forward(inputs):
    (
        arg0_1,  # bf16 [N, 240, H, W] channels-last
        arg1_1,  # bf16 [N, 120, H, W] channels-last
        arg2_1,  # bf16 [N, 120, H, W] channels-last (mask)
        arg3_1,  # bf16 [] fill
        arg4_1,  # bf16 [N, 120, H, W] channels-last (centered_src)
        arg5_1,  # f32  [1, 120, 1, 1] mean
        arg6_1,  # f32  [120] invstd
        arg7_1,  # f32  [120] weight
    ) = inputs
    device = arg1_1.device

    # Reshape to NHWC-flat (R, IN_C) / (R, C) contiguous.
    def _nhwc_flat(t, c_dim):
        return t.permute(0, 2, 3, 1).contiguous().view(R, c_dim)

    wide = _nhwc_flat(arg0_1, IN_C)         # (R, 240)
    rhs = _nhwc_flat(arg1_1, C)             # (R, 120)
    mask = _nhwc_flat(arg2_1, C)
    src = _nhwc_flat(arg4_1, C)

    # Pad channel dim to C_PAD=128 for the narrow tensors (fill zeros in
    # the padding; the reduce won't be affected because C is loaded to
    # BLOCK_C=128 anyway).
    def _pad_c(t):
        pad = torch.zeros((R, C_PAD - C), device=device, dtype=t.dtype)
        return torch.cat([t, pad], dim=1).contiguous()

    rhs_pad = _pad_c(rhs)
    mask_pad = _pad_c(mask)
    src_pad = _pad_c(src)

    def _pad_c_vec(v):
        pad = torch.zeros((C_PAD - C,), device=device, dtype=v.dtype)
        return torch.cat([v, pad], dim=0).contiguous()

    mean_pad = _pad_c_vec(arg5_1.view(C))
    invstd_pad = _pad_c_vec(arg6_1)
    weight_pad = _pad_c_vec(arg7_1)
    fill_1d = arg3_1.reshape(1)

    partial_where = torch.empty(
        (NUM_R_CHUNKS, C_PAD), device=device, dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_where)
    sum_where_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    out_vec_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    mean_term_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    prod_coeff_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    output_scale_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUM_R_CHUNKS, 1, 1), _partial_dual_reduce_kernel,
        (wide, rhs_pad, mask_pad, fill_1d, src_pad, mean_pad,
         partial_where, partial_prod, C_PAD, BLOCK_R),
    )
    ct.launch(
        stream, (1, 1, 1), _finalize_kernel,
        (partial_where, partial_prod, invstd_pad, weight_pad,
         sum_where_pad, out_vec_pad, mean_term_pad, prod_coeff_pad,
         output_scale_pad, C_PAD, NUM_R_CHUNKS_PAD, REDUCE_SCALE),
    )

    out_flat_pad = torch.empty((R, C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (NUM_R_CHUNKS, 1, 1), _epilogue_kernel,
        (wide, rhs_pad, mask_pad, fill_1d, src_pad, mean_pad,
         mean_term_pad, prod_coeff_pad, output_scale_pad, out_flat_pad,
         C_PAD, BLOCK_R),
    )

    # Trim padding, reshape back to channels-last (N, C, H, W).
    sum_where = sum_where_pad[:C].contiguous()
    out_vec = out_vec_pad[:C].contiguous()
    out_flat = out_flat_pad[:, :C].contiguous()
    out_nhwc = out_flat.view(N, H, W, C)
    out_cl = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_where, out_vec, out_cl
