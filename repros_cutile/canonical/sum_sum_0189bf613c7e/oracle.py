"""cuTile port of sum_sum_0189bf613c7e (COOPERATIVE_SPLIT_K): GhostNet BN
backward with slice/add/mask producer + channel reductions + dense epilogue.

Three-kernel structure mirrors the Triton reference:
1. `_partial_dual_reduce_kernel` — split-K partial per-channel sums for the
   masked (wide + rhs)-slice producer and its centered product.
2. `_finalize_kernel` — reduces chunks; computes mean_term/prod_coeff/
   output_scale/scale_grad in-kernel.
3. `_epilogue_kernel` — per-element BN-backward writeback.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BATCH = 512
IN_CHANNELS = 48
CHANNELS = 24
HEIGHT = 112
WIDTH = 112
HW = HEIGHT * WIDTH
R = BATCH * HW              # 6422528
REDUCE_SCALE = 1.5570192920918366e-07

# Padded channel count (next pow2 >= 24 is 32).
C_PAD = 32
IN_C_PAD = 64  # >= 48

BLOCK_R = 1024
NUM_R_CHUNKS = R // BLOCK_R  # 6272


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


NUM_R_CHUNKS_PAD = _next_pow2(NUM_R_CHUNKS)  # 8192

assert R == BLOCK_R * NUM_R_CHUNKS


@ct.kernel
def _partial_dual_reduce_kernel(
    wide_ptr,          # bf16 [R, IN_C_PAD]  (nhwc-flat, zero-padded to IN_C_PAD)
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
    wide = ct.load(
        wide_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_),
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
    # bf16-rounded add (matches Triton's _bf16_add_to_f32).
    add_bf = ct.astype(wide_f + rhs_f, ct.bfloat16)
    add_value = ct.astype(add_bf, ct.float32)
    fill_f = ct.astype(fill, ct.float32)
    fill_bc = ct.broadcast_to(ct.reshape(fill_f, (1, 1)), (BLOCK_R_, C_PAD_))
    mask_f = ct.astype(mask_v, ct.float32)
    where_value = ct.where(mask_f <= 0.0, fill_bc, add_value)

    src_f = ct.astype(src, ct.float32)
    mean_2d = ct.reshape(mean, (1, C_PAD_))
    centered = src_f - mean_2d
    prod = where_value * centered

    p_where = ct.sum(where_value, axis=0)  # (C_PAD,)
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
    scale_grad_ptr,      # f32 [C_PAD]
    mean_term_ptr,       # f32 [C_PAD]
    prod_coeff_ptr,      # f32 [C_PAD]
    output_scale_ptr,    # f32 [C_PAD]
    C_PAD_: ct.Constant[int],
    BLOCK_CHUNKS: ct.Constant[int],
    NUM_CHUNKS: ct.Constant[int],
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
    # Mask chunks past NUM_CHUNKS
    idx = ct.arange(BLOCK_CHUNKS, dtype=ct.int32)
    valid = ct.reshape(idx < NUM_CHUNKS, (BLOCK_CHUNKS, 1))
    where_v = ct.where(valid, where_v, 0.0)
    prod_v = ct.where(valid, prod_v, 0.0)
    sum_where = ct.sum(where_v, axis=0)   # (C_PAD,)
    sum_prod = ct.sum(prod_v, axis=0)

    invstd = ct.load(invstd_ptr, index=(0,), shape=(C_PAD_,))
    weight = ct.load(weight_ptr, index=(0,), shape=(C_PAD_,))

    mean_term = sum_where * SCALE_
    prod_scaled = sum_prod * SCALE_
    prod_coeff = prod_scaled * (invstd * invstd)
    scale = invstd * weight
    scale_grad = sum_prod * invstd

    ct.store(sum_where_ptr, index=(0,), tile=sum_where)
    ct.store(scale_grad_ptr, index=(0,), tile=scale_grad)
    ct.store(mean_term_ptr, index=(0,), tile=mean_term)
    ct.store(prod_coeff_ptr, index=(0,), tile=prod_coeff)
    ct.store(output_scale_ptr, index=(0,), tile=scale)


@ct.kernel
def _epilogue_kernel(
    wide_ptr,           # bf16 [R, IN_C_PAD]
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
    )
    rhs = ct.load(rhs_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    mask_v = ct.load(mask_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    src = ct.load(centered_src_ptr, index=(chunk, 0), shape=(BLOCK_R_, C_PAD_))
    fill = ct.load(fill_ptr, index=(0,), shape=(1,))

    mean = ct.load(mean_ptr, index=(0,), shape=(C_PAD_,))
    mean_term = ct.load(mean_term_ptr, index=(0,), shape=(C_PAD_,))
    prod_coeff = ct.load(prod_coeff_ptr, index=(0,), shape=(C_PAD_,))
    output_scale = ct.load(output_scale_ptr, index=(0,), shape=(C_PAD_,))

    # bf16-rounded add
    add_bf = ct.astype(
        ct.astype(wide, ct.float32) + ct.astype(rhs, ct.float32),
        ct.bfloat16,
    )
    fill_bc = ct.broadcast_to(ct.reshape(fill, (1, 1)), (BLOCK_R_, C_PAD_))
    mask_f = ct.astype(mask_v, ct.float32)
    add_f = ct.astype(add_bf, ct.float32)
    fill_f = ct.astype(fill, ct.float32)
    fill_f_bc = ct.broadcast_to(ct.reshape(fill_f, (1, 1)), (BLOCK_R_, C_PAD_))
    where_f = ct.where(mask_f <= 0.0, fill_f_bc, add_f)

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


@oracle_impl(hardware="B200", point="19c66925")
def oracle_forward(inputs):
    (
        arg0_1,  # bf16 [512, 48, 112, 112] channels-last
        arg1_1,  # bf16 [512, 24, 112, 112] channels-last (narrow rhs)
        arg2_1,  # bf16 [512, 24, 112, 112] channels-last (mask)
        arg3_1,  # bf16 [] fill
        arg4_1,  # bf16 [512, 24, 112, 112] channels-last (centered_src)
        arg5_1,  # f32  [1, 24, 1, 1] mean
        arg6_1,  # f32  [24] invstd
        arg7_1,  # f32  [24] weight
    ) = inputs
    device = arg1_1.device

    # Reshape channels-last strided tensors to NHWC-flat (R, C_or_IN_C).
    def _nhwc_flat(t, c_dim):
        # arg tensors have stride (C*HW, 1, W*C, C) which is NHWC-contiguous.
        return t.as_strided((R, c_dim), (c_dim, 1))

    # Note: the arg0 with IN_C=48 needs slicing [:, 0:24] semantics. But the
    # Triton `_partial_reduce_kernel` accesses arg0 via wide_offsets which
    # implicitly reads channels 0..C-1 of a 48-wide row. So we access with
    # stride pointing to first C columns of the wide tensor.
    # arg0_1 shape [N, 48, H, W] channels-last: NHWC layout with stride
    # (48*HW, 1, 48*W, 48). Cut to the first 24 channels:
    wide_full = arg0_1.as_strided((R, IN_CHANNELS), (IN_CHANNELS, 1))
    # For simplicity, pad the narrow tensors to C_PAD and use only first C
    # columns of the wide tensor (equivalent to Triton wide[:, 0:C] behavior).
    # We need wide_pad of shape (R, C_PAD) - just take first C_PAD from wide.
    # Only the first C=24 columns of wide are actually used, but we need
    # shape (R, C_PAD) for the kernel. Slice first C then pad.
    wide_first_c = wide_full[:, :CHANNELS].contiguous()  # (R, 24)

    rhs = _nhwc_flat(arg1_1, CHANNELS)      # (R, 24)
    mask = _nhwc_flat(arg2_1, CHANNELS)
    src = _nhwc_flat(arg4_1, CHANNELS)

    # Pad channel dim to C_PAD=32.
    def _pad_c(t, dtype):
        pad = torch.zeros((R, C_PAD - CHANNELS), device=device, dtype=dtype)
        return torch.cat([t.contiguous(), pad], dim=1).contiguous()

    wide_pad = _pad_c(wide_first_c, torch.bfloat16)
    rhs_pad = _pad_c(rhs, torch.bfloat16)
    mask_pad = _pad_c(mask, torch.bfloat16)
    src_pad = _pad_c(src, torch.bfloat16)

    def _pad_c_vec(v, dtype):
        pad = torch.zeros((C_PAD - CHANNELS,), device=device, dtype=dtype)
        return torch.cat([v, pad], dim=0).contiguous()

    mean_pad = _pad_c_vec(arg5_1.view(CHANNELS), torch.float32)
    invstd_pad = _pad_c_vec(arg6_1, torch.float32)
    weight_pad = _pad_c_vec(arg7_1, torch.float32)
    fill_1d = arg3_1.reshape(1)

    partial_where = torch.empty(
        (NUM_R_CHUNKS, C_PAD), device=device, dtype=torch.float32,
    )
    partial_prod = torch.empty_like(partial_where)
    sum_where_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    scale_grad_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    mean_term_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    prod_coeff_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)
    output_scale_pad = torch.empty((C_PAD,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (NUM_R_CHUNKS, 1, 1), _partial_dual_reduce_kernel,
        (wide_pad, rhs_pad, mask_pad, fill_1d, src_pad, mean_pad,
         partial_where, partial_prod, C_PAD, BLOCK_R),
    )
    ct.launch(
        stream, (1, 1, 1), _finalize_kernel,
        (partial_where, partial_prod, invstd_pad, weight_pad,
         sum_where_pad, scale_grad_pad, mean_term_pad, prod_coeff_pad,
         output_scale_pad, C_PAD, NUM_R_CHUNKS_PAD, NUM_R_CHUNKS, REDUCE_SCALE),
    )

    out_flat_pad = torch.empty((R, C_PAD), device=device, dtype=torch.bfloat16)
    ct.launch(
        stream, (NUM_R_CHUNKS, 1, 1), _epilogue_kernel,
        (wide_pad, rhs_pad, mask_pad, fill_1d, src_pad, mean_pad,
         mean_term_pad, prod_coeff_pad, output_scale_pad, out_flat_pad,
         C_PAD, BLOCK_R),
    )

    # Trim to C and reshape.
    sum_where = sum_where_pad[:CHANNELS].contiguous()
    scale_grad = scale_grad_pad[:CHANNELS].contiguous()
    out_flat = out_flat_pad[:, :CHANNELS].contiguous()
    out_nhwc = out_flat.view(BATCH, HEIGHT, WIDTH, CHANNELS)
    out_cl = torch.empty_strided(
        (BATCH, CHANNELS, HEIGHT, WIDTH),
        (CHANNELS * HW, 1, WIDTH * CHANNELS, CHANNELS),
        device=device, dtype=torch.bfloat16,
    )
    out_cl.copy_(out_nhwc.permute(0, 3, 1, 2))
    return sum_where, scale_grad, out_cl
