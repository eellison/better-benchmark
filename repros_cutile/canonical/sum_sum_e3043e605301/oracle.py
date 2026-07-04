"""cuTile port of sum_sum_e3043e605301: GhostNet BN-backward with hard-sigmoid pooled add.

Eager scope:
  1. add_1 = arg1 * hard_sigmoid_bf16(arg0) + expand(arg2, N,C,H,W)/49  (bf16, NHWC)
  2. slice_1 = add_1[:, 480:960]
  3. affine_bf16 = ((centered_source - mean) * invstd * weight + bias).to(bf16)
  4. where = torch.where(affine_bf16 <= 0, fill, slice_1)
  5. sum_1 = sum(where.f32, [0,2,3])                              (in cuTile)
  6. sub_1 = centered_source.f32 - mean
  7. sum_2 = sum(where.f32 * sub_1, [0,2,3])                      (in cuTile)
  8. dense = BN-backward.to(bf16)                                 (in cuTile)
  9. mul_11 = sum_2 * invstd  (returned scale_grad)

Torch computes the hard-sigmoid producer and the affine mask. cuTile performs
both channel reductions (sum_1, sum_2) via a partial-reduce + finalize kernel
pair, and the final BN backward as a channels-last-flat epilogue.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 512
C_FULL = 960
C = 480
H = 7
W = 7
HW = H * W        # 49
K_TOTAL = N * HW  # 25088
SCALE = 3.985969387755102e-05


@ct.kernel
def _partial_reduce_kernel(
    where_ptr,        # bf16 [K_TOTAL, C]  (NHWC-flat view of where_bf)
    csrc_ptr,         # bf16 [K_TOTAL, C]  (NHWC-flat view of arg3_1)
    mean_ptr,         # f32  [C]
    partial_sum_ptr,  # f32  [num_k_tiles, C]
    partial_dot_ptr,  # f32  [num_k_tiles, C]
    C_: ct.Constant[int],
    K_TOTAL_: ct.Constant[int],
    BLOCK_K_: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    k_block = ct.bid(1)

    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    k_local = ct.arange(BLOCK_K_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    k_idx = k_block * BLOCK_K_ + k_local
    c_valid = ct.reshape(c_idx < C_, (1, BLOCK_C_))
    k_valid = ct.reshape(k_idx < K_TOTAL_, (BLOCK_K_, 1))
    valid = c_valid & k_valid

    where_bf = ct.load(
        where_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    csrc_bf = ct.load(
        csrc_ptr, index=(k_block, c_block),
        shape=(BLOCK_K_, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    where_f = ct.astype(where_bf, ct.float32)
    csrc_f = ct.astype(csrc_bf, ct.float32)

    mean_1d = ct.load(
        mean_ptr, index=(c_block,), shape=(BLOCK_C_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean_f = ct.astype(mean_1d, ct.float32)
    mean_bcast = ct.reshape(mean_f, (1, BLOCK_C_))
    centered = csrc_f - mean_bcast

    zero_2d = ct.zeros((BLOCK_K_, BLOCK_C_), dtype=ct.float32)
    where_masked = ct.where(valid, where_f, zero_2d)
    centered_masked = ct.where(valid, centered, zero_2d)

    partial_sum = ct.sum(where_masked, axis=0)            # (BLOCK_C_,)
    partial_dot = ct.sum(where_masked * centered_masked, axis=0)
    ct.store(
        partial_sum_ptr, index=(k_block, c_block),
        tile=ct.reshape(partial_sum, (1, BLOCK_C_)),
    )
    ct.store(
        partial_dot_ptr, index=(k_block, c_block),
        tile=ct.reshape(partial_dot, (1, BLOCK_C_)),
    )


@ct.kernel
def _finalize_kernel(
    partial_sum_ptr,  # f32  [num_k_tiles, C]
    partial_dot_ptr,  # f32  [num_k_tiles, C]
    sum_out_ptr,      # f32  [C]  == sum_1
    dot_out_ptr,      # f32  [C]  == sum_2
    C_: ct.Constant[int],
    NUM_TILES: ct.Constant[int],
    BLOCK_TILES: ct.Constant[int],
    BLOCK_C_: ct.Constant[int],
):
    c_block = ct.bid(0)
    c_local = ct.arange(BLOCK_C_, dtype=ct.int32)
    c_idx = c_block * BLOCK_C_ + c_local
    c_valid = c_idx < C_

    tile_local = ct.arange(BLOCK_TILES, dtype=ct.int32)
    tile_valid = tile_local < NUM_TILES
    mask = ct.reshape(tile_valid, (BLOCK_TILES, 1)) & ct.reshape(c_valid, (1, BLOCK_C_))
    zero_2d = ct.zeros((BLOCK_TILES, BLOCK_C_), dtype=ct.float32)

    ps = ct.load(
        partial_sum_ptr, index=(0, c_block),
        shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    pd = ct.load(
        partial_dot_ptr, index=(0, c_block),
        shape=(BLOCK_TILES, BLOCK_C_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ps_masked = ct.where(mask, ps, zero_2d)
    pd_masked = ct.where(mask, pd, zero_2d)
    sum_value = ct.sum(ps_masked, axis=0)  # (BLOCK_C_,)
    dot_value = ct.sum(pd_masked, axis=0)

    ct.store(sum_out_ptr, index=(c_block,), tile=sum_value)
    ct.store(dot_out_ptr, index=(c_block,), tile=dot_value)


@ct.kernel
def _bn_epilogue_flat_kernel(
    where_ptr,        # bf16 [total]  (channels-last flat)
    centered_src_ptr, # bf16 [total]
    mean_ptr,         # f32  [C]
    coeff_ptr,        # f32  [C]  = sum_2 * SCALE * invstd^2
    mean_term_ptr,    # f32  [C]  = sum_1 * SCALE
    out_scale_ptr,    # f32  [C]  = invstd * weight
    out_ptr,          # bf16 [total]
    C_: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    """where and centered_src are viewed as (N*H*W, C) NHWC-flat rows —
    the last-dim stride is 1 and channel is the last axis. So the channel
    index of the k-th element in the flat array is `k % C`."""
    pid = ct.bid(0)
    where_bf = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    csrc_bf = ct.load(centered_src_ptr, index=(pid,), shape=(BLOCK,))
    where_f = ct.astype(where_bf, ct.float32)
    csrc_f = ct.astype(csrc_bf, ct.float32)

    idx = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int32)
    c = idx % C_

    mean = ct.gather(mean_ptr, c)
    coeff = ct.gather(coeff_ptr, c)
    mean_term = ct.gather(mean_term_ptr, c)
    out_scale = ct.gather(out_scale_ptr, c)

    centered = csrc_f - mean
    corrected = where_f - centered * coeff - mean_term
    scaled = corrected * out_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(scaled, ct.bfloat16))


@oracle_impl(
    hardware="B200", point="a1e026e9",
    PRODUCER_BLOCK=256, REDUCE_BLOCK_K=256, XBLOCK=8,
    EPILOGUE_BLOCK=256, BLOCK_TILES=128,
    BLOCK_C=16,
)
def oracle_forward(inputs, *, PRODUCER_BLOCK, REDUCE_BLOCK_K, XBLOCK,
                   EPILOGUE_BLOCK, BLOCK_TILES, BLOCK_C):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1,
     _shape0) = inputs
    device = arg1_1.device

    # Step 1 — Hard-sigmoid producer.
    # hard_sigmoid(gate) = clamp(gate + 3, 0, 6) / 6, in f32, then cast to bf16.
    hs_f = (arg0_1 + 3.0).clamp(min=0.0, max=6.0) / 6.0  # f32 [N, C_FULL, 1, 1]
    hs_bf = hs_f.to(torch.bfloat16)  # bf16
    mul_val = arg1_1 * hs_bf
    div_1 = arg2_1.expand(N, C_FULL, H, W) / 49
    add_1 = mul_val + div_1
    add_1 = add_1.contiguous(memory_format=torch.channels_last)

    # Step 2 — slice channels 480:960
    slice_1 = add_1[:, C:C_FULL, :, :]

    # Step 3 — affine mask
    sub = arg3_1 - arg4_1
    mul_1 = sub * arg5_1
    mul_2 = mul_1 * arg6_1.view(1, C, 1, 1)
    add_2 = mul_2 + arg7_1.view(1, C, 1, 1)
    affine_bf = add_2.to(torch.bfloat16)
    relu_bf = torch.relu(affine_bf)
    le_mask = relu_bf <= 0
    where_bf = torch.where(le_mask, arg8_1, slice_1)  # bf16 [N, C, H, W]

    # Prepare NHWC-flat views for the reductions and the BN epilogue.
    where_nhwc = where_bf.contiguous(memory_format=torch.channels_last)
    csrc_nhwc = arg3_1.contiguous(memory_format=torch.channels_last)
    where_2d = where_nhwc.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)
    csrc_2d = csrc_nhwc.permute(0, 2, 3, 1).contiguous().view(K_TOTAL, C)

    invstd_1d = arg5_1.view(C).contiguous()
    weight_1d = arg6_1.contiguous()
    mean_1d = arg4_1.view(C).contiguous()

    # Grid params for reductions.
    num_k_tiles = (K_TOTAL + REDUCE_BLOCK_K - 1) // REDUCE_BLOCK_K
    num_c_tiles = (C + BLOCK_C - 1) // BLOCK_C

    partial_sum = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_k_tiles, C), device=device, dtype=torch.float32)
    sum_1 = torch.empty((C,), device=device, dtype=torch.float32)
    sum_2 = torch.empty((C,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (num_c_tiles, num_k_tiles, 1), _partial_reduce_kernel,
        (where_2d, csrc_2d, mean_1d, partial_sum, partial_dot,
         C, K_TOTAL, REDUCE_BLOCK_K, BLOCK_C),
    )
    ct.launch(
        stream, (num_c_tiles, 1, 1), _finalize_kernel,
        (partial_sum, partial_dot, sum_1, sum_2,
         C, num_k_tiles, BLOCK_TILES, BLOCK_C),
    )

    # Step 8 — dense BN-backward
    mean_term = sum_1 * SCALE
    coeff = sum_2 * SCALE * invstd_1d * invstd_1d
    out_scale = invstd_1d * weight_1d
    mul_11 = sum_2 * invstd_1d  # returned

    dense_out = torch.empty_strided(
        (N, C, H, W), (C * HW, 1, W * C, C),
        device=device, dtype=torch.bfloat16,
    )

    where_flat = where_2d.view(-1)
    csrc_flat = csrc_2d.view(-1)
    dense_flat = dense_out.permute(0, 2, 3, 1).view(-1)

    total = N * C * HW  # 12042240
    BLOCK = EPILOGUE_BLOCK

    ct.launch(
        stream, (ct.cdiv(total, BLOCK), 1, 1),
        _bn_epilogue_flat_kernel,
        (where_flat, csrc_flat,
         mean_1d, coeff.contiguous(),
         mean_term.contiguous(), out_scale.contiguous(),
         dense_flat,
         C, BLOCK),
    )

    return add_1, sum_1, mul_11, dense_out
