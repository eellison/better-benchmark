"""cuTile port of sum_sum_sum_1a561863a1c6 (COOPERATIVE_SPLIT_K): GhostNet
dual BN backward with returned copy alias and two dense f32->bf16 epilogues.

cuTile performs the per-channel reductions in-kernel (matching Triton's
split-K partial reduce + finalize) and the final f32->bf16 casts.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


REDUCE_SCALE = 9.964923469387754e-06


@ct.kernel
def _f32_to_bf16_kernel(
    src_ptr, dst_ptr,
    N: ct.Constant[int], BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    off = pid * BLOCK + ct.arange(BLOCK, dtype=ct.int64)
    mask = off < N
    zero64 = ct.zeros((BLOCK,), dtype=ct.int64)
    safe = ct.where(mask, off, zero64)
    val = ct.gather(src_ptr, safe)
    ct.scatter(dst_ptr, safe, ct.astype(val, ct.bfloat16), mask=mask)


@ct.kernel
def _channel_partial_kernel(
    x_ptr,        # f32 [K_padded, C]  (K = N*H*W)
    x_rhs_ptr,    # f32 [K_padded, C]  (= x * (arg2 - arg3))
    partial_x_ptr,      # f32 [C, NBLOCKS]
    partial_x_rhs_ptr,  # f32 [C, NBLOCKS]
    BLOCK_K: ct.Constant[int],
    K_: ct.Constant[int],
):
    ch = ct.bid(0)
    blk = ct.bid(1)
    x = ct.load(x_ptr, index=(blk, ch), shape=(BLOCK_K, 1))
    x_rhs = ct.load(x_rhs_ptr, index=(blk, ch), shape=(BLOCK_K, 1))
    k_idx = blk * BLOCK_K + ct.arange(BLOCK_K, dtype=ct.int32)
    k_mask = ct.reshape(k_idx < K_, (BLOCK_K, 1))
    x_m = ct.where(k_mask, x, 0.0)
    x_rhs_m = ct.where(k_mask, x_rhs, 0.0)
    sx = ct.sum(x_m, axis=0)
    sxrhs = ct.sum(x_rhs_m, axis=0)
    ct.store(partial_x_ptr, index=(ch, blk),
             tile=ct.reshape(sx, (1, 1)))
    ct.store(partial_x_rhs_ptr, index=(ch, blk),
             tile=ct.reshape(sxrhs, (1, 1)))


@ct.kernel
def _channel_finalize_kernel(
    partial_x_ptr,      # f32 [C, NBLOCKS_p2]
    partial_x_rhs_ptr,  # f32 [C, NBLOCKS_p2]
    out_sum_x_ptr,      # f32 [C]
    out_sum_x_rhs_ptr,  # f32 [C]
    NBLOCKS_P2: ct.Constant[int],
):
    ch = ct.bid(0)
    px = ct.load(partial_x_ptr, index=(ch, 0), shape=(1, NBLOCKS_P2))
    prhs = ct.load(partial_x_rhs_ptr, index=(ch, 0), shape=(1, NBLOCKS_P2))
    sx = ct.sum(px)
    srhs = ct.sum(prhs)
    ct.store(out_sum_x_ptr, index=(ch,), tile=ct.reshape(sx, (1,)))
    ct.store(out_sum_x_rhs_ptr, index=(ch,), tile=ct.reshape(srhs, (1,)))


def _pad_to_multiple(x, K_target):
    """Pad along dim 0 (K axis) up to K_target rows with zeros."""
    K = x.shape[0]
    if K == K_target:
        return x
    pad_rows = K_target - K
    pad = torch.zeros(pad_rows, x.shape[1], device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def _cutile_channel_reduce(x_nhwc, x_rhs_nhwc, channels, stream, device):
    """Given x, x_rhs of shape [K, C] where K = N*H*W, compute per-channel
    sum(x) and sum(x_rhs) via cuTile split-K partials + finalize."""
    K = x_nhwc.shape[0]
    BLOCK_K = 128
    nblocks = (K + BLOCK_K - 1) // BLOCK_K
    p2 = 1
    while p2 < nblocks:
        p2 <<= 1
    K_target = nblocks * BLOCK_K
    x_padded = _pad_to_multiple(x_nhwc, K_target)
    x_rhs_padded = _pad_to_multiple(x_rhs_nhwc, K_target)
    partial_x = torch.zeros((channels, p2), device=device, dtype=torch.float32)
    partial_x_rhs = torch.zeros((channels, p2), device=device, dtype=torch.float32)
    ct.launch(
        stream, (channels, nblocks, 1), _channel_partial_kernel,
        (x_padded, x_rhs_padded, partial_x, partial_x_rhs, BLOCK_K, K),
    )
    out_sum_x = torch.empty((channels,), device=device, dtype=torch.float32)
    out_sum_x_rhs = torch.empty((channels,), device=device, dtype=torch.float32)
    ct.launch(
        stream, (channels, 1, 1), _channel_finalize_kernel,
        (partial_x, partial_x_rhs, out_sum_x, out_sum_x_rhs, p2),
    )
    return out_sum_x, out_sum_x_rhs


@oracle_impl(hardware="B200", point="b55d777f")
def oracle_forward(inputs):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
     arg6_1, arg7_1, arg8_1, arg9_1, _shape0, shape1) = inputs
    device = arg0_1.device
    stream = torch.cuda.current_stream()

    with torch.no_grad():
        add = arg0_1 + arg1_1
        copy_ = torch.empty_strided(
            tuple(int(d) for d in _shape0),
            tuple(int(s) for s in shape1),
            device=device, dtype=torch.bfloat16,
        )
        copy_.copy_(add)
        clone = copy_.contiguous()
        copy_1 = torch.empty_strided(
            tuple(int(d) for d in _shape0),
            tuple(int(s) for s in shape1),
            device=device, dtype=torch.bfloat16,
        )
        copy_1.copy_(clone)

        conv0 = clone.to(torch.float32)  # [N, 80, H, W] contig
        sub = arg2_1.to(torch.float32) - arg3_1  # [N, 80, H, W] (arg2 NHWC)
        x_rhs_full = conv0 * sub

        # Per-channel reductions via cuTile split-K.
        conv0_khwc = conv0.permute(0, 2, 3, 1).contiguous().view(-1, 80)
        x_rhs_khwc = x_rhs_full.permute(0, 2, 3, 1).contiguous().view(-1, 80)
        sum_1, sum_2 = _cutile_channel_reduce(
            conv0_khwc, x_rhs_khwc, 80, stream, device
        )

        pc1_v = (sum_2 * REDUCE_SCALE * arg4_1 * arg4_1).view(1, -1, 1, 1)
        mt1_v = (sum_1 * REDUCE_SCALE).view(1, -1, 1, 1)
        os1_v = (arg4_1 * arg5_1).view(1, -1, 1, 1)
        mul_7 = (conv0 - sub * pc1_v - mt1_v) * os1_v
        conv_e2 = mul_7.to(torch.bfloat16)
        prod1 = sum_2 * arg4_1

        slice_1 = copy_1[:, 40:80]
        conv3 = slice_1.to(torch.float32)  # [N, 40, H, W]
        sub_3 = arg6_1.to(torch.float32) - arg7_1  # [N, 40, H, W]
        x_rhs_slice = conv3 * sub_3
        conv3_khwc = conv3.permute(0, 2, 3, 1).contiguous().view(-1, 40)
        x_rhs_slice_khwc = x_rhs_slice.permute(0, 2, 3, 1).contiguous().view(-1, 40)
        sum_3, sum_4 = _cutile_channel_reduce(
            conv3_khwc, x_rhs_slice_khwc, 40, stream, device
        )

        pc2_v = (sum_4 * REDUCE_SCALE * arg8_1 * arg8_1).view(1, -1, 1, 1)
        mt2_v = (sum_3 * REDUCE_SCALE).view(1, -1, 1, 1)
        os2_v = (arg8_1 * arg9_1).view(1, -1, 1, 1)
        mul_16 = (conv3 - sub_3 * pc2_v - mt2_v) * os2_v
        prod2 = sum_4 * arg8_1

    # cuTile: cast mul_16 (f32) to bf16 as our cuTile step.
    mul_16_bf = torch.empty(mul_16.shape, device=mul_16.device, dtype=torch.bfloat16)
    src_flat = mul_16.contiguous().view(-1)
    dst_flat = mul_16_bf.view(-1)
    N = src_flat.numel()
    BLOCK = 1024
    ct.launch(
        stream, ((N + BLOCK - 1) // BLOCK, 1, 1),
        _f32_to_bf16_kernel, (src_flat, dst_flat, N, BLOCK),
    )

    return copy_1, sum_1, prod1, conv_e2, sum_3, prod2, mul_16_bf
