"""cuTile port of sum_894e623ad263 (SCATTER_REDUCE): NFNet bf16
scatter/expand + sigmoid-gradient + per-channel sum.

Two kernels:
  1. For each (n, c_block) tile, expand the pooled scale over H*W, compute
     sigmoid-gradient in fp32, write bf16 output, and write per-(n, c)
     partial spatial sums.
  2. For each c_block, reduce the [N, C] partial matrix down to a bf16-rounded
     f32 per-channel sum vector.

C=2304 is not power-of-2; we tile channels with BLOCK_C=32 or 64 and use
ct.gather / ct.scatter to handle irregular boundaries.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 2304
H = 7
W = 7
HW = H * W


@ct.kernel
def _materialize_spatial_sum_kernel(
    scale_ptr,      # bf16 [N, C]
    act_ptr,        # bf16 flat [N*C*HW] channels-last (NHWC layout)
    out_ptr,        # bf16 flat [N*C*HW] contiguous NCHW
    partial_ptr,    # f32 [N, C]
    C_: ct.Constant[int],
    H_: ct.Constant[int],
    W_: ct.Constant[int],
    HW_: ct.Constant[int],
    ACT_S0: ct.Constant[int],
    ACT_S1: ct.Constant[int],
    ACT_S2: ct.Constant[int],
    ACT_S3: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    channel_block = ct.bid(1)

    channels = ct.arange(BLOCK_C, dtype=ct.int64) + channel_block * BLOCK_C
    hw = ct.arange(BLOCK_HW, dtype=ct.int64)
    channels_2d = ct.reshape(channels, (BLOCK_C, 1))
    hw_2d = ct.reshape(hw, (1, BLOCK_HW))
    channels_broad = ct.broadcast_to(channels_2d, (BLOCK_C, BLOCK_HW))
    hw_broad = ct.broadcast_to(hw_2d, (BLOCK_C, BLOCK_HW))

    ch_mask = channels_broad < C_
    hw_mask = hw_broad < HW_
    mask = ch_mask & hw_mask

    # Load scale [BLOCK_C] via gather over (n, channels).
    ch_1d_mask = channels < C_
    zero64 = ct.zeros((BLOCK_C,), dtype=ct.int64)
    safe_ch = ct.where(ch_1d_mask, channels, zero64)
    scale_offsets = safe_ch + n * C_
    scale_bf = ct.gather(scale_ptr, scale_offsets)
    scale_f = ct.astype(scale_bf, ct.float32) * (1.0 / 49.0)
    scale_bf16 = ct.astype(scale_f, ct.bfloat16)
    scale_f = ct.astype(scale_bf16, ct.float32)
    scale_2d = ct.reshape(scale_f, (BLOCK_C, 1))
    scale_broad = ct.broadcast_to(scale_2d, (BLOCK_C, BLOCK_HW))

    # Compute activation offsets for the channels-last strided input.
    h = hw_broad // W_
    w = hw_broad - h * W_
    zero64_2d = ct.zeros((BLOCK_C, BLOCK_HW), dtype=ct.int64)
    safe_channels = ct.where(mask, channels_broad, zero64_2d)
    safe_hw = ct.where(mask, hw_broad, zero64_2d)
    safe_h = ct.where(mask, h, zero64_2d)
    safe_w = ct.where(mask, w, zero64_2d)
    act_offsets = (
        n * ACT_S0
        + safe_channels * ACT_S1
        + safe_h * ACT_S2
        + safe_w * ACT_S3
    )
    act_bf = ct.gather(act_ptr, act_offsets)
    act = ct.astype(act_bf, ct.float32)
    # Sigmoid: 1/(1+exp(-x))
    sigmoid = 1.0 / (1.0 + ct.exp(-act))
    value = scale_broad * sigmoid * (act * (1.0 - sigmoid) + 1.0)
    value_bf16 = ct.astype(value, ct.bfloat16)

    # Out is contiguous NCHW: offset = n*C*HW + c*HW + hw
    out_offsets = n * (C_ * HW_) + safe_channels * HW_ + safe_hw
    ct.scatter(out_ptr, out_offsets, value_bf16, mask=mask)

    # Compute partial spatial sum per (n, c) over HW.
    contrib = ct.where(mask,
                        ct.astype(value_bf16, ct.float32),
                        ct.full((BLOCK_C, BLOCK_HW), 0.0, dtype=ct.float32))
    spatial_sum = ct.sum(contrib, axis=1)  # [BLOCK_C]
    partial_offsets = n * C_ + channels
    ct.scatter(partial_ptr, partial_offsets, spatial_sum, mask=ch_1d_mask)


@ct.kernel
def _final_channel_sum_kernel(
    partial_ptr,     # f32 [N, C]
    sum_ptr,         # f32 [C]
    N_: ct.Constant[int],
    C_: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    channel_block = ct.bid(0)
    channels = ct.arange(BLOCK_C, dtype=ct.int64) + channel_block * BLOCK_C
    rows = ct.arange(BLOCK_N, dtype=ct.int64)
    channels_2d = ct.reshape(channels, (1, BLOCK_C))
    rows_2d = ct.reshape(rows, (BLOCK_N, 1))
    channels_broad = ct.broadcast_to(channels_2d, (BLOCK_N, BLOCK_C))
    rows_broad = ct.broadcast_to(rows_2d, (BLOCK_N, BLOCK_C))
    ch_mask = channels_broad < C_
    row_mask = rows_broad < N_
    mask = ch_mask & row_mask

    zero64 = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.int64)
    safe_ch = ct.where(mask, channels_broad, zero64)
    safe_row = ct.where(mask, rows_broad, zero64)
    flat_off = safe_row * C_ + safe_ch
    values = ct.gather(partial_ptr, flat_off)
    values = ct.where(mask, values, ct.full((BLOCK_N, BLOCK_C), 0.0, dtype=ct.float32))
    total = ct.sum(values, axis=0)  # [BLOCK_C]
    rounded_bf = ct.astype(total, ct.bfloat16)
    rounded_f = ct.astype(rounded_bf, ct.float32)
    ch_1d_mask = channels < C_
    zero64_1 = ct.zeros((BLOCK_C,), dtype=ct.int64)
    safe_ch_1 = ct.where(ch_1d_mask, channels, zero64_1)
    ct.scatter(sum_ptr, safe_ch_1, rounded_f, mask=ch_1d_mask)


@oracle_impl(hardware="B200", point="943c9ed8", BLOCK_HW=64, BLOCK_C=64, FINAL_BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_HW: int, BLOCK_C: int, FINAL_BLOCK_C: int):
    scale_src, activation, _sp0, _sp1, _sp2, _sp3, _sp4, _sp5, _sp6 = inputs

    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=scale_src.device,
        dtype=torch.bfloat16,
    )
    partial = torch.empty_strided((N, C), (C, 1), device=scale_src.device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=scale_src.device, dtype=torch.float32)

    # Flatten. scale_src is bf16[128, 2304] contiguous.
    scale_flat = scale_src.reshape(-1)
    out_flat = out.reshape(-1)
    partial_flat = partial.reshape(-1)
    sum_flat = sum_out.reshape(-1)

    # activation is NHWC-strided; flatten via storage. Use as_strided to hold
    # its underlying buffer as flat.
    act_flat_size = int(activation.storage().nbytes() // activation.element_size())
    act_flat = activation.as_strided((act_flat_size,), (1,))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, (C + BLOCK_C - 1) // BLOCK_C, 1),
        _materialize_spatial_sum_kernel,
        (scale_flat, act_flat, out_flat, partial_flat,
         C, H, W, HW,
         activation.stride(0), activation.stride(1),
         activation.stride(2), activation.stride(3),
         BLOCK_C, BLOCK_HW),
    )
    # For final reduction: BLOCK_N must be power of 2 >= N=128.
    ct.launch(
        stream,
        ((C + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _final_channel_sum_kernel,
        (partial_flat, sum_flat, N, C, N, FINAL_BLOCK_C),
    )
    return out, sum_out
