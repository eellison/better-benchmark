"""cuTile port of mean_ea7081208754: GhostNet BN + ReLU + channel-cat + spatial mean.

Two-kernel split over channels-last NHWC-flat views:
  1. Left-copy kernel: reads left[N,C,H,W] channels-last (viewed as
     NHWC-flat (N, HW, C)). Writes to a scratch (N, BLOCK_HW, C) buffer
     and computes spatial mean per (n, c).
  2. BN+ReLU kernel: reads conv[N,C,H,W] channels-last, applies BN affine
     (fp32 arithmetic, eps=1e-5) followed by bf16-rounded ReLU. Writes to
     a second scratch buffer and computes spatial mean per (n, c).

Post-kernel: narrow both scratches to HW along dim 1 and copy into the
final channels-last cat[N,2C,H,W] via its NHWC-flat view (zero-copy).
Mean slots for the two halves are concatenated into mean[N,2C,1,1].
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


BN_EPS = 1.0e-5


@ct.kernel
def _left_kernel(
    left_ptr,        # bf16 (N, HW, C) NHWC-flat view (contiguous)
    scratch_ptr,     # bf16 (N, BLOCK_HW, C) padded scratch
    mean_ptr,        # bf16 (N, C) left-half spatial mean
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    tile = ct.load(
        left_ptr,
        index=(n, 0, c),
        shape=(1, BLOCK_HW, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(scratch_ptr, index=(n, 0, c), tile=tile)

    tile_f = ct.astype(tile, ct.float32)
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = ct.reshape(hw_idx < HW_, (1, BLOCK_HW, 1))
    zero3d = ct.zeros(shape=(1, BLOCK_HW, 1), dtype=ct.float32)
    masked = ct.where(valid, tile_f, zero3d)
    total = ct.sum(masked)
    mean_val = total * (1.0 / HW_)
    mean_tile = ct.reshape(ct.full((1,), mean_val, dtype=ct.float32), (1, 1))
    ct.store(mean_ptr, index=(n, c), tile=ct.astype(mean_tile, ct.bfloat16))


@ct.kernel
def _bn_relu_kernel(
    conv_ptr,           # bf16 (N, HW, C)
    running_mean_ptr,   # bf16 (C,)
    running_var_ptr,    # bf16 (C,)
    weight_ptr,         # bf16 (C,)
    bias_ptr,           # bf16 (C,)
    scratch_ptr,        # bf16 (N, BLOCK_HW, C)
    mean_ptr,           # bf16 (N, C)
    HW_: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)
    conv_bf16 = ct.load(
        conv_ptr,
        index=(n, 0, c),
        shape=(1, BLOCK_HW, 1),
        padding_mode=ct.PaddingMode.ZERO,
    )
    conv_f = ct.astype(conv_bf16, ct.float32)
    rm_f = ct.astype(ct.load(running_mean_ptr, index=(c,), shape=(1,)), ct.float32)
    rv_f = ct.astype(ct.load(running_var_ptr, index=(c,), shape=(1,)), ct.float32)
    w_f = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    b_f = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    invstd = 1.0 / ct.sqrt(rv_f + BN_EPS)
    centered = conv_f - rm_f
    normalized = centered * invstd
    affine = normalized * w_f + b_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.full(shape=(1, BLOCK_HW, 1), fill_value=0.0, dtype=ct.bfloat16)
    relu_bf16 = ct.where(affine_bf16 < zero_bf16, zero_bf16, affine_bf16)
    ct.store(scratch_ptr, index=(n, 0, c), tile=relu_bf16)

    relu_f = ct.astype(relu_bf16, ct.float32)
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid = ct.reshape(hw_idx < HW_, (1, BLOCK_HW, 1))
    zero3d = ct.zeros(shape=(1, BLOCK_HW, 1), dtype=ct.float32)
    masked = ct.where(valid, relu_f, zero3d)
    total = ct.sum(masked)
    mean_val = total * (1.0 / HW_)
    mean_tile = ct.reshape(ct.full((1,), mean_val, dtype=ct.float32), (1, 1))
    ct.store(mean_ptr, index=(n, c), tile=ct.astype(mean_tile, ct.bfloat16))


@oracle_impl(hardware="B200", point="5339df69", BLOCK_ROWS=8, BLOCK_HW=64)
@oracle_impl(hardware="B200", point="25326742", BLOCK_ROWS=4, BLOCK_HW=256)
@oracle_impl(hardware="B200", point="7258878f", BLOCK_ROWS=4, BLOCK_HW=256)
@oracle_impl(hardware="B200", point="a5185a17", BLOCK_ROWS=1, BLOCK_HW=1024)
def oracle_forward(inputs, *, BLOCK_ROWS: int, BLOCK_HW: int):
    del BLOCK_ROWS  # unused in cuTile port; Triton-only knob
    running_mean, conv, running_var, weight, bias, left = inputs
    N = int(conv.shape[0])
    C = int(conv.shape[1])
    H = int(conv.shape[2])
    W = int(conv.shape[3])
    HW = H * W
    OUT_C = 2 * C

    # Channels-last -> NHWC-flat (zero-copy views)
    conv_nhwc = conv.permute(0, 2, 3, 1).reshape(N, HW, C)
    left_nhwc = left.permute(0, 2, 3, 1).reshape(N, HW, C)

    # Final cat: channels-last (N, 2C, H, W)
    cat = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, 1, OUT_C * W, OUT_C),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    cat_nhwc = cat.permute(0, 2, 3, 1).reshape(N, HW, OUT_C)

    # Scratch buffers padded along dim 1 to BLOCK_HW
    scratch_left = torch.empty(
        (N, BLOCK_HW, C), device=conv.device, dtype=torch.bfloat16,
    )
    scratch_bn = torch.empty(
        (N, BLOCK_HW, C), device=conv.device, dtype=torch.bfloat16,
    )
    # Two contiguous half-buffers for mean; concatenated at the end.
    mean_left = torch.empty(
        (N, C), device=conv.device, dtype=torch.bfloat16,
    )
    mean_bn = torch.empty(
        (N, C), device=conv.device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, C, 1),
        _left_kernel,
        (left_nhwc, scratch_left, mean_left, HW, BLOCK_HW),
    )
    ct.launch(
        stream,
        (N, C, 1),
        _bn_relu_kernel,
        (
            conv_nhwc,
            running_mean,
            running_var,
            weight,
            bias,
            scratch_bn,
            mean_bn,
            HW,
            BLOCK_HW,
        ),
    )

    # Copy scratches into cat_nhwc, narrowing to HW along dim 1.
    cat_nhwc[:, :, :C].copy_(scratch_left[:, :HW, :])
    cat_nhwc[:, :, C:].copy_(scratch_bn[:, :HW, :])

    # Concat mean halves into (N, 2C, 1, 1)
    mean_out = torch.empty_strided(
        (N, OUT_C, 1, 1),
        (OUT_C, 1, 1, 1),
        device=conv.device,
        dtype=torch.bfloat16,
    )
    mean_out_flat = mean_out.view(N, OUT_C)
    mean_out_flat[:, :C].copy_(mean_left)
    mean_out_flat[:, C:].copy_(mean_bn)

    return cat, mean_out
