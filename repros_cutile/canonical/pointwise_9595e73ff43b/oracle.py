"""cuTile port of pointwise_9595e73ff43b (SCHEDULER_FUSION): MobileNetV2 bf16
BN-inference affine + ReLU6 for channels-last activations.

Layout: input strides are [C*H*W, 1, C*W, C] (NHWC). We flatten to (N*H*W, C)
and tile the (rows, cols) domain. Non-power-of-2 C is handled with
padding_mode.ZERO. The intermediate bf16 rounding before clamp is preserved
by casting to bf16 and back to f32 (cuTile default RTNE).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu6_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [ROWS, C_ROUNDED_STRIDE]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [ROWS, C_ROUNDED_STRIDE]
    C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_block = ct.bid(0)
    col_block = ct.bid(1)

    # Load a tile [BLOCK_R, BLOCK_C] with ZERO padding for out-of-bounds C.
    x = ct.load(
        x_ptr,
        index=(row_block, col_block),
        shape=(BLOCK_R, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    # For each column of this tile, look up the per-channel scalars.
    col_offsets = ct.arange(BLOCK_C, dtype=ct.int32) + col_block * BLOCK_C
    col_mask = col_offsets < C
    # Clamp to zero when OOB (won't be stored anyway).
    safe_col = ct.where(col_mask, col_offsets, ct.zeros((BLOCK_C,), dtype=ct.int32))

    mean_bf = ct.gather(mean_ptr, safe_col)
    var_bf = ct.gather(var_ptr, safe_col)
    weight_bf = ct.gather(weight_ptr, safe_col)
    bias_bf = ct.gather(bias_ptr, safe_col)
    mean_f = ct.reshape(ct.astype(mean_bf, ct.float32), (1, BLOCK_C))
    var_f = ct.reshape(ct.astype(var_bf, ct.float32), (1, BLOCK_C))
    weight_f = ct.reshape(ct.astype(weight_bf, ct.float32), (1, BLOCK_C))
    bias_f = ct.reshape(ct.astype(bias_bf, ct.float32), (1, BLOCK_C))

    denom = ct.sqrt(var_f + 1.0e-5)
    invstd = 1.0 / denom
    centered = x_f - mean_f
    normalized = centered * invstd
    scaled = normalized * weight_f
    affine = scaled + bias_f
    # Intermediate bf16 rounding before clamp (cuTile default RTNE).
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    # NaN-preserving clamp:
    # torch.clamp_min(NaN, 0.0) => NaN; torch.clamp_max(NaN, 6.0) => NaN.
    zero_tile = ct.full((BLOCK_R, BLOCK_C), 0.0, dtype=ct.float32)
    six_tile = ct.full((BLOCK_R, BLOCK_C), 6.0, dtype=ct.float32)
    is_nan = rounded_f32 != rounded_f32
    max0 = ct.where(rounded_f32 > 0.0, rounded_f32, zero_tile)
    clamp_min_val = ct.where(is_nan, rounded_f32, max0)
    is_nan2 = clamp_min_val != clamp_min_val
    min6 = ct.where(clamp_min_val < 6.0, clamp_min_val, six_tile)
    clamp_max_val = ct.where(is_nan2, clamp_min_val, min6)
    out_bf16 = ct.astype(clamp_max_val, ct.bfloat16)
    ct.store(out_ptr, index=(row_block, col_block), tile=out_bf16)


@ct.kernel
def _bn_relu6_nchw_kernel(
    mean_ptr,     # bf16 [C]
    x_ptr,        # bf16 [N*C, HW]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [N*C, HW]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_R: ct.Constant[int],  # =1
    BLOCK_HW: ct.Constant[int],
):
    row = ct.bid(0)   # index in N*C
    col_block = ct.bid(1)

    x = ct.load(
        x_ptr,
        index=(row, col_block),
        shape=(BLOCK_R, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(x, ct.float32)

    # Compute channel from row index: c = row % C.
    # Since BLOCK_R=1, we can use scalar-based load via ct.gather with a 1-elem index.
    row_ids = ct.arange(BLOCK_R, dtype=ct.int32) + row
    channel = row_ids - (row_ids // C) * C

    mean_bf = ct.gather(mean_ptr, channel)
    var_bf = ct.gather(var_ptr, channel)
    weight_bf = ct.gather(weight_ptr, channel)
    bias_bf = ct.gather(bias_ptr, channel)
    mean_f = ct.reshape(ct.astype(mean_bf, ct.float32), (BLOCK_R, 1))
    var_f = ct.reshape(ct.astype(var_bf, ct.float32), (BLOCK_R, 1))
    weight_f = ct.reshape(ct.astype(weight_bf, ct.float32), (BLOCK_R, 1))
    bias_f = ct.reshape(ct.astype(bias_bf, ct.float32), (BLOCK_R, 1))

    denom = ct.sqrt(var_f + 1.0e-5)
    invstd = 1.0 / denom
    centered = x_f - mean_f
    normalized = centered * invstd
    scaled = normalized * weight_f
    affine = scaled + bias_f
    rounded_bf16 = ct.astype(affine, ct.bfloat16)
    rounded_f32 = ct.astype(rounded_bf16, ct.float32)
    zero_tile = ct.full((BLOCK_R, BLOCK_HW), 0.0, dtype=ct.float32)
    six_tile = ct.full((BLOCK_R, BLOCK_HW), 6.0, dtype=ct.float32)
    is_nan = rounded_f32 != rounded_f32
    max0 = ct.where(rounded_f32 > 0.0, rounded_f32, zero_tile)
    clamp_min_val = ct.where(is_nan, rounded_f32, max0)
    is_nan2 = clamp_min_val != clamp_min_val
    min6 = ct.where(clamp_min_val < 6.0, clamp_min_val, six_tile)
    clamp_max_val = ct.where(is_nan2, clamp_min_val, min6)
    out_bf16 = ct.astype(clamp_max_val, ct.bfloat16)
    ct.store(out_ptr, index=(row, col_block), tile=out_bf16)


def _next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="8a10670c")
@oracle_impl(hardware="B200", point="82cef4c9")
@oracle_impl(hardware="B200", point="2cf7960f")
@oracle_impl(hardware="B200", point="54640677")
@oracle_impl(hardware="B200", point="1ee4ce98")
@oracle_impl(hardware="B200", point="ba8126d9")
@oracle_impl(hardware="B200", point="ebe204a7")
@oracle_impl(hardware="B200", point="51719261")
@oracle_impl(hardware="B200", point="d53a7e50")
@oracle_impl(hardware="B200", point="2e1844e5")
@oracle_impl(hardware="B200", point="9c97edfa")
@oracle_impl(hardware="B200", point="3bdac97c")
@oracle_impl(hardware="B200", point="66878456")
@oracle_impl(hardware="B200", point="0598891b")
@oracle_impl(hardware="B200", point="91334d30")
@oracle_impl(hardware="B200", point="e5f12ec5")
@oracle_impl(hardware="B200", point="94494857")
@oracle_impl(hardware="B200", point="1cc752a4")
@oracle_impl(hardware="B200", point="92d91bac")
@oracle_impl(hardware="B200", point="3de61eda")
@oracle_impl(hardware="B200", point="8dd8c35d")
@oracle_impl(hardware="B200", point="522f10e2")
def oracle_forward(inputs):
    mean, x, var, weight, bias = inputs
    n, c, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    hw = h * w
    rows = n * hw

    # Allocate output with matching strides.
    out = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()), device=x.device, dtype=torch.bfloat16
    )

    channels_last = x.stride(1) == 1
    if channels_last:
        # Reshape as (N*H*W, C) - already contiguous memory-wise in NHWC layout.
        x_2d = x.permute(0, 2, 3, 1).reshape(rows, c)
        out_2d = out.permute(0, 2, 3, 1).reshape(rows, c)
    else:
        # NCHW contiguous: reshape to (N*C, H*W) and tile over rows=N*C, cols=H*W.
        # Channel for the r-th row is r % C. Use a different kernel path.
        x_2d_nchw = x.view(n * c, hw)
        out_2d_nchw = out.view(n * c, hw)
        BLOCK_R = 1
        BLOCK_HW = _next_power_of_2(hw)
        if BLOCK_HW > 4096:
            BLOCK_HW = 4096
        grid = (n * c, (hw + BLOCK_HW - 1) // BLOCK_HW, 1)
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            grid,
            _bn_relu6_nchw_kernel,
            (mean, x_2d_nchw, var, weight, bias, out_2d_nchw, c, hw, BLOCK_R, BLOCK_HW),
        )
        return out

    # Round-up tile dims for cuTile power-of-2 requirement.
    BLOCK_R = 8
    BLOCK_C = _next_power_of_2(c)
    if BLOCK_C > 1024:
        BLOCK_C = 1024

    grid = ((rows + BLOCK_R - 1) // BLOCK_R,
            (c + BLOCK_C - 1) // BLOCK_C, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _bn_relu6_kernel,
        (mean, x_2d, var, weight, bias, out_2d, c, BLOCK_R, BLOCK_C),
    )
    return out
