"""cuTile port of sum_sum_sum_46c77e1ef34a: Electra LayerNorm-backward tail.

Grouped: ROWS_PER_GROUP rows per block, iterating BLOCK_R rows at a time and
accumulating column partials. Second kernel column-reduces to the three [C]
outputs.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROW_FACTOR = 256.0
INV_ROW_FACTOR = 0.00390625
DROP_SCALE = 1.1111111111111112


@ct.kernel
def _store_partials_kernel(
    bf16_residual_ptr,  # bf16 [rows, C]
    f32_residual_ptr,   # f32  [rows, C]
    weight_ptr,         # f32  [C]
    norm_source_ptr,    # bf16 [rows, C]
    mean_ptr,           # f32  [rows]
    scale_ptr,          # f32  [rows]
    mask_ptr,           # b8   [rows, C]
    grad_bf16_out_ptr,  # bf16 [rows, C]
    side_bf16_out_ptr,  # bf16 [rows, C]
    partials_ptr,       # f32  [NUM_GROUPS, 3, C]
    CHANNELS: ct.Constant[int],
    ROW_FACTOR_: ct.Constant[float],
    INV_ROW_FACTOR_: ct.Constant[float],
    DROP_SCALE_: ct.Constant[float],
    ROWS_PER_GROUP: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    group = ct.bid(0)

    weight = ct.load(weight_ptr, index=(0,), shape=(CHANNELS,))
    weight_2d = ct.reshape(weight, (1, CHANNELS))

    acc_x_rhs = ct.zeros((CHANNELS,), dtype=ct.float32)
    acc_x = ct.zeros((CHANNELS,), dtype=ct.float32)
    acc_side = ct.zeros((CHANNELS,), dtype=ct.float32)

    n_steps = ROWS_PER_GROUP // BLOCK_R
    for step in range(n_steps):
        row_block = group * n_steps + step
        # Load BLOCK_R rows of x (from bf16 and f32 residuals), norm_source, mask
        x_bf = ct.load(bf16_residual_ptr, index=(row_block, 0),
                       shape=(BLOCK_R, CHANNELS))
        x_res = ct.load(f32_residual_ptr, index=(row_block, 0),
                        shape=(BLOCK_R, CHANNELS))
        x = ct.astype(x_bf, ct.float32) + x_res
        norm_src = ct.astype(
            ct.load(norm_source_ptr, index=(row_block, 0),
                    shape=(BLOCK_R, CHANNELS)),
            ct.float32)
        mean_v = ct.load(mean_ptr, index=(row_block,), shape=(BLOCK_R,))
        scale_v = ct.load(scale_ptr, index=(row_block,), shape=(BLOCK_R,))
        mean_2d = ct.reshape(mean_v, (BLOCK_R, 1))
        scale_2d = ct.reshape(scale_v, (BLOCK_R, 1))
        keep = ct.astype(
            ct.load(mask_ptr, index=(row_block, 0),
                    shape=(BLOCK_R, CHANNELS)),
            ct.float32)

        rhs = (norm_src - mean_2d) * scale_2d
        weighted = x * weight_2d
        row_sum = ct.sum(weighted, axis=1, keepdims=True)
        row_dot = ct.sum(weighted * rhs, axis=1, keepdims=True)
        centered = weighted * ROW_FACTOR_ - row_sum
        centered = centered - rhs * row_dot
        row_scale = scale_2d * INV_ROW_FACTOR_
        grad = row_scale * centered
        grad_bf16 = ct.astype(grad, ct.bfloat16)
        ct.store(grad_bf16_out_ptr, index=(row_block, 0), tile=grad_bf16)

        keep_scale_bf = ct.astype(keep * DROP_SCALE_, ct.bfloat16)
        grad_f = ct.astype(grad_bf16, ct.float32)
        keep_scale_f = ct.astype(keep_scale_bf, ct.float32)
        side_bf16 = ct.astype(grad_f * keep_scale_f, ct.bfloat16)
        ct.store(side_bf16_out_ptr, index=(row_block, 0), tile=side_bf16)

        # Column-partial accumulation
        acc_x_rhs = acc_x_rhs + ct.sum(x * rhs, axis=0)
        acc_x = acc_x + ct.sum(x, axis=0)
        acc_side = acc_side + ct.sum(ct.astype(side_bf16, ct.float32), axis=0)

    # Store partials [group, 0..2, :] as 2D tiles
    ct.store(partials_ptr, index=(group, 0, 0),
             tile=ct.reshape(acc_x_rhs, (1, 1, CHANNELS)))
    ct.store(partials_ptr, index=(group, 1, 0),
             tile=ct.reshape(acc_x, (1, 1, CHANNELS)))
    ct.store(partials_ptr, index=(group, 2, 0),
             tile=ct.reshape(acc_side, (1, 1, CHANNELS)))


@ct.kernel
def _finalize_partials_kernel(
    partials_ptr,       # f32 [NUM_GROUPS, 3, C]
    out_x_rhs_ptr,      # f32 [C]
    out_x_ptr,          # f32 [C]
    out_side_sum_ptr,   # f32 [C]
    NUM_GROUPS: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    col_block = ct.bid(0)
    xr = ct.load(partials_ptr, index=(0, 0, col_block),
                 shape=(NUM_GROUPS, 1, BLOCK_C))
    x_ = ct.load(partials_ptr, index=(0, 1, col_block),
                 shape=(NUM_GROUPS, 1, BLOCK_C))
    sd = ct.load(partials_ptr, index=(0, 2, col_block),
                 shape=(NUM_GROUPS, 1, BLOCK_C))

    xr_sum = ct.reshape(ct.sum(xr, axis=0), (BLOCK_C,))
    x_sum = ct.reshape(ct.sum(x_, axis=0), (BLOCK_C,))
    sd_sum = ct.reshape(ct.sum(sd, axis=0), (BLOCK_C,))
    # bf16-round for side sum
    sd_bf = ct.astype(sd_sum, ct.bfloat16)
    sd_f = ct.astype(sd_bf, ct.float32)

    ct.store(out_x_rhs_ptr, index=(col_block,), tile=xr_sum)
    ct.store(out_x_ptr, index=(col_block,), tile=x_sum)
    ct.store(out_side_sum_ptr, index=(col_block,), tile=sd_f)


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@oracle_impl(hardware="B200", point="b0f8aad0",
             ROWS_PER_GROUP=128, BLOCK_R=32, FINAL_BLOCK_C=8)
def oracle_forward(inputs, *, ROWS_PER_GROUP, BLOCK_R, FINAL_BLOCK_C):
    (
        arg0_1,  # bf16 [32768, 256]
        arg1_1,  # f32  [64, 512, 256]
        arg2_1,  # f32  [256]
        arg3_1,  # bf16 [64, 512, 256]
        arg4_1,  # f32  [64, 512, 1]
        arg5_1,  # f32  [64, 512, 1]
        arg6_1,  # b8   [64, 512, 256]
        _sp0, _sp1, _sp2,
    ) = inputs

    device = arg1_1.device
    batch = int(arg1_1.shape[0])
    seq = int(arg1_1.shape[1])
    channels = int(arg1_1.shape[2])
    rows = batch * seq

    bf16_residual_2d = arg0_1
    f32_residual_2d = arg1_1.view(rows, channels)
    norm_source_2d = arg3_1.view(rows, channels)
    mean_1d = arg4_1.view(rows)
    scale_1d = arg5_1.view(rows)
    mask_2d = arg6_1.view(rows, channels)

    grad_bf16 = torch.empty_strided(
        (batch, seq, channels),
        (seq * channels, channels, 1),
        device=device, dtype=torch.bfloat16,
    )
    side_bf16 = torch.empty((rows, channels), device=device, dtype=torch.bfloat16)
    grad_bf16_2d = grad_bf16.view(rows, channels)

    num_groups = (rows + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    partials = torch.zeros((num_groups, 3, channels), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (num_groups, 1, 1),
        _store_partials_kernel,
        (
            bf16_residual_2d, f32_residual_2d, arg2_1, norm_source_2d,
            mean_1d, scale_1d, mask_2d, grad_bf16_2d, side_bf16, partials,
            channels, ROW_FACTOR, INV_ROW_FACTOR, DROP_SCALE,
            ROWS_PER_GROUP, BLOCK_R,
        ),
    )

    out_x_rhs = torch.empty((channels,), device=device, dtype=torch.float32)
    out_x = torch.empty((channels,), device=device, dtype=torch.float32)
    out_side_sum = torch.empty((channels,), device=device, dtype=torch.float32)

    # For the finalize, NUM_GROUPS must be a power of 2 tile shape. Round up:
    num_groups_pow2 = _next_pow2(num_groups)
    if num_groups_pow2 != num_groups:
        partials_padded = torch.zeros((num_groups_pow2, 3, channels), device=device,
                                       dtype=torch.float32)
        partials_padded[:num_groups] = partials
        partials = partials_padded

    ct.launch(
        stream,
        (ct.cdiv(channels, FINAL_BLOCK_C), 1, 1),
        _finalize_partials_kernel,
        (partials, out_x_rhs, out_x, out_side_sum,
         num_groups_pow2, channels, FINAL_BLOCK_C),
    )

    return out_x_rhs, out_x, grad_bf16, side_bf16, side_bf16.permute(1, 0), out_side_sum
