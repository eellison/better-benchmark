"""cuTile port of var_mean_764425449966: SigLIP ViT patch-position LayerNorm.

Reads directly from the bf16 [batch, hidden, 16, 16] channels-contiguous
input via ct.gather (avoids the giant f32 patch materialization in torch).
Padded BLOCK_H=1024 for hidden=768 with masking.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _patch_position_layernorm_kernel(
    patch_ptr,       # bf16 flat NCHW-ish [batch * hidden * tokens]
    pos_ptr,         # f32 flat [tokens * hidden]
    weight_ptr,      # f32 [hidden]
    bias_ptr,        # f32 [hidden]
    add_ptr,         # f32 [rows, hidden]
    mean_ptr,        # f32 [rows]
    rsqrt_ptr,       # f32 [rows]
    final_bf16_ptr,  # bf16 [rows, hidden]
    ROWS: ct.Constant[int],
    TOKENS: ct.Constant[int],
    HIDDEN_: ct.Constant[int],
    PATCH_STRIDE_B: ct.Constant[int],
    PATCH_STRIDE_T: ct.Constant[int],
    PATCH_STRIDE_H: ct.Constant[int],
    POS_STRIDE_T: ct.Constant[int],
    POS_STRIDE_H: ct.Constant[int],
    BLOCK_H_C: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)
    batch = row // TOKENS
    token = row - batch * TOKENS
    cols = ct.arange(BLOCK_H_C, dtype=ct.int32)
    col_mask = cols < HIDDEN_

    patch_off = batch * PATCH_STRIDE_B + token * PATCH_STRIDE_T + cols * PATCH_STRIDE_H
    pos_off = token * POS_STRIDE_T + cols * POS_STRIDE_H
    patch = ct.astype(ct.gather(patch_ptr, (patch_off,), mask=col_mask), ct.float32)
    pos = ct.gather(pos_ptr, (pos_off,), mask=col_mask)
    x = patch + pos

    zero_f = ct.zeros((BLOCK_H_C,), dtype=ct.float32)
    x_masked = ct.where(col_mask, x, zero_f)

    row_2d = ct.full((BLOCK_H_C,), row, dtype=ct.int32)
    add_off = row_2d * HIDDEN_ + cols
    ct.scatter(add_ptr, (add_off,), x_masked, mask=col_mask)

    mean = ct.sum(x_masked) * INV_HIDDEN
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * INV_HIDDEN
    invstd = ct.rsqrt(variance + EPS)

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    a = centered * invstd * weight + bias
    a_bf = ct.astype(a, ct.bfloat16)
    ct.scatter(final_bf16_ptr, (add_off,), a_bf, mask=col_mask)


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="3006dd3d")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = _as_shape(shape0)
    final_shape = _as_shape(shape1)
    batch = int(view_shape[0])
    hidden = int(view_shape[1])
    tokens = int(view_shape[2])
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)
    stat_shape = (batch, tokens, 1)

    add_out = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    mean_out = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    rsqrt_out = torch.empty_strided(
        stat_shape, _contiguous_stride(stat_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape, _contiguous_stride(final_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )

    # patch is [batch, hidden, tokens] (from view of [batch, hidden, 16, 16]).
    patch_3d = arg0_1.view(batch, hidden, tokens)
    p_s_b = int(patch_3d.stride(0))
    p_s_h = int(patch_3d.stride(1))
    p_s_t = int(patch_3d.stride(2))
    patch_flat = patch_3d.as_strided((batch * hidden * tokens,), (1,))
    pos_flat = arg1_1.reshape(-1)
    pos_2d = arg1_1.view(tokens, hidden)
    pos_s_t = int(pos_2d.stride(0))
    pos_s_h = int(pos_2d.stride(1))

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _patch_position_layernorm_kernel,
        (
            patch_flat, pos_flat, arg2_1, arg3_1,
            add_out.view(-1),
            mean_out.view(rows),
            rsqrt_out.view(rows),
            final_bf16.view(-1),
            rows, tokens, hidden,
            p_s_b, p_s_t, p_s_h,
            pos_s_t, pos_s_h,
            BLOCK_H, 1.0 / hidden,
        ),
    )
    return add_out, mean_out, rsqrt_out, final_bf16
