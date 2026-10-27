"""cuTile port of var_mean_499e4b26c010: ViT class+patch cat + LayerNorm.

Single cuTile kernel matching the Triton `_vit_patch_layernorm_kernel`:
  - Gather class-token / patch-token per row (token 0 -> class, else patch).
  - Add positional embedding, round to bf16, store add_out (scatter with mask).
  - Row-wise fp32 var_mean (via ct.sum), eps=1e-6 rsqrt, bf16 affine, store.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _vit_patch_layernorm_kernel(
    class_token_ptr,  # bf16 [HIDDEN]
    conv_flat_ptr,    # bf16 [batch, patches, hidden]
    pos_flat_ptr,     # bf16 [tokens, HIDDEN]
    weight_ptr,       # bf16 [HIDDEN]
    bias_ptr,         # bf16 [HIDDEN]
    add_out_ptr,      # bf16 flat [rows * HIDDEN]
    norm_out_ptr,     # bf16 flat [rows * HIDDEN]
    ROWS: ct.Constant[int],
    TOKENS: ct.Constant[int],
    HIDDEN: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    rows_start = row_tile * BLOCK_R
    row_ids = rows_start + ct.arange(BLOCK_R, dtype=ct.int32)
    row_lim = ct.full(shape=(BLOCK_R,), fill_value=ROWS, dtype=ct.int32)
    row_valid = row_ids < row_lim
    zeros_i = ct.full(shape=(BLOCK_R,), fill_value=0, dtype=ct.int32)
    safe_rows = ct.where(row_valid, row_ids, zeros_i)

    tokens_vec = ct.full(shape=(BLOCK_R,), fill_value=TOKENS, dtype=ct.int32)
    token = safe_rows % tokens_vec
    batch = safe_rows // tokens_vec
    ones_i = ct.full(shape=(BLOCK_R,), fill_value=1, dtype=ct.int32)
    patch = token - ones_i
    safe_patch = ct.where(token > zeros_i, patch, zeros_i)
    is_class_1d = token == zeros_i

    # Column mask (HIDDEN may be < BLOCK_C).
    cols_1d = ct.arange(BLOCK_C, dtype=ct.int32)
    col_lim = ct.full(shape=(BLOCK_C,), fill_value=HIDDEN, dtype=ct.int32)
    col_valid_1d = cols_1d < col_lim  # (BLOCK_C,)
    col_valid_2d = ct.reshape(col_valid_1d, (1, BLOCK_C))
    row_valid_2d = ct.reshape(row_valid, (BLOCK_R, 1))
    tile_valid_2d = row_valid_2d & col_valid_2d  # (BLOCK_R, BLOCK_C)

    is_class_2d = ct.reshape(is_class_1d, (BLOCK_R, 1))
    batch_bc = ct.broadcast_to(
        ct.reshape(batch, (BLOCK_R, 1)), (BLOCK_R, BLOCK_C)
    )
    patch_bc = ct.broadcast_to(
        ct.reshape(safe_patch, (BLOCK_R, 1)), (BLOCK_R, BLOCK_C)
    )
    token_bc = ct.broadcast_to(
        ct.reshape(token, (BLOCK_R, 1)), (BLOCK_R, BLOCK_C)
    )
    cols_bc = ct.broadcast_to(
        ct.reshape(cols_1d, (1, BLOCK_C)), (BLOCK_R, BLOCK_C)
    )

    class_1d = ct.load(class_token_ptr, index=(0,), shape=(BLOCK_C,),
                        padding_mode=ct.PaddingMode.ZERO)
    class_bc = ct.broadcast_to(ct.reshape(class_1d, (1, BLOCK_C)),
                               (BLOCK_R, BLOCK_C))
    class_f = ct.astype(class_bc, ct.float32)

    patch_bf = ct.gather(conv_flat_ptr, (batch_bc, patch_bc, cols_bc),
                         mask=tile_valid_2d)
    patch_f = ct.astype(patch_bf, ct.float32)

    pos_bf = ct.gather(pos_flat_ptr, (token_bc, cols_bc), mask=tile_valid_2d)
    pos_f = ct.astype(pos_bf, ct.float32)

    src_f = ct.where(is_class_2d, class_f, patch_f)
    x_unrounded = src_f + pos_f
    x_bf = ct.astype(x_unrounded, ct.bfloat16)
    x_f = ct.astype(x_bf, ct.float32)

    # Mask before reduce so OOB columns don't contaminate mean/variance.
    zeros_f = ct.zeros((BLOCK_R, BLOCK_C), dtype=ct.float32)
    x_masked = ct.where(col_valid_2d, x_f, zeros_f)

    inv_h = 1.0 / HIDDEN
    row_sum = ct.sum(x_masked, axis=1, keepdims=True)
    mean = row_sum * inv_h
    centered = x_f - mean
    centered_masked = ct.where(col_valid_2d, centered, zeros_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1,
                       keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    weight_bf = ct.load(weight_ptr, index=(0,), shape=(BLOCK_C,),
                         padding_mode=ct.PaddingMode.ZERO)
    bias_bf = ct.load(bias_ptr, index=(0,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight_bf, ct.float32)
    bias_f = ct.astype(bias_bf, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_C))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_C))
    y = centered * invstd * weight_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)

    # Scatter with mask: write only valid (row, col) pairs.
    hidden_v = ct.full(shape=(BLOCK_R, BLOCK_C), fill_value=HIDDEN,
                        dtype=ct.int32)
    row_offsets = ct.broadcast_to(
        ct.reshape(safe_rows, (BLOCK_R, 1)), (BLOCK_R, BLOCK_C)
    )
    flat_off = row_offsets * hidden_v + cols_bc
    ct.scatter(add_out_ptr, flat_off, x_bf, mask=tile_valid_2d)
    ct.scatter(norm_out_ptr, flat_off, y_bf, mask=tile_valid_2d)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="b1389f17", BLOCK_R=4, BLOCK_C=256)
@oracle_impl(hardware="B200", point="5b1eacf5", BLOCK_R=1, BLOCK_C=1024)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C):
    class_token, conv, pos, weight, bias, _sp0, _sp1, out_shape = inputs
    batch = int(conv.shape[0])
    hidden = int(weight.shape[0])
    tokens = int(pos.shape[1])
    rows = batch * tokens
    add_shape = (batch, tokens, hidden)
    norm_shape = _shape_tuple(out_shape)

    add_out = torch.empty_strided(
        add_shape, _contiguous_stride(add_shape),
        device=conv.device, dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        norm_shape, _contiguous_stride(norm_shape),
        device=conv.device, dtype=torch.bfloat16,
    )

    h, w = int(conv.shape[2]), int(conv.shape[3])
    conv_nhwc = conv.permute(0, 2, 3, 1)
    patches = h * w
    conv_flat = conv_nhwc.contiguous().view(batch, patches, hidden)
    pos_flat = pos.view(tokens, hidden)
    class_flat = class_token.view(hidden)
    weight_flat = weight.view(hidden)
    bias_flat = bias.view(hidden)

    add_flat = add_out.view(-1)
    norm_flat = norm_out.view(-1)

    stream = torch.cuda.current_stream()
    grid = ((rows + BLOCK_R - 1) // BLOCK_R, 1, 1)
    ct.launch(
        stream, grid, _vit_patch_layernorm_kernel,
        (class_flat, conv_flat, pos_flat, weight_flat, bias_flat,
         add_flat, norm_flat, rows, tokens, hidden, BLOCK_R, BLOCK_C),
    )
    return add_out, norm_out
