"""cuTile port of var_mean_2b21beb0932e (SCHEDULER_FUSION): ViT/DeiT
patch + class-token positional LayerNorm.

Each row is one token (class token or one of the 196/1369 patches). For each
row, gather the cat value from either the class token or the patch tensor,
add the positional embedding, compute row var/mean, normalize, apply affine,
and cast to bf16.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _vit_patch_layernorm_kernel(
    patch_ptr,        # bf16 flat storage of channels-last input
    class_token_ptr,  # f32 [1, 1, HIDDEN]
    position_ptr,     # f32 [1, TOKENS, HIDDEN]
    weight_ptr,       # f32 [HIDDEN]
    bias_ptr,         # f32 [HIDDEN]
    cat_out_ptr,      # f32 [ROWS * HIDDEN]
    add_out_ptr,      # f32 [ROWS * HIDDEN]
    mean_out_ptr,     # f32 [ROWS]
    invstd_out_ptr,   # f32 [ROWS]
    bf16_out_ptr,     # bf16 [ROWS * HIDDEN]
    rows_total: ct.Constant[int],
    tokens: ct.Constant[int],
    patches: ct.Constant[int],
    hidden: ct.Constant[int],
    patch_stride_b: ct.Constant[int],
    patch_stride_c: ct.Constant[int],
    patch_stride_p: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    cols = ct.arange(BLOCK_H, dtype=ct.int64)
    col_mask = cols < hidden

    batch = row // tokens
    token = row - batch * tokens
    is_class_token = token == 0

    zero64 = ct.zeros((BLOCK_H,), dtype=ct.int64)

    # Class token values (f32).
    safe_cols = ct.where(col_mask, cols, zero64)
    class_values = ct.gather(class_token_ptr, safe_cols)
    # Patch values (bf16) with runtime address:
    # patch_ptr[batch*stride_b + (token-1)*stride_p + cols*stride_c]
    patch_index = token - 1
    patch_off = batch * patch_stride_b + patch_index * patch_stride_p + safe_cols * patch_stride_c
    # For class token rows, we don't need to load valid patch data (multiply by 0 later).
    # Use safe indexing:
    patch_valid = (patch_index >= 0) & (patch_index < patches) & col_mask
    safe_patch = ct.where(patch_valid, patch_off, ct.zeros((BLOCK_H,), dtype=ct.int64))
    patch_bf = ct.gather(patch_ptr, safe_patch)
    patch_f = ct.astype(patch_bf, ct.float32)

    zero_f = ct.full((BLOCK_H,), 0.0, dtype=ct.float32)
    # Select class or patch value.
    cat_values = ct.where(is_class_token, class_values, patch_f)
    cat_values = ct.where(col_mask, cat_values, zero_f)

    # Store cat_out.
    cat_offsets = row * hidden + cols
    safe_cat_off = ct.where(col_mask, cat_offsets, zero64)
    ct.scatter(cat_out_ptr, safe_cat_off, cat_values, mask=col_mask)

    # Load positional embedding: [1, TOKENS, HIDDEN] contiguous.
    pos_off = token * hidden + safe_cols
    position = ct.gather(position_ptr, pos_off)
    position = ct.where(col_mask, position, zero_f)

    add_values = cat_values + position
    ct.scatter(add_out_ptr, safe_cat_off, add_values, mask=col_mask)

    # Reduce over cols.
    reduce_input = ct.where(col_mask, add_values, zero_f)
    inv_h = 1.0 / float(hidden)
    mean = ct.sum(reduce_input) * inv_h
    centered = add_values - ct.broadcast_to(ct.reshape(mean, (1,)), (BLOCK_H,))
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked) * inv_h
    invstd = ct.rsqrt(variance + EPS)

    # Store mean and invstd scalars for this row.
    row_idx_1 = ct.arange(1, dtype=ct.int64) + row
    ct.scatter(mean_out_ptr, row_idx_1, ct.reshape(mean, (1,)))
    ct.scatter(invstd_out_ptr, row_idx_1, ct.reshape(invstd, (1,)))

    weight = ct.gather(weight_ptr, safe_cols)
    bias = ct.gather(bias_ptr, safe_cols)
    invstd_broad = ct.broadcast_to(ct.reshape(invstd, (1,)), (BLOCK_H,))
    affine = centered * invstd_broad * weight + bias
    affine_bf = ct.astype(affine, ct.bfloat16)
    ct.scatter(bf16_out_ptr, safe_cat_off, affine_bf, mask=col_mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="efb792b2")
@oracle_impl(hardware="B200", point="4d589a1c")
def oracle_forward(inputs):
    patch_tensor, class_token, position, weight, bias, _patch_view_shape, _expand_shape, view_shape = inputs

    batch = int(patch_tensor.shape[0])
    hidden = int(patch_tensor.shape[1])
    patch_count = int(patch_tensor.shape[2]) * int(patch_tensor.shape[3])
    tokens = patch_count + 1
    rows = batch * tokens
    token_shape = (batch, tokens, hidden)

    cat_out = torch.empty_strided(token_shape, (tokens * hidden, hidden, 1),
                                    device=patch_tensor.device, dtype=torch.float32)
    add_out = torch.empty_strided(token_shape, (tokens * hidden, hidden, 1),
                                    device=patch_tensor.device, dtype=torch.float32)
    mean = torch.empty_strided((batch, tokens, 1), (tokens, 1, 1),
                                device=patch_tensor.device, dtype=torch.float32)
    invstd = torch.empty_strided((batch, tokens, 1), (tokens, 1, 1),
                                  device=patch_tensor.device, dtype=torch.float32)
    bf16_base = torch.empty_strided(token_shape, (tokens * hidden, hidden, 1),
                                      device=patch_tensor.device, dtype=torch.bfloat16)

    # Access patch_tensor storage as flat.
    patch_storage_size = int(patch_tensor.untyped_storage().nbytes() // patch_tensor.element_size())
    patch_flat = patch_tensor.as_strided((patch_storage_size,), (1,))
    class_flat = class_token.reshape(-1)
    position_flat = position.reshape(-1)
    weight_flat = weight.reshape(-1)
    bias_flat = bias.reshape(-1)
    cat_flat = cat_out.reshape(-1)
    add_flat = add_out.reshape(-1)
    mean_flat = mean.reshape(-1)
    invstd_flat = invstd.reshape(-1)
    bf16_flat = bf16_base.reshape(-1)

    BLOCK_H = _next_power_of_2(hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _vit_patch_layernorm_kernel,
        (patch_flat, class_flat, position_flat, weight_flat, bias_flat,
         cat_flat, add_flat, mean_flat, invstd_flat, bf16_flat,
         rows, tokens, patch_count, hidden,
         int(patch_tensor.stride(0)), int(patch_tensor.stride(1)),
         int(patch_tensor.stride(3)),
         BLOCK_H),
    )

    return cat_out, add_out, mean, invstd, bf16_base.view(_shape_tuple(view_shape))
