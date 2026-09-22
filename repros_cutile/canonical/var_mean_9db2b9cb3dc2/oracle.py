"""cuTile port of var_mean_9db2b9cb3dc2: DeiT distilled-token patch LayerNorm.

Builds the concatenated [class_token, dist_token, patches] tensor of shape
[128, 198, 768], adds positional embeddings, then runs LayerNorm along the
last dim.

Class/dist tokens are broadcast; patch tensor has channels-last strides.
Do the concat + positional add in torch (it involves gather/permute), then
LayerNorm in a cuTile row kernel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
HIDDEN = 768
BLOCK_H = 1024


@ct.kernel
def _layernorm_row_kernel(
    add_ptr,        # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN_: ct.Constant[int],
    BLOCK_H_: ct.Constant[int],
):
    row = ct.bid(0)
    add_bf = ct.load(
        add_ptr, index=(row, 0), shape=(1, BLOCK_H_),
        padding_mode=ct.PaddingMode.ZERO,
    )
    x_f = ct.astype(add_bf, ct.float32)
    col_idx = ct.arange(BLOCK_H_, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN_, (1, BLOCK_H_))
    zero_2d = ct.zeros((1, BLOCK_H_), dtype=ct.float32)
    x_masked = ct.where(col_mask, x_f, zero_2d)
    total = ct.sum(x_masked)
    mean = total * (1.0 / HIDDEN_)
    centered = x_f - mean
    centered_masked = ct.where(col_mask, centered, zero_2d)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H_,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_f = ct.astype(ct.reshape(weight, (1, BLOCK_H_)), ct.float32)
    bias_f = ct.astype(ct.reshape(bias, (1, BLOCK_H_)), ct.float32)
    affine = normalized * weight_f + bias_f
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(d) for d in shape)


@oracle_impl(hardware="B200", point="03ce16f1")
def oracle_forward(inputs):
    (
        class_token,     # bf16 [1, 1, 768]
        dist_token,      # bf16 [1, 1, 768]
        patches,         # bf16 [128, 768, 14, 14] channels-last
        pos_embed,       # bf16 [1, 198, 768]
        weight,          # bf16 [768]
        bias,            # bf16 [768]
        class_expand_shape,
        dist_expand_shape,
        patch_view_shape,
        output_view_shape,
    ) = inputs

    batch = int(patches.shape[0])
    hidden = int(weight.shape[0])
    patch_count = int(patches.shape[2]) * int(patches.shape[3])  # 196
    tokens = patch_count + 2  # 198
    rows = batch * tokens

    device = patches.device

    # Build the concatenated pre-norm tensor:
    #   class_expand: [128, 1, 768]  (broadcast class_token[0,0,:])
    #   dist_expand:  [128, 1, 768]  (broadcast dist_token[0,0,:])
    #   permute:      [128, 196, 768]  (permute+contig of patches)
    class_expand = class_token.expand(_as_shape(class_expand_shape))
    dist_expand = dist_token.expand(_as_shape(dist_expand_shape))
    # patches [128, 768, 14, 14] channels-last -> view [128, 768, 196]
    patch_flat = patches.reshape(batch, hidden, patch_count)
    # -> permute [128, 196, 768]
    patch_permute = patch_flat.permute(0, 2, 1).contiguous()

    cat = torch.cat([class_expand, dist_expand, patch_permute], dim=1)  # [128, 198, 768]
    add = cat + pos_embed  # broadcast [1,198,768] -> bf16 add
    # add is bf16 [128, 198, 768]

    # LayerNorm along last dim via cuTile.
    add_2d = add.contiguous().view(rows, hidden)
    out_2d = torch.empty_strided(
        (rows, hidden), (hidden, 1),
        device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_row_kernel,
        (add_2d, weight, bias, out_2d, hidden, BLOCK_H),
    )

    out_view = out_2d.view(_as_shape(output_view_shape))
    return add, out_view
