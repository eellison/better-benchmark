"""cuTile port of var_mean_88c8b20940d5: BEiT class-token + patch cat + LayerNorm.

Constructs the [BATCH, TOKENS, HIDDEN] cat tensor on the Python side using
torch.cat/permute (which are cheap and portable), then runs a single row
cuTile kernel for LayerNorm (var/mean, rsqrt eps=1e-6, affine, bf16 cast).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _layernorm_kernel(
    cat_ptr,        # f32 [rows, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    mean_ptr,       # f32 [rows]
    rsqrt_ptr,      # f32 [rows]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    x = ct.load(
        cat_ptr, index=(row, 0), shape=(1, BLOCK_H),
        padding_mode=ct.PaddingMode.ZERO,
    )
    col_idx = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(col_idx < HIDDEN, (1, BLOCK_H))
    x_masked = ct.where(col_mask, x, 0.0)
    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.load(
        weight_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    bias = ct.load(
        bias_ptr, index=(0,), shape=(BLOCK_H,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.store(out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="304d27fb", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1, shape2 = inputs
    batch = int(arg0_1.shape[0])
    hidden = int(arg2_1.shape[0])
    patch_count = int(arg0_1.shape[2]) * int(arg0_1.shape[3])
    tokens = patch_count + 1
    rows = batch * tokens
    device = arg0_1.device
    out_shape = _as_shape(shape2)

    # Build the cat tensor: expand class token to [BATCH, 1, HIDDEN],
    # permute+view patches to [BATCH, PATCHES, HIDDEN], then cat.
    view_patches = arg0_1.view(batch, hidden, patch_count)
    permuted_patches = view_patches.permute(0, 2, 1)
    # bf16 -> f32 for concat with the class-token f32 tensor.
    permuted_f32 = permuted_patches.to(torch.float32)
    class_expanded = arg1_1.expand(batch, 1, hidden)
    cat = torch.cat([class_expanded, permuted_f32], dim=1).contiguous()
    # cat shape: [batch, tokens, hidden]

    mean = torch.empty((batch, tokens, 1), device=device, dtype=torch.float32)
    rsqrt = torch.empty((batch, tokens, 1), device=device, dtype=torch.float32)
    out = torch.empty_strided(
        out_shape, _contiguous_stride(out_shape),
        device=device, dtype=torch.bfloat16,
    )

    cat_2d = cat.view(rows, hidden)
    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)
    out_2d = out.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _layernorm_kernel,
        (cat_2d, arg2_1, arg3_1, mean_1d, rsqrt_1d, out_2d,
         hidden, BLOCK_H),
    )
    return cat, mean, rsqrt, out
