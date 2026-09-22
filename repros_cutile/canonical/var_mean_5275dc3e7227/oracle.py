"""cuTile port of var_mean_5275dc3e7227: DeiT selective-token LayerNorm.

Compute the LayerNorm normalized f32 for all rows via cuTile row kernel,
then post-process in torch to (1) select tokens 0 and 1 with affine cast
to bf16, and (2) compute rsqrt/HIDDEN. HIDDEN=768, TOKENS=198, needs
padding to 1024 power-of-two.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _layernorm_normalized_and_invstd_kernel(
    residual_ptr,   # bf16 [rows, BLOCK_H]
    activation_ptr, # f32 [rows, BLOCK_H]
    normalized_ptr, # f32 [rows, BLOCK_H]
    invstd_ptr,     # f32 [rows]
    HIDDEN: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    activation = ct.load(activation_ptr, index=(row, 0), shape=(1, BLOCK_H))

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask = ct.reshape(cols < HIDDEN, (1, BLOCK_H))

    x = activation + ct.astype(residual, ct.float32)
    x_masked = ct.where(col_mask, x, 0.0)

    inv_h = 1.0 / HIDDEN_F
    mean = ct.sum(x_masked, axis=1, keepdims=True) * inv_h
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * inv_h
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(invstd_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="f13eb73e", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    residual_bf16, activation, weight, bias, shape0 = inputs
    norm_shape = _shape_tuple(shape0)
    batch = int(activation.shape[0])
    tokens = int(activation.shape[1])
    hidden = int(activation.shape[2])
    rows = int(residual_bf16.shape[0])
    device = activation.device

    # Prepare padded inputs.
    residual_2d = residual_bf16.contiguous().view(rows, hidden)
    activation_2d = activation.contiguous().view(rows, hidden)

    if BLOCK_H == hidden:
        residual_pad = residual_2d
        activation_pad = activation_2d
    else:
        residual_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.bfloat16)
        residual_pad[:, :hidden].copy_(residual_2d)
        activation_pad = torch.zeros((rows, BLOCK_H), device=device, dtype=torch.float32)
        activation_pad[:, :hidden].copy_(activation_2d)

    norm_pad = torch.empty((rows, BLOCK_H), device=device, dtype=torch.float32)
    invstd_1d = torch.empty((rows,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows, 1, 1), _layernorm_normalized_and_invstd_kernel,
        (residual_pad, activation_pad, norm_pad, invstd_1d, hidden, float(hidden), BLOCK_H),
    )

    if BLOCK_H == hidden:
        normalized = norm_pad.view(norm_shape)
    else:
        normalized = torch.empty(norm_shape, device=device, dtype=torch.float32)
        normalized.view(rows, hidden).copy_(norm_pad.narrow(1, 0, hidden))

    # Post-process: compute affine and select tokens 0, 1.
    # normalized is [batch, tokens, hidden]. Select tokens then affine.
    norm_3d = normalized.view(batch, tokens, hidden)
    token0 = norm_3d[:, 0, :] * weight + bias
    token1 = norm_3d[:, 1, :] * weight + bias
    selected0 = token0.to(torch.bfloat16)
    selected1 = token1.to(torch.bfloat16)

    div = invstd_1d.view(batch, tokens, 1) / hidden
    return normalized, selected0, selected1, div
