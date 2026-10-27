"""cuTile port of var_mean_0584ac272c37: Swin residual add + LayerNorm."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 128 * 49
HIDDEN = 1024
EPS = 1.0e-5


@ct.kernel
def _swin_layernorm_kernel(
    flat_ptr,      # (ROWS, HIDDEN) bf16
    residual_ptr,  # (ROWS, HIDDEN) bf16
    weight_ptr,    # (HIDDEN,) bf16
    bias_ptr,      # (HIDDEN,) bf16
    add_out_ptr,   # (ROWS, HIDDEN) bf16
    norm_out_ptr,  # (ROWS, HIDDEN) bf16
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    added_bf16 = flat + residual
    ct.store(add_out_ptr, index=(row, 0), tile=added_bf16)

    x = ct.astype(added_bf16, ct.float32)
    mean = ct.sum(x) * (1.0 / BLOCK_H)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / BLOCK_H)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.astype(ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(norm_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


@oracle_impl(hardware="B200", point="d965afe2", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, *_shapes = inputs
    # arg0_1 is (6272, 1024) bf16 = flat; arg1_1 is (128, 7, 7, 1024) bf16 = residual.
    # Reshape residual to (rows, hidden). Note: the Swin permute/view chain reduces
    # to identity for singleton grid — arg1_1.view(rows, hidden) is correct.
    residual_2d = arg1_1.view(ROWS, HIDDEN)

    add_out = torch.empty_strided(
        (128, 49, HIDDEN),
        (49 * HIDDEN, HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _swin_layernorm_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, add_out.view(ROWS, HIDDEN), norm_out, BLOCK_H),
    )
    return add_out, norm_out
