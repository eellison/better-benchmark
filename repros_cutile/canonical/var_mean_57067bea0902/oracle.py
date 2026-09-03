"""cuTile port of var_mean_57067bea0902: Add + LayerNorm row kernel with multiple side outputs."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


ROWS = 4096
HIDDEN = 4096
EPS = 1.0e-12


@ct.kernel
def _add_layernorm_full_kernel(
    flat_bf16_ptr,     # (ROWS, HIDDEN) bf16
    residual_ptr,      # (ROWS, HIDDEN) f32
    weight_ptr,        # (HIDDEN,) f32
    bias_ptr,          # (HIDDEN,) f32
    normalized_ptr,    # (ROWS, HIDDEN) f32
    affine_ptr,        # (ROWS, HIDDEN) f32
    final_bf16_ptr,    # (ROWS, HIDDEN) bf16
    invstd_div_ptr,    # (ROWS,) f32
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.astype(ct.load(flat_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    x = flat + residual

    # var_mean correction=0: mean = sum/N, var = sum((x-mean)^2)/N.
    mean = ct.sum(x) * (1.0 / BLOCK_H)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / BLOCK_H)
    invstd = ct.rsqrt(variance + EPS)
    normalized = centered * invstd

    weight = ct.reshape(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)),
        (1, BLOCK_H),
    )
    bias = ct.reshape(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)),
        (1, BLOCK_H),
    )
    affine = normalized * weight + bias

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(final_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    ct.store(invstd_div_ptr, index=(row,), tile=ct.reshape(invstd * (1.0 / BLOCK_H), (1,)))


@oracle_impl(hardware="B200", point="ba44cc6a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs

    normalized = torch.empty_strided(
        (8, 512, 4096),
        (2097152, 4096, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    affine = torch.empty_strided(
        (8, 512, 4096),
        (2097152, 4096, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        (4096, 4096),
        (4096, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        (8, 512, 1),
        (512, 1, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    # Flatten to (ROWS, HIDDEN)
    normalized_flat = normalized.view(ROWS, HIDDEN)
    affine_flat = affine.view(ROWS, HIDDEN)
    residual_flat = arg1_1.view(ROWS, HIDDEN)
    invstd_flat = invstd_div.view(ROWS)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _add_layernorm_full_kernel,
        (arg0_1, residual_flat, arg2_1, arg3_1, normalized_flat, affine_flat,
         final_bf16, invstd_flat, BLOCK_H),
    )
    return normalized, affine, final_bf16, invstd_div
