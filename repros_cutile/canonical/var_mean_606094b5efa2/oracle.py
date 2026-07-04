"""cuTile port of var_mean_606094b5efa2: BEiT residual-scale LayerNorm multi-output.

Per row: gamma*flat_bf16 (fp32), residual add, var_mean over hidden, rsqrt
with eps, affine (weight, bias), final bf16 cast. Returns (add, mean, rsqrt,
view of final_bf16).

Uses padding_mode=ZERO to avoid all the torch pad copies.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6
ROWS = 25216
HIDDEN = 768
BATCH = 128
TOKENS = 197


@ct.kernel
def _beit_layernorm_kernel(
    flat_bf16_ptr,  # bf16 [ROWS, HIDDEN]
    gamma_ptr,      # f32 [HIDDEN]
    residual_ptr,   # f32 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    add_ptr,        # f32 [ROWS, HIDDEN]
    mean_ptr,       # f32 [ROWS]
    rsqrt_ptr,      # f32 [ROWS]
    final_bf16_ptr, # bf16 [ROWS, HIDDEN]
    HIDDEN_SIZE: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    row = ct.bid(0)

    flat = ct.load(flat_bf16_ptr, index=(row, 0), shape=(1, BLOCK_H),
                   padding_mode=ct.PaddingMode.ZERO)
    gamma = ct.load(gamma_ptr, index=(0,), shape=(BLOCK_H,),
                    padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)

    flat_f = ct.astype(flat, ct.float32)
    gamma_f = ct.astype(gamma, ct.float32)
    gamma_2d = ct.reshape(gamma_f, (1, BLOCK_H))
    scaled = gamma_2d * flat_f
    x = residual + scaled

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_1d = cols < HIDDEN_SIZE
    col_mask = ct.reshape(col_mask_1d, (1, BLOCK_H))
    x_for_sum = ct.where(col_mask, x, ct.zeros((1, BLOCK_H), dtype=ct.float32))

    # Scatter x_for_sum to add_ptr — full-tile store would OOB into empty pad;
    # instead scatter with column mask.
    cols_2d = ct.reshape(cols, (1, BLOCK_H))
    row_2d = ct.full((1, BLOCK_H), row, dtype=ct.int32)
    ct.scatter(add_ptr, (row_2d, cols_2d), x_for_sum, mask=col_mask)

    mean_val = ct.sum(x_for_sum) * INV_HIDDEN
    centered = x - mean_val
    centered_masked = ct.where(col_mask, centered, ct.zeros((1, BLOCK_H), dtype=ct.float32))
    variance = ct.sum(centered_masked * centered_masked) * INV_HIDDEN
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, BLOCK_H))
    bias_2d = ct.reshape(bias_f, (1, BLOCK_H))
    normalized = centered * invstd
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)

    ct.store(mean_ptr, index=(row,), tile=ct.reshape(mean_val, (1,)))
    ct.store(rsqrt_ptr, index=(row,), tile=ct.reshape(invstd, (1,)))
    ct.scatter(final_bf16_ptr, (row_2d, cols_2d), affine_bf16, mask=col_mask)


@oracle_impl(hardware="B200", point="f4c82f7a", BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, _shape0, _shape_param_1 = inputs
    device = arg0_1.device

    residual_2d = arg2_1.reshape(ROWS, HIDDEN)
    add_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.float32)
    final_out = torch.empty((ROWS, HIDDEN), device=device, dtype=torch.bfloat16)
    mean = torch.empty((ROWS,), device=device, dtype=torch.float32)
    rsqrt = torch.empty((ROWS,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ROWS, 1, 1),
        _beit_layernorm_kernel,
        (arg0_1, arg1_1, residual_2d, arg3_1, arg4_1,
         add_out, mean, rsqrt, final_out,
         HIDDEN, BLOCK_H, 1.0 / HIDDEN),
    )

    add_out_3d = add_out.view(BATCH, TOKENS, HIDDEN)
    mean_out = mean.view(BATCH, TOKENS, 1)
    rsqrt_out = rsqrt.view(BATCH, TOKENS, 1)
    return add_out_3d, mean_out, rsqrt_out, final_out.view(tuple(int(dim) for dim in _shape_param_1))
