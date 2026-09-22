"""cuTile port of var_mean_3e7683cb24f2: GPT-J mixed-dtype residual LayerNorm.

SCHEDULER_FUSION: for each row: bf16 add lhs+rhs (bf16 side output),
fp32 add(round-through-bf16, residual) (fp32 side output), fp32 mean/var (dim=2,
correction=0), rsqrt (eps=1e-5) side output, mean side output, per-feature
weight/bias affine, final bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _gptj_residual_layernorm_kernel(
    lhs_ptr,          # bf16 [rows, hidden]
    rhs_ptr,          # bf16 [rows, hidden]
    residual_ptr,     # f32 [rows, hidden]
    weight_ptr,       # f32 [hidden]
    bias_ptr,         # f32 [hidden]
    add_bf16_out_ptr, # bf16 [rows, hidden]
    add_f32_out_ptr,  # f32 [rows, hidden]
    mean_out_ptr,     # f32 [rows]
    rsqrt_out_ptr,    # f32 [rows]
    final_out_ptr,    # bf16 [rows, hidden]
    HIDDEN: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    lhs = ct.load(lhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    rhs = ct.load(rhs_ptr, index=(row, 0), shape=(1, BLOCK_H))
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H))
    lhs_f = ct.astype(lhs, ct.float32)
    rhs_f = ct.astype(rhs, ct.float32)

    add_f32_ = lhs_f + rhs_f
    add_bf16 = ct.astype(add_f32_, ct.bfloat16)
    ct.store(add_bf16_out_ptr, index=(row, 0), tile=add_bf16)

    x = ct.astype(add_bf16, ct.float32) + residual
    ct.store(add_f32_out_ptr, index=(row, 0), tile=x)

    mean = ct.sum(x) * (1.0 / HIDDEN)
    centered = x - mean
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS)

    # Store mean and invstd as length-1 vectors (indexed by row).
    mean_tile = ct.full((1,), 0.0, dtype=ct.float32) + mean
    invstd_tile = ct.full((1,), 0.0, dtype=ct.float32) + invstd
    ct.store(mean_out_ptr, index=(row,), tile=mean_tile)
    ct.store(rsqrt_out_ptr, index=(row,), tile=invstd_tile)

    normalized = centered * invstd
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,))
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(final_out_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="3fdaed2a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    lhs, rhs, residual, weight, bias, add_shape, add_f32_shape, final_shape = inputs
    rows = int(lhs.shape[0])
    hidden = int(lhs.shape[1])
    add_shape = _as_shape(add_shape)
    add_f32_shape = _as_shape(add_f32_shape)
    final_shape = _as_shape(final_shape)

    add_bf16 = torch.empty_strided(
        add_shape, (rows * hidden, hidden, 1),
        device=lhs.device, dtype=torch.bfloat16,
    )
    add_f32 = torch.empty_strided(
        add_f32_shape, (rows * hidden, hidden, 1),
        device=lhs.device, dtype=torch.float32,
    )
    mean = torch.empty_strided(
        (1, rows, 1), (rows, 1, 1),
        device=lhs.device, dtype=torch.float32,
    )
    rsqrt = torch.empty_strided(
        (1, rows, 1), (rows, 1, 1),
        device=lhs.device, dtype=torch.float32,
    )
    final = torch.empty_strided(
        final_shape, (hidden, 1),
        device=lhs.device, dtype=torch.bfloat16,
    )
    # 2D views for the row kernel.
    add_bf16_2d = add_bf16.view(rows, hidden)
    add_f32_2d = add_f32.view(rows, hidden)
    residual_2d = residual.view(rows, hidden)
    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _gptj_residual_layernorm_kernel,
        (lhs, rhs, residual_2d, weight, bias,
         add_bf16_2d, add_f32_2d, mean_1d, rsqrt_1d, final,
         hidden, BLOCK_H),
    )
    return add_bf16, add_f32, mean, rsqrt, final
