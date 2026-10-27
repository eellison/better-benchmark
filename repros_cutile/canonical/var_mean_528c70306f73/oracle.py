"""cuTile port of var_mean_528c70306f73 (SCHEDULER_FUSION): mixed bf16/fp32
residual LayerNorm with returned normalized-f32/final-bf16/saved-scale outputs.

Per row of `[ROWS, HIDDEN]`:
  x = residual_f32 + flat_bf16.to(f32)
  mean = sum(x) / HIDDEN
  var = sum((x - mean)^2) / HIDDEN     (correction=0)
  invstd = rsqrt(var + 1e-6)
  normalized = (x - mean) * invstd
  final_bf16 = (normalized * weight + bias).to(bf16)
  saved = invstd / HIDDEN
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mixed_residual_layernorm_kernel(
    flat_ptr,        # bf16 [ROWS, HIDDEN]
    residual_ptr,    # f32 [ROWS, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    normalized_ptr,  # f32 [ROWS, HIDDEN]
    final_bf16_ptr,  # bf16 [ROWS, HIDDEN]
    invstd_div_ptr,  # f32 [ROWS]
    HIDDEN: ct.Constant[int],
    SIDE_DIVISOR: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    residual = ct.astype(
        ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    flat = ct.astype(
        ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H)), ct.float32
    )
    x = residual + flat

    inv_h = 1.0 / HIDDEN
    x_sum = ct.sum(x, axis=1, keepdims=True)
    x_sq_sum = ct.sum(x * x, axis=1, keepdims=True)
    mean = x_sum * inv_h
    variance = x_sq_sum * inv_h - mean * mean
    centered = x - mean
    invstd = ct.rsqrt(variance + 1.0e-6)
    normalized = centered * invstd
    ct.store(normalized_ptr, index=(row, 0), tile=normalized)

    weight = ct.astype(
        ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    bias = ct.astype(
        ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,)), ct.float32
    )
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    final = ct.astype(affine, ct.bfloat16)
    ct.store(final_bf16_ptr, index=(row, 0), tile=final)

    invstd_scalar = ct.reshape(invstd, (1,))
    ct.store(invstd_div_ptr, index=(row,), tile=invstd_scalar * SIDE_DIVISOR)


@oracle_impl(hardware="B200", point="7b097b88", BLOCK_H=1024, ROW_BLOCK=1)
@oracle_impl(hardware="B200", point="5d43e450", BLOCK_H=2048, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H, ROW_BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    flat_shape = tuple(int(dim) for dim in _shape_param_1)

    # arg1_1 is f32[BATCH, SEQ, HIDDEN] — view as [rows, hidden].
    residual_2d = arg1_1.view(rows, hidden)

    normalized = torch.empty(
        view_shape, device=arg0_1.device, dtype=torch.float32
    )
    normalized_2d = normalized.view(rows, hidden)
    final_bf16 = torch.empty(
        flat_shape, device=arg0_1.device, dtype=torch.bfloat16
    )
    invstd_div = torch.empty(
        (view_shape[0], view_shape[1], 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    invstd_flat = invstd_div.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _mixed_residual_layernorm_kernel,
        (
            arg0_1,
            residual_2d,
            arg2_1,
            arg3_1,
            normalized_2d,
            final_bf16,
            invstd_flat,
            hidden,
            1.0 / hidden,
            BLOCK_H,
        ),
    )
    return normalized, final_bf16, invstd_div
