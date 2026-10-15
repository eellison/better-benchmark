"""cuTile port of var_mean_38fe02d3252c: DeiT LayerNorm with class-token affine.

Two kernels:
1. LN row kernel: for all rows, compute normalized (fp32) and div (fp32).
2. Class-token affine kernel: for class-token rows only (rows 0::TOKENS),
   apply affine and cast to bf16.

HIDDEN=192 loaded directly with PaddingMode.ZERO (no torch pre-pad); only the
first HIDDEN cols are valid for reductions and for the final store into a
padded output buffer.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-6


@ct.kernel
def _layernorm_row_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # f32  [rows, HIDDEN]
    norm_out_ptr,   # f32  [rows, BLOCK_H] (padded)
    div_out_ptr,    # f32  [rows]
    HIDDEN: ct.Constant[int],
    EPS_C: ct.Constant[float],
    BLOCK_H: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.astype(
        ct.load(flat_ptr, index=(row, 0), shape=(1, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, BLOCK_H),
                       padding_mode=ct.PaddingMode.ZERO)
    x = residual + flat

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_valid = cols < HIDDEN
    col_valid_2d = ct.reshape(col_valid, (1, BLOCK_H))
    x_masked = ct.where(col_valid_2d, x, 0.0)

    mean = ct.sum(x_masked) * (1.0 / HIDDEN)
    centered = x - mean
    centered_masked = ct.where(col_valid_2d, centered, 0.0)
    variance = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + EPS_C)
    normalized = centered * invstd

    ct.store(norm_out_ptr, index=(row, 0), tile=normalized)
    ct.store(div_out_ptr, index=(row,), tile=invstd * (1.0 / HIDDEN))


@ct.kernel
def _class_affine_kernel(
    norm_ptr,       # f32 [rows, BLOCK_H] (padded)
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    class_bf16_ptr, # bf16 [batch, BLOCK_H] (padded)
    class_row_stride: ct.Constant[int],  # TOKENS (row idx = batch * TOKENS)
    BLOCK_H: ct.Constant[int],
):
    batch = ct.bid(0)
    src_row = batch * class_row_stride
    normalized = ct.load(norm_ptr, index=(src_row, 0), shape=(1, BLOCK_H))
    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    affine = normalized * weight_2d + bias_2d
    ct.store(class_bf16_ptr, index=(batch, 0), tile=ct.astype(affine, ct.bfloat16))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(hardware="B200", point="0ff22f63", BLOCK_H=256, ROW_BLOCK=1)
def oracle_forward(inputs, *, BLOCK_H: int, ROW_BLOCK: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0 = inputs
    norm_shape = _as_shape(shape0)
    batch = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])
    hidden = int(arg1_1.shape[2])
    rows = int(arg0_1.shape[0])

    # arg0_1 is [rows, hidden] bf16 (as loaded from Repro)
    # arg1_1 is [batch, tokens, hidden] fp32 — view as [rows, hidden]
    residual_2d = arg1_1.view(rows, hidden)

    norm_padded = torch.empty((rows, BLOCK_H), device=arg0_1.device, dtype=torch.float32)
    div_flat = torch.empty((rows,), device=arg0_1.device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _layernorm_row_kernel,
        (arg0_1, residual_2d, norm_padded, div_flat, hidden, EPS, BLOCK_H),
    )

    class_bf16_padded = torch.empty((batch, BLOCK_H), device=arg0_1.device, dtype=torch.bfloat16)
    ct.launch(
        stream,
        (batch, 1, 1),
        _class_affine_kernel,
        (norm_padded, arg2_1, arg3_1, class_bf16_padded, tokens, BLOCK_H),
    )

    # Slice back to expected valid ranges
    normalized = norm_padded[:, :hidden].view(norm_shape).contiguous()
    class_bf16 = class_bf16_padded[:, :hidden].contiguous()
    div = div_flat.view(batch, tokens, 1)

    return normalized, class_bf16, div
