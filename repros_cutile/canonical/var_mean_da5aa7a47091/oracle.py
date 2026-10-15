"""cuTile port of var_mean_da5aa7a47091: residual-add LayerNorm.

Multi-row (ROW_BLOCK per block, matches Triton). Uses padding_mode=ZERO if
HIDDEN < BLOCK_N; otherwise loads exactly.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _residual_layernorm_view_kernel(
    flat_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    norm_out_ptr,   # bf16 [rows, HIDDEN]
    HIDDEN_C: ct.Constant[int],
    ROW_BLOCK: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
    ROUND_ADD: ct.Constant[int],
):
    block = ct.bid(0)
    flat = ct.load(flat_ptr, index=(block, 0), shape=(ROW_BLOCK, BLOCK_N),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(block, 0), shape=(ROW_BLOCK, BLOCK_N),
                       padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(flat, ct.float32) + ct.astype(residual, ct.float32)
    if ROUND_ADD:
        x = ct.astype(ct.astype(x_f, ct.bfloat16), ct.float32)
    else:
        x = x_f

    cols = ct.arange(BLOCK_N, dtype=ct.int32)
    col_mask_1d = cols < HIDDEN_C
    col_mask = ct.broadcast_to(ct.reshape(col_mask_1d, (1, BLOCK_N)),
                               (ROW_BLOCK, BLOCK_N))
    zero = ct.zeros((ROW_BLOCK, BLOCK_N), dtype=ct.float32)

    x_masked = ct.where(col_mask, x, zero)
    mean = ct.sum(x_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    centered = x - mean
    centered_masked = ct.where(col_mask, centered, zero)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * (1.0 / HIDDEN_F)
    invstd = ct.rsqrt(variance + 1.0e-12)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_N,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_N,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(ct.astype(weight, ct.float32), (1, BLOCK_N))
    bias_2d = ct.reshape(ct.astype(bias, ct.float32), (1, BLOCK_N))
    y = centered * invstd * weight_2d + bias_2d
    y_bf = ct.astype(y, ct.bfloat16)
    rows_1d = block * ROW_BLOCK + ct.arange(ROW_BLOCK, dtype=ct.int32)
    rows_bc = ct.broadcast_to(ct.reshape(rows_1d, (ROW_BLOCK, 1)),
                              (ROW_BLOCK, BLOCK_N))
    cols_bc = ct.broadcast_to(ct.reshape(cols, (1, BLOCK_N)),
                              (ROW_BLOCK, BLOCK_N))
    ct.scatter(norm_out_ptr, (rows_bc, cols_bc), y_bf, mask=col_mask)


@oracle_impl(hardware="B200", point="63bebcf6", BLOCK_N=1024, ROW_BLOCK=2, ROUND_ADD=True)
@oracle_impl(hardware="B200", point="107053b2", BLOCK_N=2048, ROW_BLOCK=1, ROUND_ADD=False)
@oracle_impl(hardware="B200", point="82b81ff3", BLOCK_N=1024, ROW_BLOCK=2, ROUND_ADD=True)
@oracle_impl(hardware="B200", point="94a8a62c", BLOCK_N=256, ROW_BLOCK=4, ROUND_ADD=True)
def oracle_forward(inputs, *, BLOCK_N, ROW_BLOCK, ROUND_ADD):
    # eager repro always rounds the add to bf16 first
    ROUND_ADD = True
    arg0_1, arg1_1, arg2_1, arg3_1, _shape0, _shape1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    norm_shape = tuple(int(dim) for dim in _shape0)
    view_shape = tuple(int(dim) for dim in _shape1)

    norm_out = torch.empty_strided(
        norm_shape,
        (norm_shape[1] * norm_shape[2], norm_shape[2], 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    norm_out_2d = norm_out.view(rows, hidden)
    residual_2d = arg1_1.view(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, ROW_BLOCK), 1, 1),
        _residual_layernorm_view_kernel,
        (arg0_1, residual_2d, arg2_1, arg3_1, norm_out_2d,
         hidden, ROW_BLOCK, BLOCK_N, float(hidden), int(ROUND_ADD)),
    )

    return (norm_out, norm_out.reshape(view_shape))
