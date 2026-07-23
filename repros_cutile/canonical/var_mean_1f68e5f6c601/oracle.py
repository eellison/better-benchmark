"""cuTile port of var_mean_1f68e5f6c601: bf16 add + f32 LayerNorm + affine.

Per row: bf16 flat + f32 residual → f32 x; population var_mean over HIDDEN,
rsqrt(var+1e-12), fp32 normalized, fp32 affine, bf16 output, and side output
`invstd / HIDDEN`.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _add_layernorm_kernel(
    flat_ptr,          # bf16 (rows, HIDDEN)
    residual_ptr,      # f32 (rows, HIDDEN)
    weight_ptr,        # f32 (HIDDEN,)
    bias_ptr,          # f32 (HIDDEN,)
    normalized_ptr,    # f32 (rows, HIDDEN_PADDED)
    affine_ptr,        # f32 (rows, HIDDEN_PADDED)
    final_bf16_ptr,    # bf16 (rows, HIDDEN_PADDED)
    invstd_div_ptr,    # f32 (rows,)
    HIDDEN: ct.Constant[int],
    HIDDEN_PADDED: ct.Constant[int],
):
    row = ct.bid(0)
    flat = ct.load(flat_ptr, index=(row, 0), shape=(1, HIDDEN_PADDED),
                   padding_mode=ct.PaddingMode.ZERO)
    residual = ct.load(residual_ptr, index=(row, 0), shape=(1, HIDDEN_PADDED),
                       padding_mode=ct.PaddingMode.ZERO)
    x = ct.astype(flat, ct.float32) + residual

    # Build mask for padding (padded elements have zero data but must be
    # excluded from mean/var).
    idx = ct.arange(HIDDEN_PADDED, dtype=ct.int32)
    idx_2d = ct.reshape(idx, (1, HIDDEN_PADDED))
    mask = idx_2d < HIDDEN
    zero_tile = ct.zeros(shape=(1, HIDDEN_PADDED), dtype=ct.float32)
    x_valid = ct.where(mask, x, zero_tile)

    mean = ct.sum(x_valid) * (1.0 / HIDDEN)
    centered_all = x - mean
    centered = ct.where(mask, centered_all, zero_tile)
    variance = ct.sum(centered * centered) * (1.0 / HIDDEN)
    invstd = ct.rsqrt(variance + 1.0e-12)
    normalized = centered_all * invstd  # includes garbage on masked cols

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PADDED,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PADDED,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, HIDDEN_PADDED))
    bias_2d = ct.reshape(bias, (1, HIDDEN_PADDED))
    affine = normalized * weight_2d + bias_2d

    ct.store(normalized_ptr, index=(row, 0), tile=normalized)
    ct.store(affine_ptr, index=(row, 0), tile=affine)
    ct.store(final_bf16_ptr, index=(row, 0), tile=ct.astype(affine, ct.bfloat16))
    invstd_over_h = invstd * (1.0 / HIDDEN)
    ct.store(invstd_div_ptr, index=(row,), tile=ct.reshape(invstd_over_h, (1,)))


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= dim
    return tuple(reversed(stride))


@oracle_impl(hardware="B200", point="f8dbfbf1", BLOCK_H=1024)
@oracle_impl(hardware="B200", point="ba44cc6a", BLOCK_H=4096)
def oracle_forward(inputs, *, BLOCK_H: int):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    view_shape = tuple(int(dim) for dim in shape0)
    final_shape = tuple(int(dim) for dim in shape1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    hidden_padded = _next_pow2(hidden)

    # Load inputs directly with padding_mode=ZERO — no pre-pad copies needed.
    residual_2d = arg1_1.view(rows, hidden)

    normalized = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    affine = torch.empty_strided(
        view_shape, _contiguous_stride(view_shape),
        device=arg0_1.device, dtype=torch.float32,
    )
    final_bf16 = torch.empty_strided(
        final_shape, _contiguous_stride(final_shape),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    invstd_div = torch.empty_strided(
        view_shape[:-1] + (1,),
        _contiguous_stride(view_shape[:-1] + (1,)),
        device=arg0_1.device, dtype=torch.float32,
    )

    # When hidden is a pow2, the store tile exactly covers the row — safe to
    # write straight into the output. Otherwise pad the output side and copy.
    if hidden == hidden_padded:
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (rows, 1, 1),
            _add_layernorm_kernel,
            (arg0_1, residual_2d, arg2_1, arg3_1,
             normalized.view(rows, hidden),
             affine.view(rows, hidden),
             final_bf16.view(rows, hidden),
             invstd_div.view(rows), hidden, hidden_padded),
        )
    else:
        normalized_padded = torch.empty((rows, hidden_padded), device=arg0_1.device, dtype=torch.float32)
        affine_padded = torch.empty((rows, hidden_padded), device=arg0_1.device, dtype=torch.float32)
        final_padded = torch.empty((rows, hidden_padded), device=arg0_1.device, dtype=torch.bfloat16)
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (rows, 1, 1),
            _add_layernorm_kernel,
            (arg0_1, residual_2d, arg2_1, arg3_1,
             normalized_padded, affine_padded, final_padded,
             invstd_div.view(rows), hidden, hidden_padded),
        )
        normalized.view(rows, hidden).copy_(normalized_padded[:, :hidden])
        affine.view(rows, hidden).copy_(affine_padded[:, :hidden])
        final_bf16.view(rows, hidden).copy_(final_padded[:, :hidden])
    return normalized, affine, final_bf16, invstd_div
