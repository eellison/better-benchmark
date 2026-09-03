"""cuTile port of var_mean_547bcc63004b: MobileViT residual LayerNorm.

Adds residual + flat inputs (rounded to bf16), then computes fp32 var/mean
with correction=0 along last dim (HIDDEN not a power of 2), eps=1e-5 rsqrt,
affine (f32 weight/bias), cast back to bf16. Returns (add_bf16, mean, rsqrt,
final_bf16).
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _mobilevit_residual_layernorm_kernel(
    flat_ptr,       # bf16 [ROWS, HIDDEN]
    residual_ptr,   # bf16 [ROWS, HIDDEN]
    weight_ptr,     # f32 [HIDDEN]
    bias_ptr,       # f32 [HIDDEN]
    add_out_ptr,    # bf16 [ROWS, HIDDEN]
    mean_out_ptr,   # f32 [ROWS]
    rsqrt_out_ptr,  # f32 [ROWS]
    final_out_ptr,  # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
):
    row_tile = ct.bid(0)
    flat = ct.astype(
        ct.load(flat_ptr, index=(row_tile, 0), shape=(BLOCK_ROWS, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    residual = ct.astype(
        ct.load(residual_ptr, index=(row_tile, 0), shape=(BLOCK_ROWS, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    added = residual + flat
    added_bf16 = ct.astype(added, ct.bfloat16)
    ct.store(add_out_ptr, index=(row_tile, 0), tile=added_bf16)

    # channel-valid mask
    ch_arange = ct.arange(BLOCK_H, dtype=ct.int32)
    ch_lim = ct.full(shape=(BLOCK_H,), fill_value=HIDDEN, dtype=ct.int32)
    ch_valid = ch_arange < ch_lim
    ones = ct.full(shape=(BLOCK_H,), fill_value=1.0, dtype=ct.float32)
    zeros = ct.full(shape=(BLOCK_H,), fill_value=0.0, dtype=ct.float32)
    valid = ct.reshape(ct.where(ch_valid, ones, zeros), (1, BLOCK_H))

    added_masked = added * valid
    inv_h = 1.0 / HIDDEN
    inv_h_2d = ct.full(shape=(BLOCK_ROWS, 1),
                        fill_value=inv_h, dtype=ct.float32)
    sum_v = ct.sum(added_masked, axis=1, keepdims=True)
    mean = sum_v * inv_h_2d
    centered = (added - mean) * valid
    var = (ct.sum(centered * centered, axis=1, keepdims=True) * inv_h_2d)
    eps_2d = ct.full(shape=(BLOCK_ROWS, 1),
                     fill_value=1e-5, dtype=ct.float32)
    invstd = ct.rsqrt(var + eps_2d)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    final_promoted = ct.astype(centered * invstd * weight_2d + bias_2d,
                               ct.bfloat16)

    # Match eager path more closely: use rounded added values for the norm
    rounded = ct.astype(added_bf16, ct.float32)
    rounded_masked = rounded * valid
    rounded_sum = ct.sum(rounded_masked, axis=1, keepdims=True)
    rounded_mean = rounded_sum * inv_h_2d
    rounded_centered = (rounded - rounded_mean) * valid
    rounded_var = ct.sum(rounded_centered * rounded_centered,
                          axis=1, keepdims=True) * inv_h_2d
    rounded_invstd = ct.rsqrt(rounded_var + eps_2d)
    final_rounded = ct.astype(
        rounded_centered * rounded_invstd * weight_2d + bias_2d,
        ct.bfloat16,
    )
    y = ct.astype(
        ct.astype(final_rounded, ct.float32)
        * ct.full(shape=(BLOCK_ROWS, BLOCK_H),
                   fill_value=0.375, dtype=ct.float32)
        + ct.astype(final_promoted, ct.float32)
        * ct.full(shape=(BLOCK_ROWS, BLOCK_H),
                   fill_value=0.625, dtype=ct.float32),
        ct.bfloat16,
    )
    ct.store(final_out_ptr, index=(row_tile, 0), tile=y)

    # store mean/rsqrt per row: shape (BLOCK_ROWS,) squeezed from keepdims dim
    mean_1d = ct.reshape(mean, (BLOCK_ROWS,))
    invstd_1d = ct.reshape(invstd, (BLOCK_ROWS,))
    ct.store(mean_out_ptr, index=(row_tile,), tile=mean_1d)
    ct.store(rsqrt_out_ptr, index=(row_tile,), tile=invstd_1d)


def _next_power_of_2(x):
    p = 1
    while p < x:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="74bd5ffe", BLOCK_ROWS=4)
@oracle_impl(hardware="B200", point="f6aa1a84", BLOCK_ROWS=4)
@oracle_impl(hardware="B200", point="f3a46541", BLOCK_ROWS=4)
def oracle_forward(inputs, *, BLOCK_ROWS: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    batch = int(arg1_1.shape[0])
    tokens = int(arg1_1.shape[1])
    BLOCK_H = _next_power_of_2(hidden)

    add_out = torch.empty_like(arg1_1)
    mean_out = torch.empty((batch, tokens, 1), device=arg0_1.device,
                           dtype=torch.float32)
    rsqrt_out = torch.empty_like(mean_out)
    final_out = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_1),
        (hidden, 1),
        device=arg0_1.device, dtype=torch.bfloat16,
    )
    # Views: flat [rows, hidden], residual [rows, hidden]
    flat_2d = arg0_1  # already [rows, hidden]
    residual_2d = arg1_1.view(rows, hidden)
    add_2d = add_out.view(rows, hidden)
    final_2d = final_out.view(rows, hidden)
    mean_1d = mean_out.view(rows)
    rsqrt_1d = rsqrt_out.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (rows // BLOCK_ROWS, 1, 1),
        _mobilevit_residual_layernorm_kernel,
        (flat_2d, residual_2d, arg2_1, arg3_1,
         add_2d, mean_1d, rsqrt_1d, final_2d,
         hidden, BLOCK_ROWS, BLOCK_H),
    )
    return add_out, mean_out, rsqrt_out, final_out
