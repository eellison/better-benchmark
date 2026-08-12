"""cuTile port of var_mean_45f7dfd4a983: residual + LayerNorm row kernel.

Per row: add residual (bf16), var_mean over hidden dim, rsqrt eps=1e-12,
affine with bf16 weight/bias, bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-12


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _residual_ln_kernel(
    view_ptr,       # bf16 [rows, HIDDEN]
    residual_ptr,   # bf16 [rows, HIDDEN]
    weight_ptr,     # bf16 [HIDDEN]
    bias_ptr,       # bf16 [HIDDEN]
    out_ptr,        # bf16 [rows, HIDDEN]
    HIDDEN: ct.Constant[int],
    HIDDEN_PAD: ct.Constant[int],
    HIDDEN_F: ct.Constant[float],
):
    row = ct.bid(0)

    view = ct.load(
        view_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, 0), shape=(1, HIDDEN_PAD),
        padding_mode=ct.PaddingMode.ZERO,
    )
    # add: bf16 + bf16 -> bf16 (with rtne)
    add_bf16 = ct.astype(ct.astype(view, ct.float32) + ct.astype(residual, ct.float32), ct.bfloat16)
    x_f = ct.astype(add_bf16, ct.float32)

    cols = ct.arange(HIDDEN_PAD, dtype=ct.int32)
    valid = cols < HIDDEN
    valid_2d = ct.reshape(valid, (1, HIDDEN_PAD))
    zero_f = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.float32)
    x_masked = ct.where(valid_2d, x_f, zero_f)

    mean_val = ct.sum(x_masked) * (1.0 / HIDDEN_F)
    centered = x_f - mean_val
    centered_masked = ct.where(valid_2d, centered, zero_f)
    variance_val = ct.sum(centered_masked * centered_masked) * (1.0 / HIDDEN_F)
    invstd_val = ct.rsqrt(variance_val + EPS)
    normalized = centered * invstd_val

    weight = ct.load(weight_ptr, index=(0,), shape=(HIDDEN_PAD,), padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(HIDDEN_PAD,), padding_mode=ct.PaddingMode.ZERO)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)
    weight_2d = ct.reshape(weight_f, (1, HIDDEN_PAD))
    bias_2d = ct.reshape(bias_f, (1, HIDDEN_PAD))
    affine = normalized * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    zero_bf16 = ct.full(shape=(1, HIDDEN_PAD), fill_value=0.0, dtype=ct.bfloat16)
    ct.store(out_ptr, index=(row, 0), tile=ct.where(valid_2d, affine_bf16, zero_bf16))


@oracle_impl(hardware="B200", point="d4701d13")
@oracle_impl(hardware="B200", point="0b3dc49f")
@oracle_impl(hardware="B200", point="63bebcf6")
@oracle_impl(hardware="B200", point="d1f40ce2")
@oracle_impl(hardware="B200", point="d4cc3e3e")
def oracle_forward(inputs):
    arg0_1, arg1_1, arg2_1, arg3_1, shape0, shape1 = inputs
    hidden = int(arg2_1.shape[0])
    HIDDEN_PAD = _next_pow2(hidden)
    residual_2d = arg1_1.reshape(-1, hidden)
    rows = residual_2d.shape[0]
    view_2d = arg0_1.reshape(rows, hidden)
    device = arg0_1.device

    out = torch.empty_strided(
        tuple(int(v) for v in shape0),
        (int(shape0[1]) * hidden, hidden, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_2d = out.reshape(rows, hidden)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (rows, 1, 1),
        _residual_ln_kernel,
        (view_2d, residual_2d, arg2_1, arg3_1, out_2d, hidden, HIDDEN_PAD, float(hidden)),
    )
    view_1 = out.view(tuple(int(v) for v in shape1))
    return out, view_1
