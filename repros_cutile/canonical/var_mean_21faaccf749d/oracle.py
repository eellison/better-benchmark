"""cuTile port of var_mean_21faaccf749d: exact-erf GELU + LayerNorm.

Uses Abramowitz-Stegun 7.1.26 polynomial approximation to erf (max err ~1.5e-7)
so we can fuse GELU + LayerNorm into a single kernel, matching Triton's fused
pattern.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-7


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _gelu_layernorm_kernel(
    x_ptr,           # bf16 [ROWS, HIDDEN]
    weight_ptr,      # f32 [HIDDEN]
    bias_ptr,        # f32 [HIDDEN]
    mean_ptr,        # f32 [ROWS]
    rsqrt_ptr,       # f32 [ROWS]
    out_ptr,         # bf16 [ROWS, HIDDEN]
    HIDDEN: ct.Constant[int],
    BLOCK_M: ct.Constant[int],
    BLOCK_H: ct.Constant[int],
    INV_HIDDEN: ct.Constant[float],
):
    block = ct.bid(0)
    x = ct.load(x_ptr, index=(block, 0), shape=(BLOCK_M, BLOCK_H),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)

    # In-kernel erf approximation (Abramowitz-Stegun 7.1.26)
    erf_arg = x_f * 0.7071067811865476
    zero = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.float32)
    sign = ct.where(erf_arg >= zero,
                    ct.full((BLOCK_M, BLOCK_H), 1.0, dtype=ct.float32),
                    ct.full((BLOCK_M, BLOCK_H), -1.0, dtype=ct.float32))
    abs_arg = ct.where(erf_arg >= zero, erf_arg, -erf_arg)
    t = 1.0 / (1.0 + 0.3275911 * abs_arg)
    poly = (((((1.061405429 * t) - 1.453152027) * t + 1.421413741) * t
             - 0.284496736) * t + 0.254829592) * t
    erf_v = sign * (1.0 - poly * ct.exp(-abs_arg * abs_arg))
    gelu = (x_f * 0.5) * (erf_v + 1.0)
    gelu_bf16_f32 = ct.astype(ct.astype(gelu, ct.bfloat16), ct.float32)

    cols = ct.arange(BLOCK_H, dtype=ct.int32)
    col_mask_1d = cols < HIDDEN
    col_mask = ct.broadcast_to(ct.reshape(col_mask_1d, (1, BLOCK_H)),
                               (BLOCK_M, BLOCK_H))
    zero_f = ct.zeros((BLOCK_M, BLOCK_H), dtype=ct.float32)
    gelu_masked = ct.where(col_mask, gelu_bf16_f32, zero_f)
    mean = ct.sum(gelu_masked, axis=1, keepdims=True) * INV_HIDDEN
    centered = gelu_bf16_f32 - mean
    centered_masked = ct.where(col_mask, centered, zero_f)
    variance = ct.sum(centered_masked * centered_masked, axis=1, keepdims=True) * INV_HIDDEN
    invstd = ct.rsqrt(variance + EPS)

    weight = ct.load(weight_ptr, index=(0,), shape=(BLOCK_H,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(BLOCK_H,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight_2d = ct.reshape(weight, (1, BLOCK_H))
    bias_2d = ct.reshape(bias, (1, BLOCK_H))
    out = centered * invstd * weight_2d + bias_2d
    out_bf = ct.astype(out, ct.bfloat16)

    # Store mean/rsqrt per row
    mean_1d = ct.reshape(mean, (BLOCK_M,))
    invstd_1d = ct.reshape(invstd, (BLOCK_M,))
    ct.store(mean_ptr, index=(block,), tile=mean_1d)
    ct.store(rsqrt_ptr, index=(block,), tile=invstd_1d)

    if BLOCK_H == HIDDEN:
        ct.store(out_ptr, index=(block, 0), tile=out_bf)
    else:
        # Scatter with column mask
        rows_1d = block * BLOCK_M + ct.arange(BLOCK_M, dtype=ct.int32)
        rows_2d = ct.reshape(rows_1d, (BLOCK_M, 1))
        cols_2d = ct.reshape(cols, (1, BLOCK_H))
        rows_bc = ct.broadcast_to(rows_2d, (BLOCK_M, BLOCK_H))
        cols_bc = ct.broadcast_to(cols_2d, (BLOCK_M, BLOCK_H))
        ct.scatter(out_ptr, (rows_bc, cols_bc), out_bf, mask=col_mask)


def _launch(inputs, *, BLOCK_M, BLOCK_H):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1 = inputs
    view_shape = tuple(int(dim) for dim in _shape_param_0)
    out_shape = tuple(int(dim) for dim in _shape_param_1)
    rows = int(arg0_1.shape[0])
    hidden = int(arg0_1.shape[1])
    HIDDEN_PAD = _next_pow2(hidden)
    block_h = max(BLOCK_H, HIDDEN_PAD)

    stat_shape = (*view_shape[:-1], 1)
    stat_stride = (view_shape[1], 1, 1)
    mean = torch.empty_strided(stat_shape, stat_stride,
                               device=arg0_1.device, dtype=torch.float32)
    rsqrt = torch.empty_strided(stat_shape, stat_stride,
                                device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(out_shape, (hidden, 1),
                              device=arg0_1.device, dtype=torch.bfloat16)

    mean_1d = mean.view(rows)
    rsqrt_1d = rsqrt.view(rows)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(rows, BLOCK_M), 1, 1),
        _gelu_layernorm_kernel,
        (arg0_1, arg1_1, arg2_1, mean_1d, rsqrt_1d, out,
         hidden, BLOCK_M, block_h, 1.0 / hidden),
    )
    return mean, rsqrt, out


@oracle_impl(hardware="B200", point="cbab746f", BLOCK_M=1, BLOCK_H=2048)
@oracle_impl(hardware="B200", point="b2a77780", BLOCK_M=1, BLOCK_H=1024)
@oracle_impl(hardware="B200", point="ca0eabd2", BLOCK_M=1, BLOCK_H=1024)
@oracle_impl(hardware="B200", point="1398f333", BLOCK_M=4, BLOCK_H=128)
@oracle_impl(hardware="B200", point="a565199e", BLOCK_M=1, BLOCK_H=1024)
@oracle_impl(hardware="B200", point="3871c2e1", BLOCK_M=1, BLOCK_H=1024)
def oracle_forward(inputs, *, BLOCK_M, BLOCK_H):
    return _launch(inputs, BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H)
