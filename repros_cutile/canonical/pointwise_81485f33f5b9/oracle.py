"""cuTile port of pointwise_81485f33f5b9: BN inference + residual add + NaN-preserving ReLU.

Per-channel/per-row-block kernel that loads BN params for the channel, applies
BN affine (rounded to bf16), adds residual (rounded to bf16 again), then
applies NaN-preserving ReLU.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_residual_relu_kernel(
    mean_ptr,        # bf16 [C]
    x_ptr,           # bf16 [rows=B*C, HW]
    var_ptr,         # bf16 [C]
    weight_ptr,      # bf16 [C]
    bias_ptr,        # bf16 [C]
    residual_ptr,    # bf16 [rows, HW]
    out_ptr,         # bf16 [rows, HW]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    C: ct.Constant[int],
):
    row = ct.bid(0)
    col_block = ct.bid(1)
    channel = row - (row // C) * C

    x = ct.load(
        x_ptr, index=(row, col_block), shape=(1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    residual = ct.load(
        residual_ptr, index=(row, col_block), shape=(1, BLOCK_HW),
        padding_mode=ct.PaddingMode.ZERO,
    )
    mean = ct.astype(ct.load(mean_ptr, index=(channel,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(channel,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(channel,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(channel,), shape=(1,)), ct.float32)

    x_f = ct.astype(x, ct.float32)
    residual_f = ct.astype(residual, ct.float32)
    invstd = 1.0 / ct.sqrt(var + EPS)
    centered = x_f - mean
    normalized = centered * invstd
    scaled = normalized * weight
    biased = scaled + bias
    affine_bf16 = ct.astype(biased, ct.bfloat16)
    residual_sum_bf = ct.astype(ct.astype(affine_bf16, ct.float32) + residual_f, ct.bfloat16)
    residual_sum_f = ct.astype(residual_sum_bf, ct.float32)
    zero_f = ct.full((1, BLOCK_HW), 0.0, dtype=ct.float32)
    max_val = ct.where(residual_sum_f > zero_f, residual_sum_f, zero_f)
    is_nan = residual_sum_f != residual_sum_f
    relu = ct.where(is_nan, residual_sum_f, max_val)

    ct.store(out_ptr, index=(row, col_block), tile=ct.astype(relu, ct.bfloat16))


POINTS = [
    "85db042c", "0af2d571", "3aeeed95", "fd6e1886", "cd893d5b",
    "28e6bebb", "08581e62", "50714806", "4c0cf88f", "6b748dff",
    "7111f2e4", "2e6c4615", "e626f5a9", "40170966", "e2161da4",
    "72f4709b", "cce8fbb3", "2f5dac5e", "3bac8de4", "592ca691",
    "e20a55a6",
]


def _pick_block_hw(hw):
    """Pick BLOCK_HW: 16 if it divides hw exactly, otherwise the largest
    power-of-2 divisor. Falls back to 64 for very small hw.
    """
    if hw % 16 == 0:
        return 16
    b = 1
    while (b * 2) <= hw and (hw % (b * 2)) == 0:
        b *= 2
    return b if b >= 16 else 64


def _make_oracle():
    def _impl(inputs):
        mean, x, var, weight, bias, residual = inputs
        b = int(x.shape[0])
        c = int(x.shape[1])
        h = int(x.shape[2])
        w = int(x.shape[3])
        hw = h * w
        rows = b * c
        device = x.device

        # Preserve x's strides in the output.
        out = torch.empty_strided(
            tuple(x.shape), tuple(x.stride()),
            device=device, dtype=torch.bfloat16)

        # x is contiguous NCHW here (default). Reshape to [B*C, HW].
        x_2d = x.contiguous().view(rows, hw)
        residual_2d = residual.contiguous().view(rows, hw)
        BLOCK_HW = _pick_block_hw(hw)
        stream = torch.cuda.current_stream()
        if hw % BLOCK_HW == 0:
            # Exact tile fit — write straight to out_2d.
            out_2d = out.view(rows, hw)
            ct.launch(
                stream, (rows, hw // BLOCK_HW, 1), _bn_residual_relu_kernel,
                (mean, x_2d, var, weight, bias, residual_2d, out_2d,
                 hw, BLOCK_HW, c),
            )
        else:
            # Use a padded buffer.
            hw_pad = ((hw + BLOCK_HW - 1) // BLOCK_HW) * BLOCK_HW
            out_buf = torch.empty((rows, hw_pad), device=device, dtype=torch.bfloat16)
            ct.launch(
                stream, (rows, hw_pad // BLOCK_HW, 1), _bn_residual_relu_kernel,
                (mean, x_2d, var, weight, bias, residual_2d, out_buf,
                 hw, BLOCK_HW, c),
            )
            out.copy_(out_buf[:, :hw].view(b, c, h, w))
        return out
    return _impl


_forward = _make_oracle()


@oracle_impl(hardware="B200", point="85db042c")
@oracle_impl(hardware="B200", point="0af2d571")
@oracle_impl(hardware="B200", point="3aeeed95")
@oracle_impl(hardware="B200", point="fd6e1886")
@oracle_impl(hardware="B200", point="cd893d5b")
@oracle_impl(hardware="B200", point="28e6bebb")
@oracle_impl(hardware="B200", point="08581e62")
@oracle_impl(hardware="B200", point="50714806")
@oracle_impl(hardware="B200", point="4c0cf88f")
@oracle_impl(hardware="B200", point="6b748dff")
@oracle_impl(hardware="B200", point="7111f2e4")
@oracle_impl(hardware="B200", point="2e6c4615")
@oracle_impl(hardware="B200", point="e626f5a9")
@oracle_impl(hardware="B200", point="40170966")
@oracle_impl(hardware="B200", point="e2161da4")
@oracle_impl(hardware="B200", point="72f4709b")
@oracle_impl(hardware="B200", point="cce8fbb3")
@oracle_impl(hardware="B200", point="2f5dac5e")
@oracle_impl(hardware="B200", point="3bac8de4")
@oracle_impl(hardware="B200", point="592ca691")
@oracle_impl(hardware="B200", point="e20a55a6")
def oracle_forward(inputs):
    return _forward(inputs)
