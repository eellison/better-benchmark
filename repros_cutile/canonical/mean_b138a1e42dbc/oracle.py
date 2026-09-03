"""cuTile port of mean_b138a1e42dbc: BN-affine residual-ReLU spatial-mean fused."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_residual_relu_mean_kernel(
    mean_ptr,       # f32 (C,)
    x_ptr,          # bf16 (N, C, HW)
    var_ptr,        # bf16 (C,)
    weight_ptr,     # bf16 (C,)
    bias_ptr,       # bf16 (C,)
    residual_ptr,   # bf16 (N, C, HW)
    out_ptr,        # bf16 (N, C)
    HW_INV: ct.Constant[float],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    x_tile = ct.load(x_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
                     padding_mode=ct.PaddingMode.ZERO)
    resid_tile = ct.load(residual_ptr, index=(n, c, 0), shape=(1, 1, BLOCK_HW),
                         padding_mode=ct.PaddingMode.ZERO)
    m_scalar = ct.load(mean_ptr, index=(c,), shape=(1,))
    v_scalar = ct.load(var_ptr, index=(c,), shape=(1,))
    w_scalar = ct.load(weight_ptr, index=(c,), shape=(1,))
    b_scalar = ct.load(bias_ptr, index=(c,), shape=(1,))

    m_f = ct.astype(m_scalar, ct.float32)
    v_f = ct.astype(v_scalar, ct.float32)
    w_f = ct.astype(w_scalar, ct.float32)
    b_f = ct.astype(b_scalar, ct.float32)

    invstd = ct.rsqrt(v_f + 1.0e-5)
    scale = invstd * w_f
    shift = b_f - m_f * scale

    x_f = ct.astype(x_tile, ct.float32)
    shifted = x_f * ct.reshape(scale, (1, 1, 1)) + ct.reshape(shift, (1, 1, 1))
    affine_bf = ct.astype(shifted, ct.bfloat16)
    added_bf = ct.astype(ct.astype(affine_bf, ct.float32) + ct.astype(resid_tile, ct.float32), ct.bfloat16)
    zero = ct.zeros(shape=(1, 1, BLOCK_HW), dtype=ct.bfloat16)
    relu_bf = ct.where(added_bf < zero, zero, added_bf)

    relu_f = ct.astype(relu_bf, ct.float32)
    # Mask out padded positions (where the tile extends past HW valid elements).
    idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    valid_1d = idx < HW
    valid = ct.reshape(valid_1d, (1, 1, BLOCK_HW))
    zero_f = ct.zeros(shape=(1, 1, BLOCK_HW), dtype=ct.float32)
    relu_valid = ct.where(valid, relu_f, zero_f)
    total = ct.sum(relu_valid)
    mean_out = total * HW_INV
    ct.store(out_ptr, index=(n, c), tile=ct.reshape(ct.astype(mean_out, ct.bfloat16), (1, 1)))


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


@oracle_impl(hardware="B200", point="187a2d53", BLOCK_HW=64)
@oracle_impl(hardware="B200", point="08581e62", BLOCK_HW=64)
@oracle_impl(hardware="B200", point="7111f2e4", BLOCK_HW=64)
@oracle_impl(hardware="B200", point="e2161da4", BLOCK_HW=64)
@oracle_impl(hardware="B200", point="2f5dac5e", BLOCK_HW=64)
def oracle_forward(inputs, *, BLOCK_HW: int):
    mean, x, var, weight, bias, residual, shape = inputs
    batch = int(x.shape[0])
    channels = int(x.shape[1])
    height = int(x.shape[2])
    width = int(x.shape[3])
    hw = height * width
    out_shape = tuple(int(d) for d in shape)
    output = torch.empty_strided(
        out_shape,
        (channels, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    # Flatten spatial dims to (N, C, HW)
    x3 = x.view(batch, channels, hw)
    r3 = residual.view(batch, channels, hw)

    # Block must be power-of-2 >= HW (or divide HW); use next_pow2(hw) with padding.
    block = _next_pow2(hw)
    stream = torch.cuda.current_stream()
    ct.launch(stream, (batch, channels, 1), _bn_residual_relu_mean_kernel,
              (mean, x3, var, weight, bias, r3, output, 1.0 / hw, hw, block))
    return output
