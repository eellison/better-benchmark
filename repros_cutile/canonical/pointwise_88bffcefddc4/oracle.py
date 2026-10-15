"""cuTile port of pointwise_88bffcefddc4: MobileViT BN inference + SiLU.

Channels-last input; per-channel BN then SiLU. Handle non-power-of-2 channels
via ZERO-padded loads + masked scatter store. Input storage is [N, H, W, C]
contiguous (channels-last NCHW), so we treat it as [N*H*W, C].
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
def _bn_silu_kernel(
    x_ptr,       # bf16 [ROWS, C]
    mean_ptr,    # bf16 [C]
    var_ptr,     # bf16 [C]
    weight_ptr,  # bf16 [C]
    bias_ptr,    # bf16 [C]
    out_ptr,     # bf16 [ROWS, C]
    C: ct.Constant[int],
    C_PAD: ct.Constant[int],
    ROWS: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
):
    row_block = ct.bid(0)

    x = ct.load(x_ptr, index=(row_block, 0), shape=(BLOCK_R, C_PAD),
                padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(0,), shape=(C_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)
    var = ct.load(var_ptr, index=(0,), shape=(C_PAD,),
                  padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(0,), shape=(C_PAD,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(0,), shape=(C_PAD,),
                   padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    invstd = 1.0 / ct.sqrt(var_f + EPS)
    mean_2d = ct.reshape(mean_f, (1, C_PAD))
    invstd_2d = ct.reshape(invstd, (1, C_PAD))
    weight_2d = ct.reshape(weight_f, (1, C_PAD))
    bias_2d = ct.reshape(bias_f, (1, C_PAD))

    centered = x_f - mean_2d
    affine = centered * invstd_2d * weight_2d + bias_2d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf16, ct.float32)
    silu = affine_f / (ct.exp(-affine_f) + 1.0)
    silu_bf16 = ct.astype(silu, ct.bfloat16)

    # Guard OOB C and rows using scatter with a valid mask.
    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + row_block * BLOCK_R
    c_idx = ct.arange(C_PAD, dtype=ct.int32)
    r_mask = r_idx < ROWS
    c_mask = c_idx < C
    r_mask_2d = ct.reshape(r_mask, (BLOCK_R, 1))
    c_mask_2d = ct.reshape(c_mask, (1, C_PAD))
    valid = r_mask_2d & c_mask_2d
    r_bc = ct.broadcast_to(ct.reshape(r_idx, (BLOCK_R, 1)), (BLOCK_R, C_PAD))
    c_bc = ct.broadcast_to(ct.reshape(c_idx, (1, C_PAD)), (BLOCK_R, C_PAD))
    ct.scatter(out_ptr, (r_bc, c_bc), silu_bf16, mask=valid)


@oracle_impl(hardware="B200", point="d6d99242")
@oracle_impl(hardware="B200", point="ec18ed25")
@oracle_impl(hardware="B200", point="58353e1a")
@oracle_impl(hardware="B200", point="9f949812")
@oracle_impl(hardware="B200", point="9afa7692")
@oracle_impl(hardware="B200", point="a423f2c2")
@oracle_impl(hardware="B200", point="6129a191")
@oracle_impl(hardware="B200", point="39025e04")
@oracle_impl(hardware="B200", point="c4fb31ef")
@oracle_impl(hardware="B200", point="bf6f4282")
@oracle_impl(hardware="B200", point="c0e7b662")
@oracle_impl(hardware="B200", point="5e96e191")
@oracle_impl(hardware="B200", point="540fcc87")
@oracle_impl(hardware="B200", point="64799fb2")
@oracle_impl(hardware="B200", point="8162a5bc")
@oracle_impl(hardware="B200", point="d46c34b3")
@oracle_impl(hardware="B200", point="9ab2d895")
@oracle_impl(hardware="B200", point="51719261")
@oracle_impl(hardware="B200", point="9c97edfa")
def oracle_forward(inputs):
    mean, x, var, weight, bias = inputs
    n, c, h, w = x.shape
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )

    # View x (channels-last NCHW) as (n, h, w, c) contiguous = (n*h*w, c) contiguous.
    rows = n * h * w
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(rows, c)
    out_2d = out.permute(0, 2, 3, 1).view(rows, c)

    C_PAD = _next_pow2(c)
    BLOCK_R = 32
    grid = ((rows + BLOCK_R - 1) // BLOCK_R, 1, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        grid,
        _bn_silu_kernel,
        (x_2d, mean, var, weight, bias, out_2d, c, C_PAD, rows, BLOCK_R),
    )
    return out
