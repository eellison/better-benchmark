"""cuTile port of pointwise_ba25986618ef: ShuffleNet BN + ReLU + cat + channel shuffle."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


def _next_pow2(v: int) -> int:
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_relu_kernel(
    x_ptr,        # bf16 [rows, C]
    mean_ptr,     # bf16 [C]
    var_ptr,      # bf16 [C]
    weight_ptr,   # bf16 [C]
    bias_ptr,     # bf16 [C]
    out_ptr,      # bf16 [rows, C]
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
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f = ct.astype(affine_bf, ct.float32)
    # NaN-preserving ReLU: keep NaN, else max(0, x).
    is_nan = affine_f != affine_f
    relu = ct.where((affine_f > 0.0) | is_nan, affine_f, 0.0)
    relu_bf = ct.astype(relu, ct.bfloat16)

    r_idx = ct.arange(BLOCK_R, dtype=ct.int32) + row_block * BLOCK_R
    c_idx = ct.arange(C_PAD, dtype=ct.int32)
    r_mask = r_idx < ROWS
    c_mask = c_idx < C
    r_mask_2d = ct.reshape(r_mask, (BLOCK_R, 1))
    c_mask_2d = ct.reshape(c_mask, (1, C_PAD))
    valid = r_mask_2d & c_mask_2d
    r_bc = ct.broadcast_to(ct.reshape(r_idx, (BLOCK_R, 1)), (BLOCK_R, C_PAD))
    c_bc = ct.broadcast_to(ct.reshape(c_idx, (1, C_PAD)), (BLOCK_R, C_PAD))
    ct.scatter(out_ptr, (r_bc, c_bc), relu_bf, mask=valid)


@oracle_impl(hardware="B200", point="ef47f5e1", BLOCK_C=64, BLOCK_OUT=512)
def oracle_forward(inputs, *, BLOCK_C=None, BLOCK_OUT=None):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, shape0, shape1 = inputs
    device = arg1_1.device
    n, c, h, w = arg1_1.shape
    rows = n * h * w

    x_2d = arg1_1.permute(0, 2, 3, 1).contiguous().view(rows, c)
    relu_2d = torch.empty((rows, c), device=device, dtype=torch.bfloat16)

    C_PAD = _next_pow2(c)
    BLOCK_R = 32
    grid = ((rows + BLOCK_R - 1) // BLOCK_R, 1, 1)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, grid, _bn_relu_kernel,
        (x_2d, arg0_1, arg2_1, arg3_1, arg4_1, relu_2d,
         c, C_PAD, rows, BLOCK_R),
    )
    relu = relu_2d.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    # Channel-shuffle: cat, view as (N, 2, C, H, W), permute [0, 2, 1, 3, 4], clone,
    # then view back and split.
    cat = torch.cat([arg5_1, relu], dim=1)
    view = cat.view(n, 2, c, h, w)
    permute = view.permute(0, 2, 1, 3, 4)
    clone = permute.contiguous()
    view_1 = clone.view(n, 2 * c, h, w)
    split = torch.split(view_1, c, dim=1)
    return split[0], split[1]
