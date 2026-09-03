"""cuTile port of pointwise_39eaa1204ef2: VisFormer 2-stage BN affine + residual.

For each spatial row (N*H*W), apply BN affine with (mean1, invstd1, w1, b1),
add residual in bf16 (rounded), apply BN affine again with (mean2, invstd2,
w2, b2), and store the final bf16 output.

Inputs are NCHW with channels-last strides — we view them as NHWC contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _invstd_kernel(
    var1_ptr, var2_ptr, inv1_ptr, inv2_ptr,
    C_LIM: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_tile = ct.bid(0)
    var1 = ct.astype(
        ct.load(var1_ptr, index=(c_tile,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    var2 = ct.astype(
        ct.load(var2_ptr, index=(c_tile,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    eps = ct.full(shape=(BLOCK_C,), fill_value=EPS, dtype=ct.float32)
    inv1 = ct.rsqrt(var1 + eps)
    inv2 = ct.rsqrt(var2 + eps)
    ct.store(inv1_ptr, index=(c_tile,), tile=inv1)
    ct.store(inv2_ptr, index=(c_tile,), tile=inv2)


@ct.kernel
def _dual_norm_kernel(
    mean1_ptr,     # bf16 [C]
    x_ptr,         # bf16 [ROWS, C]  (NHWC layout)
    inv1_ptr,      # f32 [C]
    inv2_ptr,      # f32 [C]
    weight1_ptr,   # bf16 [C]
    bias1_ptr,     # bf16 [C]
    residual_ptr,  # bf16 [ROWS, C]  (broadcasted; each residual row same across N)
    mean2_ptr,     # bf16 [C]
    weight2_ptr,   # bf16 [C]
    bias2_ptr,     # bf16 [C]
    out1_ptr,      # bf16 [ROWS, C]
    out2_ptr,      # bf16 [ROWS, C]
    BLOCK_ROWS: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    row_tile = ct.bid(0)
    x = ct.astype(
        ct.load(x_ptr, index=(row_tile, 0),
                shape=(BLOCK_ROWS, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    residual = ct.astype(
        ct.load(residual_ptr, index=(row_tile, 0),
                shape=(BLOCK_ROWS, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean1 = ct.load(mean1_ptr, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    mean1_f = ct.astype(mean1, ct.float32)
    inv1 = ct.load(inv1_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight1 = ct.astype(
        ct.load(weight1_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias1 = ct.astype(
        ct.load(bias1_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    mean2 = ct.load(mean2_ptr, index=(0,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    mean2_f = ct.astype(mean2, ct.float32)
    inv2 = ct.load(inv2_ptr, index=(0,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    weight2 = ct.astype(
        ct.load(weight2_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )
    bias2 = ct.astype(
        ct.load(bias2_ptr, index=(0,), shape=(BLOCK_C,),
                padding_mode=ct.PaddingMode.ZERO),
        ct.float32,
    )

    mean1_2d = ct.reshape(mean1_f, (1, BLOCK_C))
    inv1_2d = ct.reshape(inv1, (1, BLOCK_C))
    weight1_2d = ct.reshape(weight1, (1, BLOCK_C))
    bias1_2d = ct.reshape(bias1, (1, BLOCK_C))
    mean2_2d = ct.reshape(mean2_f, (1, BLOCK_C))
    inv2_2d = ct.reshape(inv2, (1, BLOCK_C))
    weight2_2d = ct.reshape(weight2, (1, BLOCK_C))
    bias2_2d = ct.reshape(bias2, (1, BLOCK_C))

    stage1 = (x - mean1_2d) * inv1_2d * weight1_2d + bias1_2d
    stage1_bf16 = ct.astype(stage1, ct.bfloat16)
    add_residual = ct.astype(ct.astype(stage1_bf16, ct.float32) + residual,
                             ct.bfloat16)
    add_res_f = ct.astype(add_residual, ct.float32)
    stage2 = (add_res_f - mean2_2d) * inv2_2d * weight2_2d + bias2_2d
    ct.store(out1_ptr, index=(row_tile, 0), tile=add_residual)
    ct.store(out2_ptr, index=(row_tile, 0), tile=ct.astype(stage2, ct.bfloat16))


def _channels_last_stride(n, c, h, w):
    return (c * h * w, 1, c * w, c)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="97d7b07b")
@oracle_impl(hardware="B200", point="bd2d0775")
@oracle_impl(hardware="B200", point="f1b0313e")
def oracle_forward(inputs):
    mean1, x, var1, weight1, bias1, residual, mean2, var2, weight2, bias2 = inputs
    n, c, h, w = x.shape
    hw = h * w
    total_rows = n * hw
    BLOCK_C = _next_pow2(c)
    BLOCK_ROWS = 32  # tile 32 rows at a time

    inv1 = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    inv2 = torch.empty_strided((c,), (1,), device=x.device, dtype=torch.float32)
    out1 = torch.empty_strided(
        (n, c, h, w), _channels_last_stride(n, c, h, w),
        device=x.device, dtype=torch.bfloat16,
    )
    out2 = torch.empty_strided(
        (n, c, h, w), _channels_last_stride(n, c, h, w),
        device=x.device, dtype=torch.bfloat16,
    )
    # NHWC views (channels-last means .permute(0,2,3,1) is contiguous)
    x_nhwc = x.permute(0, 2, 3, 1).view(total_rows, c)
    # residual is [1, C, H, W] channels-last; permute to [1, H, W, C]
    residual_nhwc = residual.permute(0, 2, 3, 1).contiguous().view(hw, c)
    # Broadcast residual across N by tiling
    residual_expanded = residual_nhwc.unsqueeze(0).expand(n, hw, c).reshape(total_rows, c)
    out1_nhwc = out1.permute(0, 2, 3, 1).view(total_rows, c)
    out2_nhwc = out2.permute(0, 2, 3, 1).view(total_rows, c)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (1, 1, 1), _invstd_kernel,
        (var1, var2, inv1, inv2, c, BLOCK_C),
    )
    ct.launch(
        stream, (total_rows // BLOCK_ROWS, 1, 1), _dual_norm_kernel,
        (mean1, x_nhwc, inv1, inv2, weight1, bias1,
         residual_expanded, mean2, weight2, bias2,
         out1_nhwc, out2_nhwc, BLOCK_ROWS, BLOCK_C),
    )
    return out1, out2
