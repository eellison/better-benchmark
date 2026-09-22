"""cuTile port of pointwise_7ebac70550be: U-Net BN + ReLU + 2x2 maxpool.

Two kernels:
  1. Full-relu kernel: compute BN + ReLU for the entire input, store bf16
     into relu output.
  2. Pool kernel: for each 2x2 non-overlapping window in the output shape,
     max-pool the corresponding 4 values from relu.

Because W is odd (119, 239, 479, 959), we only pool over 2*out_h x 2*out_w
positions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    mean_ptr,
    x_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    relu_ptr,
    C: ct.Constant[int],
    H: ct.Constant[int],
    W: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    off = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = off < HW
    h = off // W
    w = off - h * W

    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)
    invstd = ct.rsqrt(var + 1.0e-5)

    zero_b = ct.full((BLOCK,), 0, dtype=ct.int32)
    c_t = ct.full((BLOCK,), c, dtype=ct.int32)
    x = ct.astype(
        ct.gather(x_ptr, (zero_b, c_t, h, w), mask=active,
                  padding_value=ct.bfloat16(0.0)),
        ct.float32,
    )
    centered = x - ct.broadcast_to(mean, (BLOCK,))
    scaled = centered * ct.broadcast_to(invstd, (BLOCK,)) * ct.broadcast_to(weight, (BLOCK,))
    affine = scaled + ct.broadcast_to(bias, (BLOCK,))
    rounded_bf = ct.astype(affine, ct.bfloat16)
    zero_v = ct.full((BLOCK,), 0.0, dtype=ct.bfloat16)
    # Preserve NaN: take value if x != x OR x > 0
    is_nan = rounded_bf != rounded_bf
    y = ct.where(is_nan | (rounded_bf > zero_v), rounded_bf, zero_v)

    ct.scatter(relu_ptr, (zero_b, c_t, h, w), y, mask=active)


@ct.kernel
def _pool_kernel(
    relu_ptr,      # bf16 [1, C, H, W]
    pool_ptr,      # bf16 [1, C, OUT_H, OUT_W]
    C: ct.Constant[int],
    W: ct.Constant[int],
    OUT_H_C: ct.Constant[int],
    OUT_W_C: ct.Constant[int],
    TOTAL: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    c = ct.bid(0)
    tile = ct.bid(1)
    linear = ct.arange(BLOCK, dtype=ct.int32) + tile * BLOCK
    active = linear < TOTAL

    ow = linear % OUT_W_C
    oh = linear // OUT_W_C

    in_h = oh * 2
    in_w = ow * 2

    zero_b = ct.full((BLOCK,), 0, dtype=ct.int32)
    c_t = ct.full((BLOCK,), c, dtype=ct.int32)

    x00 = ct.gather(relu_ptr, (zero_b, c_t, in_h, in_w), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x01 = ct.gather(relu_ptr, (zero_b, c_t, in_h, in_w + 1), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x10 = ct.gather(relu_ptr, (zero_b, c_t, in_h + 1, in_w), mask=active,
                    padding_value=ct.bfloat16(0.0))
    x11 = ct.gather(relu_ptr, (zero_b, c_t, in_h + 1, in_w + 1), mask=active,
                    padding_value=ct.bfloat16(0.0))

    m1 = ct.maximum(x00, x01)
    m2 = ct.maximum(x10, x11)
    m = ct.maximum(m1, m2)

    ct.scatter(pool_ptr, (zero_b, c_t, oh, ow), m, mask=active)


def _next_power_of_2(x):
    return 1 << (int(x) - 1).bit_length()


def _launch(inputs, *, C, H, W):
    mean, x, var, weight, bias, _shape0, _shape1 = inputs
    out_h = H // 2
    out_w = W // 2
    device = x.device
    relu = torch.empty_strided(
        (1, C, H, W), (C * H * W, H * W, W, 1),
        device=device, dtype=torch.bfloat16,
    )
    pool = torch.empty_strided(
        (1, C, out_h, out_w), (C * out_h * out_w, out_h * out_w, out_w, 1),
        device=device, dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()
    HW = H * W
    bn_block = min(_next_power_of_2(HW), 4096)
    bn_tiles = (HW + bn_block - 1) // bn_block
    ct.launch(
        stream, (C, bn_tiles, 1), _bn_relu_kernel,
        (mean, x, var, weight, bias, relu, C, H, W, HW, bn_block),
    )

    total_pool = out_h * out_w
    pool_block = min(_next_power_of_2(total_pool), 4096)
    pool_tiles = (total_pool + pool_block - 1) // pool_block
    ct.launch(
        stream, (C, pool_tiles, 1), _pool_kernel,
        (relu, pool, C, W, out_h, out_w, total_pool, pool_block),
    )
    return relu, pool


@oracle_impl(hardware="B200", point="3f17dc95", C=512, H=80, W=119)
@oracle_impl(hardware="B200", point="535abe76", C=256, H=160, W=239)
@oracle_impl(hardware="B200", point="0ce7c2f4", C=128, H=320, W=479)
@oracle_impl(hardware="B200", point="4f218228", C=64, H=640, W=959)
def oracle_forward(inputs, *, C, H, W):
    return _launch(inputs, C=C, H=H, W=W)
