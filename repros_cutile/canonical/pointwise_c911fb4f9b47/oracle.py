"""cuTile port of pointwise_c911fb4f9b47: DenseNet channel-cat + BN + ReLU.

The Triton kernel does a single fused kernel over the concatenated [B, C_TOTAL, H, W]
where each element pulls from x0 (channels < 16) or x1 (channels >= 16) and applies
BN affine + ReLU. cuTile's tile-space model works better with two kernels — one over
the x0 slice, one over the x1 slice — sharing the same tile logic.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _cat_bn_relu_kernel(
    x_ptr,           # bf16 [B, C_LOCAL, H, W]
    mean_ptr,        # bf16 [C_LOCAL] (already sliced)
    var_ptr,
    weight_ptr,
    bias_ptr,
    cat_ptr,         # bf16 view of cat[:, offset:offset+C_LOCAL, :, :]
    relu_ptr,        # bf16 view of relu[:, offset:offset+C_LOCAL, :, :]
    H_DIM: ct.Constant[int],
    W_DIM: ct.Constant[int],
):
    b = ct.bid(0)
    c = ct.bid(1)

    x = ct.load(x_ptr, index=(b, c, 0, 0), shape=(1, 1, H_DIM, W_DIM))
    x_f = ct.astype(x, ct.float32)

    mean = ct.astype(ct.load(mean_ptr, index=(c,), shape=(1,)), ct.float32)
    var = ct.astype(ct.load(var_ptr, index=(c,), shape=(1,)), ct.float32)
    weight = ct.astype(ct.load(weight_ptr, index=(c,), shape=(1,)), ct.float32)
    bias = ct.astype(ct.load(bias_ptr, index=(c,), shape=(1,)), ct.float32)

    mean_bc = ct.reshape(mean, (1, 1, 1, 1))
    var_bc = ct.reshape(var, (1, 1, 1, 1))
    weight_bc = ct.reshape(weight, (1, 1, 1, 1))
    bias_bc = ct.reshape(bias, (1, 1, 1, 1))

    inv_std_bc = ct.rsqrt(var_bc + 1.0e-5)
    affine_f32 = (x_f - mean_bc) * inv_std_bc * weight_bc + bias_bc

    # ReLU with NaN preservation, matching eager's aten.relu on bf16.
    zero_tile = ct.zeros(shape=(1, 1, H_DIM, W_DIM), dtype=ct.float32)
    is_nan = affine_f32 != affine_f32
    positive = affine_f32 > zero_tile
    relu_f32 = ct.where(is_nan | positive, affine_f32, zero_tile)
    relu_bf16 = ct.astype(relu_f32, ct.bfloat16)

    ct.store(cat_ptr, index=(b, c, 0, 0), tile=x)
    ct.store(relu_ptr, index=(b, c, 0, 0), tile=relu_bf16)


@oracle_impl(hardware="B200", point="286b5304", BLOCK=256)
@oracle_impl(hardware="B200", point="c7103f45", BLOCK=256)
@oracle_impl(hardware="B200", point="d46fd5bf", BLOCK=256)
@oracle_impl(hardware="B200", point="bdfe0800", BLOCK=256)
@oracle_impl(hardware="B200", point="3ba43305", BLOCK=256)
@oracle_impl(hardware="B200", point="ca621cc5", BLOCK=256)
@oracle_impl(hardware="B200", point="02da0370", BLOCK=256)
@oracle_impl(hardware="B200", point="5ec16488", BLOCK=256)
@oracle_impl(hardware="B200", point="1f781a5e", BLOCK=256)
@oracle_impl(hardware="B200", point="fe2f0933", BLOCK=256)
@oracle_impl(hardware="B200", point="b1d4084b", BLOCK=256)
@oracle_impl(hardware="B200", point="e08032de", BLOCK=256)
@oracle_impl(hardware="B200", point="1b282c91", BLOCK=256)
@oracle_impl(hardware="B200", point="7a77a894", BLOCK=256)
@oracle_impl(hardware="B200", point="5911c005", BLOCK=256)
@oracle_impl(hardware="B200", point="519fe36e", BLOCK=256)
@oracle_impl(hardware="B200", point="d64fa266", BLOCK=256)
@oracle_impl(hardware="B200", point="3dde721a", BLOCK=256)
@oracle_impl(hardware="B200", point="d99ca30a", BLOCK=256)
@oracle_impl(hardware="B200", point="9b5495c2", BLOCK=256)
def oracle_forward(inputs, **_kwargs):
    x0, x1, mean, var, weight, bias = inputs
    B = int(x0.shape[0])
    C0 = int(x0.shape[1])
    C1 = int(x1.shape[1])
    H = int(x0.shape[2])
    W = int(x0.shape[3])
    C_TOTAL = C0 + C1
    hw = H * W

    cat = torch.empty_strided(
        (B, C_TOTAL, H, W),
        (C_TOTAL * hw, hw, W, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    relu = torch.empty_strided(
        (B, C_TOTAL, H, W),
        (C_TOTAL * hw, hw, W, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    stream = torch.cuda.current_stream()

    # First segment: channels [0, C0) from x0.
    cat0 = cat.narrow(1, 0, C0)
    relu0 = relu.narrow(1, 0, C0)
    mean0 = mean.narrow(0, 0, C0)
    var0 = var.narrow(0, 0, C0)
    weight0 = weight.narrow(0, 0, C0)
    bias0 = bias.narrow(0, 0, C0)
    ct.launch(
        stream,
        (B, C0, 1),
        _cat_bn_relu_kernel,
        (x0, mean0, var0, weight0, bias0, cat0, relu0, H, W),
    )

    # Second segment: channels [C0, C0+C1) from x1.
    cat1 = cat.narrow(1, C0, C1)
    relu1 = relu.narrow(1, C0, C1)
    mean1 = mean.narrow(0, C0, C1)
    var1 = var.narrow(0, C0, C1)
    weight1 = weight.narrow(0, C0, C1)
    bias1 = bias.narrow(0, C0, C1)
    ct.launch(
        stream,
        (B, C1, 1),
        _cat_bn_relu_kernel,
        (x1, mean1, var1, weight1, bias1, cat1, relu1, H, W),
    )

    return cat, relu
