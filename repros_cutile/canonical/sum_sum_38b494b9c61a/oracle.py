"""cuTile port of sum_sum_38b494b9c61a: MobileNetV3 BN + hard-swish spatial reduction.

Torch handles the elementwise producer chain (BN affine with bf16 rounding, hard-swish
with bf16 rounding, grad*hswish bf16 product) and the final per-channel reduction
epilogue; cuTile performs the per-(N, C) spatial reduction over the bf16 channels-last
product tensor viewed as (N, H*W, C) contiguous.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _spatial_sum_kernel(
    prod_ptr,   # bf16[N, HW, C] (channels-last permuted)
    out_ptr,    # f32[N, C]
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_tile = ct.bid(1)
    prod = ct.load(
        prod_ptr,
        index=(n, 0, c_tile),
        shape=(1, BLOCK_HW, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    prod_f = ct.astype(prod, ct.float32)
    s = ct.sum(prod_f, axis=1)  # -> (1, BLOCK_C)
    ct.store(out_ptr, index=(n, c_tile), tile=ct.reshape(s, (1, BLOCK_C)))


def _run(inputs, *, BLOCK_HW, BLOCK_C):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    N = int(arg1_1.shape[0])
    C = int(arg1_1.shape[1])
    H = int(arg1_1.shape[2])
    W = int(arg1_1.shape[3])
    HW = H * W
    device = arg1_1.device

    # ---- BN affine in f32, then bf16 round-trip (matches convert_element_type_1/2) ----
    x_f = arg1_1.to(torch.float32)
    sub = x_f - arg2_1
    mul = sub * arg3_1
    weight = arg4_1.view(1, C, 1, 1)
    bias = arg5_1.view(1, C, 1, 1)
    add = mul * weight + bias
    affine_bf16 = add.to(torch.bfloat16)
    out1 = affine_bf16.to(torch.float32)  # returned convert_element_type_2

    # ---- Hard-swish: uses the bf16-round-tripped affine (out1) ----
    add_1 = out1 + 3.0
    clamped = torch.clamp(torch.clamp(add_1, min=0.0), max=6.0)
    mul_2 = out1 * clamped
    div = mul_2 / 6.0
    hswish_bf16 = div.to(torch.bfloat16)

    # ---- Product: bf16 * bf16 -> bf16 (channels-last preserved) ----
    product_bf16 = arg6_1 * hswish_bf16

    # ---- cuTile spatial sum over H*W ----
    # For channels-last (N, C, H, W), permute(0, 2, 3, 1) yields (N, H, W, C)
    # in memory-contiguous order; view as (N, HW, C).
    product_nhwc = product_bf16.permute(0, 2, 3, 1).contiguous().view(N, HW, C)
    spatial_sum_f32 = torch.empty((N, C), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (N, ct.cdiv(C, BLOCK_C), 1),
        _spatial_sum_kernel,
        (product_nhwc, spatial_sum_f32, BLOCK_HW, BLOCK_C),
    )

    # ---- Post: bf16 boundary on the sum, scale, mask, cast, final channel reduction ----
    spatial_sum_bf16_f32 = spatial_sum_f32.to(torch.bfloat16).to(torch.float32)
    scaled = spatial_sum_bf16_f32 * 0.16666666666666666  # f32 (N, C)

    # out0 = arg0.to(f32), shape (N, C, 1, 1)
    out0 = arg0_1.to(torch.float32)

    in_range = (out0 > -3.0) & (out0 < 3.0)  # (N, C, 1, 1) bool
    masked = torch.where(in_range, scaled.view(N, C, 1, 1), arg7_1)  # f32
    out2 = masked.to(torch.bfloat16)  # bf16 (N, C, 1, 1)

    # out3: sum bf16 over [0, 2, 3] -> bf16[C], then to f32
    out3 = out2.sum(dim=[0, 2, 3]).to(torch.float32)

    return out0, out1, out2, out3


@oracle_impl(hardware="B200", point="6425bce8", BLOCK_HW=256, BLOCK_C=32, FINAL_BLOCK_C=64)
@oracle_impl(hardware="B200", point="64645353", BLOCK_HW=256, BLOCK_C=16, FINAL_BLOCK_C=64)
@oracle_impl(hardware="B200", point="e7e0ad11", BLOCK_HW=64, BLOCK_C=32, FINAL_BLOCK_C=64)
@oracle_impl(hardware="B200", point="239d1349", BLOCK_HW=64, BLOCK_C=32, FINAL_BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_HW, BLOCK_C, FINAL_BLOCK_C):
    del FINAL_BLOCK_C  # unused: final channel reduction runs in torch
    return _run(inputs, BLOCK_HW=BLOCK_HW, BLOCK_C=BLOCK_C)
