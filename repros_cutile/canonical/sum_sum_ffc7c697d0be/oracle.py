"""cuTile port of sum_sum_ffc7c697d0be: MobileNetV3 hardswish + squeeze-excite backward.

Matches Triton's 2-kernel structure:
  1. _spatial_hswish_gate_kernel: per (n, c_block), computes affine BN, h-swish,
     spatial sum-reduction over H*W, gate_f32 conversion, and gate_grad
     (predicate-selecting scaled_sum or scalar fallback). Writes affine_out,
     gate_f32_out, gate_grad_out.
  2. _final_channel_sum_kernel: per c_block, sums gate_grad_out along N ->
     channel_out (bf16-roundtripped f32).

Inputs are channels-last: NCHW view with strides encoding [N, H, W, C]. We
permute to NHWC for tile-friendly loads.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _spatial_hswish_gate_kernel(
    x_ptr,             # bf16 [N, HW, C]
    mean_ptr,          # f32  [C]
    invstd_ptr,        # f32  [C]
    weight_ptr,        # f32  [C]
    bias_ptr,          # f32  [C]
    grad_ptr,          # bf16 [N, HW, C]
    gate_ptr,          # bf16 [N, C]
    scalar_ptr,        # f32  [1]
    affine_out_ptr,    # f32  [N, HW, C]
    gate_f32_out_ptr,  # f32  [N, C]
    gate_grad_out_ptr, # bf16 [N, C]
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)

    # (1, BLOCK_HW, BLOCK_C) tile.
    x_bf = ct.load(x_ptr, index=(n, 0, c_block),
                   shape=(1, BLOCK_HW, BLOCK_C),
                   padding_mode=ct.PaddingMode.ZERO)
    grad_bf = ct.load(grad_ptr, index=(n, 0, c_block),
                      shape=(1, BLOCK_HW, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)
    invstd = ct.load(invstd_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,),
                   padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x_bf, ct.float32)
    grad_f = ct.astype(grad_bf, ct.float32)

    # Broadcast per-channel stats to (1, BLOCK_HW, BLOCK_C).
    mean_3d = ct.reshape(mean, (1, 1, BLOCK_C))
    invstd_3d = ct.reshape(invstd, (1, 1, BLOCK_C))
    weight_3d = ct.reshape(weight, (1, 1, BLOCK_C))
    bias_3d = ct.reshape(bias, (1, 1, BLOCK_C))

    centered = x_f - mean_3d
    normalized = centered * invstd_3d
    scaled = normalized * weight_3d
    affine = scaled + bias_3d
    affine_bf = ct.astype(affine, ct.bfloat16)
    affine_f32 = ct.astype(affine_bf, ct.float32)  # bf16 boundary before hswish
    ct.store(affine_out_ptr, index=(n, 0, c_block), tile=affine_f32)

    # H-swish gate = affine * clip(affine + 3, 0, 6) / 6
    hswish_arg = affine_f32 + 3.0
    hswish_min = ct.where(hswish_arg < 0.0, 0.0, hswish_arg)
    hswish_clamped = ct.where(hswish_min > 6.0, 6.0, hswish_min)
    hswish_mul = affine_f32 * hswish_clamped
    hswish_bf = ct.astype(hswish_mul * (1.0 / 6.0), ct.bfloat16)
    product_bf = ct.astype(
        grad_f * ct.astype(hswish_bf, ct.float32),
        ct.bfloat16,
    )

    # Spatial sum over BLOCK_HW dim (axis=1) with masking for OOB.
    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32)
    hw_valid = hw_idx < HW  # (BLOCK_HW,)
    hw_valid_3d = ct.reshape(hw_valid, (1, BLOCK_HW, 1))
    zero_3d = ct.zeros((1, BLOCK_HW, BLOCK_C), dtype=ct.float32)
    product_f = ct.astype(product_bf, ct.float32)
    product_masked = ct.where(hw_valid_3d, product_f, zero_3d)
    spatial_sum = ct.sum(product_masked, axis=1)  # (1, BLOCK_C)
    spatial_sum_f = ct.astype(ct.astype(spatial_sum, ct.bfloat16), ct.float32)

    # gate_f32 = gate.bf16.f32
    gate_bf = ct.load(gate_ptr, index=(n, c_block),
                      shape=(1, BLOCK_C),
                      padding_mode=ct.PaddingMode.ZERO)
    gate_f32 = ct.astype(gate_bf, ct.float32)
    ct.store(gate_f32_out_ptr, index=(n, c_block), tile=gate_f32)

    # gate_grad = where((gate > -3) & (gate < 3), spatial_sum * 1/6, scalar)
    active = (gate_f32 > -3.0) & (gate_f32 < 3.0)
    scalar_val = ct.load(scalar_ptr, index=(0,), shape=(1,))
    scalar_broad = ct.astype(scalar_val, ct.float32)
    scalar_2d = ct.reshape(scalar_broad, (1, 1))
    ones_2d = ct.full((1, BLOCK_C), 1.0, dtype=ct.float32)
    scalar_full = ones_2d * scalar_2d
    scaled_sum = spatial_sum_f * (1.0 / 6.0)
    gate_grad = ct.where(active, scaled_sum, scalar_full)
    gate_grad_bf = ct.astype(gate_grad, ct.bfloat16)
    ct.store(gate_grad_out_ptr, index=(n, c_block), tile=gate_grad_bf)


@ct.kernel
def _final_channel_sum_kernel(
    gate_grad_ptr,  # bf16 [N, C]
    out_ptr,        # f32  [C]
    N_ROWS: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    tile = ct.load(
        gate_grad_ptr, index=(0, c_block), shape=(BLOCK_N, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    tile_f = ct.astype(tile, ct.float32)
    # Mask rows > N_ROWS.
    row_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    row_valid = row_idx < N_ROWS
    row_valid_2d = ct.reshape(row_valid, (BLOCK_N, 1))
    zero_2d = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.float32)
    tile_masked = ct.where(row_valid_2d, tile_f, zero_2d)
    reduced = ct.sum(tile_masked, axis=0)  # (BLOCK_C,)
    reduced_bf = ct.astype(reduced, ct.bfloat16)
    reduced_f = ct.astype(reduced_bf, ct.float32)
    ct.store(out_ptr, index=(c_block,), tile=reduced_f)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@oracle_impl(hardware="B200", point="2d9d7e80", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=32)
@oracle_impl(hardware="B200", point="ffea7dda", BLOCK_C=16, BLOCK_HW=256, FINAL_BLOCK_C=32)
@oracle_impl(hardware="B200", point="3bcfd222", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64)
@oracle_impl(hardware="B200", point="b1ab7efd", BLOCK_C=32, BLOCK_HW=64, FINAL_BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_C, BLOCK_HW, FINAL_BLOCK_C):
    x, mean, invstd, weight, bias, grad, gate, scalar = inputs
    n, c, h, w = (int(dim) for dim in x.shape)
    hw = h * w
    device = x.device

    # Reshape to NHWC layout (channels-last is already contiguous under permute).
    x_nhwc = x.permute(0, 2, 3, 1).reshape(n, hw, c)
    grad_nhwc = grad.permute(0, 2, 3, 1).reshape(n, hw, c)
    gate_nc = gate.view(n, c)
    mean_c = mean.view(c)
    invstd_c = invstd.view(c)
    scalar_1 = scalar.view(1)

    # Output allocations (channels-last strides for affine_out).
    affine_out_nhwc = torch.empty((n, hw, c), device=device, dtype=torch.float32)
    gate_f32_out_nc = torch.empty((n, c), device=device, dtype=torch.float32)
    gate_grad_out_nc = torch.empty((n, c), device=device, dtype=torch.bfloat16)
    channel_out = torch.empty((c,), device=device, dtype=torch.float32)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (n, (c + BLOCK_C - 1) // BLOCK_C, 1),
        _spatial_hswish_gate_kernel,
        (x_nhwc, mean_c, invstd_c, weight, bias, grad_nhwc, gate_nc, scalar_1,
         affine_out_nhwc, gate_f32_out_nc, gate_grad_out_nc,
         hw, BLOCK_HW, BLOCK_C),
    )

    block_n = _next_power_of_2(n)
    ct.launch(
        stream, ((c + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C, 1, 1),
        _final_channel_sum_kernel,
        (gate_grad_out_nc, channel_out, n, block_n, FINAL_BLOCK_C),
    )

    # Reshape outputs back to (N, C, H, W)/(N, C, 1, 1) with channels-last strides.
    # affine_out expected: same shape/stride as x (channels-last).
    affine_out = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()),
        device=device, dtype=torch.float32,
    )
    affine_out.copy_(affine_out_nhwc.view(n, h, w, c).permute(0, 3, 1, 2))

    gate_f32_out = torch.empty_strided(
        tuple(gate.shape), tuple(gate.stride()),
        device=device, dtype=torch.float32,
    )
    gate_f32_out.copy_(gate_f32_out_nc.view(n, c, 1, 1))

    gate_grad_out = torch.empty_strided(
        tuple(gate.shape), tuple(gate.stride()),
        device=device, dtype=torch.bfloat16,
    )
    gate_grad_out.copy_(gate_grad_out_nc.view(n, c, 1, 1))

    return affine_out, gate_f32_out, gate_grad_out, channel_out
