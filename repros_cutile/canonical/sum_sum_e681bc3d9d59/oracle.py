"""cuTile port of sum_sum_e681bc3d9d59: MobileNetV3 BN + ReLU + spatial reduce + channel sum.

Per (n, c) program:
1. Load activations[n, c, :HW] and grad[n, c, :HW] (channels-last input)
2. Compute affine, relu, product = grad*relu, spatial sum, gate the sum
3. Store relu output, gate output, sum output

Second pass: reduce gate output across N to produce channel_out.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


def _next_pow2(v):
    r = 1
    while r < v:
        r <<= 1
    return r


@ct.kernel
def _bn_relu_reduce_kernel(
    gate_orig_arr,   # bf16 [N, C]
    x_arr,           # bf16 [N, HW, C]  (channels-last view)
    mean_arr,        # f32 [C]
    invstd_arr,      # f32 [C]
    weight_arr,      # f32 [C]
    bias_arr,        # f32 [C]
    grad_arr,        # bf16 [N, HW, C]
    fill_arr,        # f32 scalar
    out0_arr,        # f32 [N, C]  (copy of gate_orig)
    relu_out_arr,    # bf16 [N, HW, C]
    gate_out_arr,    # bf16 [N, C]
    HW: ct.Constant[int],
    HW_PAD: ct.Constant[int],
    C: ct.Constant[int],
):
    n = ct.bid(0)
    c = ct.bid(1)

    # Load full spatial region for channel c of batch n: HW elements
    x = ct.load(x_arr, index=(n, 0, c), shape=(1, HW_PAD, 1),
                padding_mode=ct.PaddingMode.ZERO)
    grad = ct.load(grad_arr, index=(n, 0, c), shape=(1, HW_PAD, 1),
                   padding_mode=ct.PaddingMode.ZERO)

    x_f = ct.astype(x, ct.float32)
    grad_f = ct.astype(grad, ct.float32)
    mean = ct.load(mean_arr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_arr, index=(c,), shape=(1,))
    weight = ct.load(weight_arr, index=(c,), shape=(1,))
    bias = ct.load(bias_arr, index=(c,), shape=(1,))

    mean_3d = ct.reshape(mean, (1, 1, 1))
    invstd_3d = ct.reshape(invstd, (1, 1, 1))
    weight_3d = ct.reshape(weight, (1, 1, 1))
    bias_3d = ct.reshape(bias, (1, 1, 1))

    centered = x_f - mean_3d
    affine = (centered * invstd_3d) * weight_3d + bias_3d
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_bf16_f = ct.astype(affine_bf16, ct.float32)
    zero = ct.full(shape=(1, HW_PAD, 1), fill_value=0.0, dtype=ct.float32)
    relu = ct.where(affine_bf16_f > 0.0, affine_bf16_f, zero)
    relu_bf16 = ct.astype(relu, ct.bfloat16)

    # Store relu (only valid HW positions)
    hw_idx = ct.arange(HW_PAD, dtype=ct.int32)
    hw_valid = hw_idx < HW
    hw_valid_3d = ct.reshape(hw_valid, (1, HW_PAD, 1))
    n_bc = ct.full(shape=(1, HW_PAD, 1), fill_value=n, dtype=ct.int32)
    hw_bc = ct.reshape(hw_idx, (1, HW_PAD, 1))
    c_bc = ct.full(shape=(1, HW_PAD, 1), fill_value=c, dtype=ct.int32)
    ct.scatter(relu_out_arr, (n_bc, hw_bc, c_bc), relu_bf16, mask=hw_valid_3d)

    # Product bf16, sum over spatial (masked)
    product = grad_f * relu
    product_bf16 = ct.astype(product, ct.bfloat16)
    product_bf16_f = ct.astype(product_bf16, ct.float32)
    product_masked = ct.where(hw_valid_3d, product_bf16_f, zero)
    spatial_sum = ct.sum(product_masked)
    spatial_sum_bf16 = ct.astype(spatial_sum, ct.bfloat16)
    spatial_sum_bf16_f = ct.astype(spatial_sum_bf16, ct.float32)
    scaled_sum = spatial_sum_bf16_f * 0.16666666666666666

    # Load gate (bf16), copy to out0 as f32
    gate = ct.load(gate_orig_arr, index=(n, c), shape=(1, 1))
    gate_f = ct.astype(gate, ct.float32)
    ct.store(out0_arr, index=(n, c), tile=gate_f)

    # Load fill scalar
    fill = ct.load(fill_arr, index=(0,), shape=(1,))
    fill_f = ct.astype(fill, ct.float32)
    fill_scalar = ct.reshape(fill_f, (1, 1))

    gate_scalar = gate_f  # (1, 1)
    scaled_sum_2d = ct.reshape(scaled_sum, (1, 1))
    condition = (gate_scalar > -3.0) & (gate_scalar < 3.0)
    gated = ct.where(condition, scaled_sum_2d, fill_scalar)
    gated_bf16 = ct.astype(gated, ct.bfloat16)
    ct.store(gate_out_arr, index=(n, c), tile=gated_bf16)


@ct.kernel
def _final_channel_sum_kernel(
    gate_out_ptr,   # bf16 [N, C]
    channel_out_ptr,  # f32 [C]
    C: ct.Constant[int],
    N: ct.Constant[int],
    BLOCK_N: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    c_block = ct.bid(0)
    # Load [BLOCK_N, BLOCK_C] tile from gate_out; OOB in C zero-padded.
    gate_bf = ct.load(
        gate_out_ptr,
        index=(0, c_block),
        shape=(BLOCK_N, BLOCK_C),
        padding_mode=ct.PaddingMode.ZERO,
    )
    gate_f = ct.astype(gate_bf, ct.float32)

    # Mask lanes with n >= N (BLOCK_N=32 covers full N=32 in our shapes,
    # but keep the guard for safety).
    n_idx = ct.arange(BLOCK_N, dtype=ct.int32)
    n_valid = n_idx < N
    n_valid_2d = ct.reshape(n_valid, (BLOCK_N, 1))
    zero_2d = ct.zeros((BLOCK_N, BLOCK_C), dtype=ct.float32)
    gate_masked = ct.where(n_valid_2d, gate_f, zero_2d)

    reduced = ct.sum(gate_masked, axis=0)  # (BLOCK_C,)
    reduced_bf16 = ct.astype(reduced, ct.bfloat16)
    reduced_back = ct.astype(reduced_bf16, ct.float32)  # rtne(bf16(x))

    # Store only valid C positions via scatter.
    c_local = ct.arange(BLOCK_C, dtype=ct.int32)
    c_idx = c_block * BLOCK_C + c_local
    c_valid = c_idx < C
    ct.scatter(channel_out_ptr, (c_idx,), reduced_back, mask=c_valid)


@oracle_impl(hardware="B200", point="c557d2c1", BLOCK_N=32, FINAL_BLOCK_C=64)
@oracle_impl(hardware="B200", point="0e83eb42", BLOCK_N=32, FINAL_BLOCK_C=64)
def oracle_forward(inputs, *, BLOCK_N: int, FINAL_BLOCK_C: int):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 = inputs
    n = int(arg1.shape[0])
    c = int(arg1.shape[1])
    h = int(arg1.shape[2])
    w = int(arg1.shape[3])
    hw = h * w

    out0 = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1.device,
        dtype=torch.float32,
    )
    relu_out = torch.empty_strided(
        (n, c, h, w),
        (c * hw, 1, c * w, c),
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    gate_out = torch.empty_strided(
        (n, c, 1, 1),
        (c, 1, 1, 1),
        device=arg1.device,
        dtype=torch.bfloat16,
    )
    channel_out = torch.empty((c,), device=arg1.device, dtype=torch.float32)

    # Channels-last view: [N, HW, C]
    x_nhwc = arg1.permute(0, 2, 3, 1).contiguous().view(n, hw, c)
    grad_nhwc = arg6.permute(0, 2, 3, 1).contiguous().view(n, hw, c)
    relu_nhwc = relu_out.permute(0, 2, 3, 1).view(n, hw, c)
    gate_2d = arg0.view(n, c)
    gate_out_2d = gate_out.view(n, c)
    out0_2d = out0.view(n, c)
    mean_1d = arg2.view(c)
    invstd_1d = arg3.view(c)
    weight_1d = arg4.view(c)
    bias_1d = arg5.view(c)
    fill_1d = arg7.view(1)

    HW_PAD = _next_pow2(hw)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (n, c, 1),
        _bn_relu_reduce_kernel,
        (gate_2d, x_nhwc, mean_1d, invstd_1d, weight_1d, bias_1d, grad_nhwc, fill_1d,
         out0_2d, relu_nhwc, gate_out_2d,
         hw, HW_PAD, c),
    )

    # Channel sum: reduce gate_out (bf16) across N in a cuTile kernel.
    num_c_blocks = (c + FINAL_BLOCK_C - 1) // FINAL_BLOCK_C
    ct.launch(
        stream,
        (num_c_blocks, 1, 1),
        _final_channel_sum_kernel,
        (gate_out_2d, channel_out, c, n, BLOCK_N, FINAL_BLOCK_C),
    )
    return out0, relu_out, gate_out, channel_out
