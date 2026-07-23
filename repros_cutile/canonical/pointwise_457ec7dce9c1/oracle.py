"""cuTile port of pointwise_457ec7dce9c1: UNet BN+relu on bf16 tile, torch-side
bilinear upsample + cat. Similar to pointwise_05ee034feb4c but larger H/W.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C: ct.Constant[int], SPATIAL: ct.Constant[int],
    BLOCK_C: ct.Constant[int], BLOCK_S: ct.Constant[int],
):
    c_block = ct.bid(0)
    s_block = ct.bid(1)

    x = ct.load(x_ptr, index=(c_block, s_block), shape=(BLOCK_C, BLOCK_S))
    x_f = ct.astype(x, ct.float32)
    mean = ct.load(mean_ptr, index=(c_block,), shape=(BLOCK_C,))
    var = ct.load(var_ptr, index=(c_block,), shape=(BLOCK_C,))
    weight = ct.load(weight_ptr, index=(c_block,), shape=(BLOCK_C,))
    bias = ct.load(bias_ptr, index=(c_block,), shape=(BLOCK_C,))

    mean_f = ct.astype(mean, ct.float32)
    var_f = ct.astype(var, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    inv_std = 1.0 / ct.sqrt(var_f + 1e-5)
    mean_2d = ct.reshape(mean_f, (BLOCK_C, 1))
    inv_std_2d = ct.reshape(inv_std, (BLOCK_C, 1))
    weight_2d = ct.reshape(weight_f, (BLOCK_C, 1))
    bias_2d = ct.reshape(bias_f, (BLOCK_C, 1))

    normalized = (x_f - mean_2d) * inv_std_2d
    affine = normalized * weight_2d + bias_2d
    zero = ct.zeros((BLOCK_C, BLOCK_S), dtype=ct.float32)
    is_nan = affine != affine
    relu_masked = ct.where(affine > zero, affine, zero)
    relu = ct.where(is_nan, affine, relu_masked)
    ct.store(out_ptr, index=(c_block, s_block), tile=ct.astype(relu, ct.bfloat16))


@oracle_impl(hardware="B200", point="e3cdab72", BLOCK_C=64, BLOCK_S=64)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_S: int):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape = inputs
    device = arg0_1.device

    C, H, W = 64, 320, 479
    spatial = H * W
    padded_S = ((spatial + BLOCK_S - 1) // BLOCK_S) * BLOCK_S

    x = arg1_1.contiguous().view(C, spatial)
    padded_x = torch.zeros((C, padded_S), device=device, dtype=torch.bfloat16)
    padded_x[:, :spatial].copy_(x)

    padded_out = torch.zeros((C, padded_S), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(C, BLOCK_C), ct.cdiv(padded_S, BLOCK_S), 1),
        _bn_relu_kernel,
        (padded_x, arg0_1, arg2_1, arg3_1, arg4_1, padded_out,
         C, padded_S, BLOCK_C, BLOCK_S),
    )

    relu = padded_out[:, :spatial].contiguous().view(1, C, H, W)
    x_f = relu.to(torch.float32)

    def _lerp_indices(out_len, in_len_minus1, scale):
        iota = torch.arange(out_len, device=device, dtype=torch.float32)
        pos = torch.clamp_min(iota * scale, 0.0)
        low = pos.to(torch.int64)
        high = torch.clamp_max(low + 1, in_len_minus1)
        frac = torch.clamp((pos - low.to(torch.float32)).clamp_min(0.0), max=1.0)
        return low, high, frac

    low_h, high_h, frac_h = _lerp_indices(640, 319, 0.49921752738654146)
    low_w, high_w, frac_w = _lerp_indices(958, 478, 0.4994775339602926)

    v00 = x_f[:, :, high_h.view(640, 1), high_w]
    v01 = x_f[:, :, high_h.view(640, 1), low_w]
    v10 = x_f[:, :, low_h.view(640, 1), high_w]
    v11 = x_f[:, :, low_h.view(640, 1), low_w]

    top = v01 + (v00 - v01) * frac_w
    bot = v11 + (v10 - v11) * frac_w
    interp = bot + (top - bot) * frac_h.view(640, 1)
    interp_bf = interp.to(torch.bfloat16)
    padded = torch.nn.functional.pad(interp_bf, [0, 1, 0, 0], value=0.0)
    cat = torch.cat([arg5_1, padded], dim=1)
    return cat
