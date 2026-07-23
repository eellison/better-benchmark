"""cuTile port of pointwise_7899d74e0838: BN-inference affine + ReLU.

Handles both channels-last and NCHW input layouts. Non-pow2 dims are handled by
masked scatter.
"""

from __future__ import annotations

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_nhwc_kernel(
    mean_arr,    # bf16 [C]
    x_arr,       # bf16 [N, H*W, C]
    var_arr,     # bf16 [C]
    weight_arr,  # bf16 [C]
    bias_arr,    # bf16 [C]
    out_arr,     # bf16 [N, H*W, C]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
):
    n = ct.bid(0)
    hw_block = ct.bid(1)
    c_block = ct.bid(2)

    mean_v = ct.load(mean_arr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    var_v = ct.load(var_arr, index=(c_block,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight_v = ct.load(weight_arr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    bias_v = ct.load(bias_arr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)

    mean_f = ct.astype(mean_v, ct.float32)
    var_f = ct.astype(var_v, ct.float32)
    weight_f = ct.astype(weight_v, ct.float32)
    bias_f = ct.astype(bias_v, ct.float32)
    invstd = 1.0 / ct.sqrt(var_f + 0.001)
    scale = invstd * weight_f
    shift = bias_f - mean_f * scale
    scale_3d = ct.reshape(scale, (1, 1, BLOCK_C))
    shift_3d = ct.reshape(shift, (1, 1, BLOCK_C))

    x = ct.load(x_arr, index=(n, hw_block, c_block), shape=(1, BLOCK_HW, BLOCK_C),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    y_f = x_f * scale_3d + shift_3d
    y_bf = ct.astype(y_f, ct.bfloat16)
    zero_bf = ct.astype(ct.zeros(shape=(1, 1, BLOCK_C), dtype=ct.float32), ct.bfloat16)
    zero_3d = ct.broadcast_to(zero_bf, (1, BLOCK_HW, BLOCK_C))
    is_nan = y_bf != y_bf
    out_bf = ct.where(is_nan, y_bf, ct.where(y_bf > zero_3d, y_bf, zero_3d))

    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32) + hw_block * BLOCK_HW
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    hw_mask = hw_idx < HW
    c_mask = c_idx < C
    hw_mask_3d = ct.reshape(hw_mask, (1, BLOCK_HW, 1))
    c_mask_3d = ct.reshape(c_mask, (1, 1, BLOCK_C))
    valid = hw_mask_3d & c_mask_3d
    n_idx = ct.full(shape=(1, BLOCK_HW, BLOCK_C), fill_value=n, dtype=ct.int32)
    hw_bc = ct.broadcast_to(ct.reshape(hw_idx, (1, BLOCK_HW, 1)), (1, BLOCK_HW, BLOCK_C))
    c_bc = ct.broadcast_to(ct.reshape(c_idx, (1, 1, BLOCK_C)), (1, BLOCK_HW, BLOCK_C))
    ct.scatter(out_arr, (n_idx, hw_bc, c_bc), out_bf, mask=valid)


@ct.kernel
def _bn_relu_nchw_kernel(
    mean_arr,    # bf16 [C]
    x_arr,       # bf16 [N, C, H*W]
    var_arr,     # bf16 [C]
    weight_arr,  # bf16 [C]
    bias_arr,    # bf16 [C]
    out_arr,     # bf16 [N, C, H*W]
    C: ct.Constant[int],
    HW: ct.Constant[int],
    BLOCK_C: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    n = ct.bid(0)
    c_block = ct.bid(1)
    hw_block = ct.bid(2)

    mean_v = ct.load(mean_arr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)
    var_v = ct.load(var_arr, index=(c_block,), shape=(BLOCK_C,),
                    padding_mode=ct.PaddingMode.ZERO)
    weight_v = ct.load(weight_arr, index=(c_block,), shape=(BLOCK_C,),
                       padding_mode=ct.PaddingMode.ZERO)
    bias_v = ct.load(bias_arr, index=(c_block,), shape=(BLOCK_C,),
                     padding_mode=ct.PaddingMode.ZERO)

    mean_f = ct.astype(mean_v, ct.float32)
    var_f = ct.astype(var_v, ct.float32)
    weight_f = ct.astype(weight_v, ct.float32)
    bias_f = ct.astype(bias_v, ct.float32)
    invstd = 1.0 / ct.sqrt(var_f + 0.001)
    scale = invstd * weight_f
    shift = bias_f - mean_f * scale
    scale_3d = ct.reshape(scale, (1, BLOCK_C, 1))
    shift_3d = ct.reshape(shift, (1, BLOCK_C, 1))

    x = ct.load(x_arr, index=(n, c_block, hw_block), shape=(1, BLOCK_C, BLOCK_HW),
                padding_mode=ct.PaddingMode.ZERO)
    x_f = ct.astype(x, ct.float32)
    y_f = x_f * scale_3d + shift_3d
    y_bf = ct.astype(y_f, ct.bfloat16)
    zero_bf = ct.astype(ct.zeros(shape=(1, BLOCK_C, 1), dtype=ct.float32), ct.bfloat16)
    zero_3d = ct.broadcast_to(zero_bf, (1, BLOCK_C, BLOCK_HW))
    is_nan = y_bf != y_bf
    out_bf = ct.where(is_nan, y_bf, ct.where(y_bf > zero_3d, y_bf, zero_3d))

    hw_idx = ct.arange(BLOCK_HW, dtype=ct.int32) + hw_block * BLOCK_HW
    c_idx = ct.arange(BLOCK_C, dtype=ct.int32) + c_block * BLOCK_C
    hw_mask = hw_idx < HW
    c_mask = c_idx < C
    hw_mask_3d = ct.reshape(hw_mask, (1, 1, BLOCK_HW))
    c_mask_3d = ct.reshape(c_mask, (1, BLOCK_C, 1))
    valid = hw_mask_3d & c_mask_3d
    n_idx = ct.full(shape=(1, BLOCK_C, BLOCK_HW), fill_value=n, dtype=ct.int32)
    c_bc = ct.broadcast_to(ct.reshape(c_idx, (1, BLOCK_C, 1)), (1, BLOCK_C, BLOCK_HW))
    hw_bc = ct.broadcast_to(ct.reshape(hw_idx, (1, 1, BLOCK_HW)), (1, BLOCK_C, BLOCK_HW))
    ct.scatter(out_arr, (n_idx, c_bc, hw_bc), out_bf, mask=valid)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _run(mean, x, var, weight, bias):
    n, channels, height, width = x.shape
    hw = height * width
    out = torch.empty_strided(
        tuple(x.shape),
        tuple(x.stride()),
        device=x.device,
        dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    if x.stride(1) == 1:
        # Channels-last
        x_nhwc = x.permute(0, 2, 3, 1).reshape(n, hw, channels)
        out_nhwc = out.permute(0, 2, 3, 1).reshape(n, hw, channels)
        BLOCK_HW = min(64, _next_pow2(hw))
        BLOCK_C = min(64, _next_pow2(channels))
        ct.launch(
            stream,
            (n, ct.cdiv(hw, BLOCK_HW), ct.cdiv(channels, BLOCK_C)),
            _bn_relu_nhwc_kernel,
            (mean, x_nhwc, var, weight, bias, out_nhwc,
             channels, hw, BLOCK_HW, BLOCK_C),
        )
    else:
        # NCHW contiguous
        x_flat = x.reshape(n, channels, hw)
        out_flat = out.reshape(n, channels, hw)
        BLOCK_C = min(16, _next_pow2(channels))
        BLOCK_HW = min(256, _next_pow2(hw))
        ct.launch(
            stream,
            (n, ct.cdiv(channels, BLOCK_C), ct.cdiv(hw, BLOCK_HW)),
            _bn_relu_nchw_kernel,
            (mean, x_flat, var, weight, bias, out_flat,
             channels, hw, BLOCK_C, BLOCK_HW),
        )
    return out


_POINTS = [
    "aeef24a2", "d0015d69", "79f50b00", "45fcf39e", "6b130750", "466338df",
    "2eaf76ca", "5ecba4cf", "2869e39e", "8a535339", "f868ccff", "2b758769",
    "a034af1b", "4c825568", "3d7c0f11", "76836be5", "37cf4567", "09d7b0e3",
    "ff6665c7", "2d204c68", "0a96753f", "c7c6ed1f", "e7ac2695", "5ddd3fd5",
    "b88e53cc", "9c97edfa", "55c49b50", "9ffe1d7a", "d0456ef8", "dc068d28",
    "23b42229", "8273eec8", "fe78a763", "fb1609c3", "3a805d91", "f9b980a0",
    "9a46a7c3", "29619cda", "49317bb2", "15c8ba03", "5536024c", "706b5a8b",
    "0c615bbc", "86d8b817", "a01c3bbe", "ba61c626", "bc2770a8", "9e8f8bc2",
    "8004dcee", "6039b181", "d438d876", "b6537524", "cc855e4e", "27be3f93",
    "a3bf2c8d", "0fe88f70", "6558df98", "793856d7", "c2f1485a", "78e9098c",
    "4fbb328e", "c1f1985a", "b83c3106", "82781865", "3a0fa905", "e3b17b69",
    "d6d70882", "c9277843", "6cc76740", "08df9f87", "a952c87d", "51b8f364",
    "e78adaa5", "09db272d", "1f81c187", "b5cedc58", "c78ca241", "11ab0e65",
    "4194c732", "05028dd8", "4f218228", "0ce7c2f4", "535abe76", "3f17dc95",
    "d957323e", "7fd9f949", "2b7370d5", "6ff82f3b", "f50736b8", "ad2ff10f",
    "7ebdbaf5", "8380c440", "e3fcf83d", "20f76211", "2ff0039c", "93683cfe",
    "6b05bfa0", "6a1621d9", "f7d56653", "c7ca5fcc", "eff1e63e", "18ebcf69",
    "13e72620", "b30eafb6", "f3036c42", "eb73312d", "9a05b224", "c356b4b1",
    "a8109c98", "a9c4f100", "7e261bd0", "fe80f1ea", "217063c8", "13bff0d2",
    "23eaa07b", "dd92c2d0", "357d64ea", "e84dbb88", "839e83c4", "608e0114",
    "a2b8c17d",
]


def oracle_forward(inputs):
    mean, x, var, weight, bias = inputs
    return _run(mean, x, var, weight, bias)


for _p in _POINTS:
    oracle_forward = oracle_impl(hardware="B200", point=_p)(oracle_forward)
