"""cuTile port of sum_sum_sum_72ccaeb6654a: Inception avg-pool backward branches
over non-pow-2 spatial 35x35. Four branches share the same pool_grad tensor
but pick different channel offsets and then call native_batch_norm_backward.

Structure per branch:
  * `_make_branch_grad_kernel` — for each flat element (n, c, h, w), gather the
    9-neighborhood of the pool_grad tensor (at [n, source_offset+c, h', w'] with
    boundary masks), accumulate bf16-rounded partial sums, apply the residual
    add chain, compute affine, apply the ReLU-le gate, and scatter into grad.

The 35x35 spatial is not power-of-two, so we use `ct.scatter`/`ct.gather` with
explicit indices + mask rather than tile loads/stores.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.1111111111111111
HEIGHT = 35
WIDTH = 35


@ct.kernel
def _make_branch_grad_kernel(
    pool_ptr,          # bf16 [N, 288, 35, 35] channels-last
    add0_ptr,          # bf16 [N, 288, 35, 35] channels-last
    add1_ptr,          # bf16 [N, 288, 35, 35] channels-last
    add2_ptr,          # bf16 [N, 288, 35, 35] channels-last
    x_ptr,             # bf16 [N, C, 35, 35] channels-last
    mean_ptr,          # f32  [C]
    invstd_ptr,        # f32  [C]
    weight_ptr,        # f32  [C]
    bias_ptr,          # f32  [C]
    grad_ptr,          # bf16 [N, C, 35, 35] channels-last (output)
    zero_scalar_ptr,   # bf16 [] (fill)
    source_offset: ct.Constant[int],
    channels: ct.Constant[int],
    hw_size: ct.Constant[int],
    total_elems: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    offs = ct.arange(BLOCK, dtype=ct.int32) + pid * BLOCK
    active = offs < total_elems

    # Decompose: offs = n * (C*HW) + hw * C + c  (channels-last order)
    # But easier: iterate as if flat NCHW; the original computed by:
    #   c = offs % channels; hw = (offs // channels) % 1225; n = offs // row_stride
    # Note row_stride = 352800 for pool but C_branch * HW for x.
    # We adopt the same convention:
    c = offs - (offs // channels) * channels
    hw_row = offs // channels          # (n * HW + hw)
    hw = hw_row - (hw_row // hw_size) * hw_size
    n = hw_row // hw_size
    h = hw // WIDTH
    w = hw - h * WIDTH

    up1 = h > 0
    up2 = h > 1
    left1 = w > 0
    left2 = w > 1

    c_pool = c + source_offset  # channel index in pool tensor

    # Broadcast mask. Clamp indices to safe values when masked-out to
    # avoid negative indices tripping bounds checks.
    zero_i = ct.zeros((BLOCK,), dtype=ct.int32)
    def _load_pool(hh, ww, extra_mask):
        m = active & extra_mask
        hh_safe = ct.where(m, hh, zero_i)
        ww_safe = ct.where(m, ww, zero_i)
        pool_val_bf = ct.gather(pool_ptr, (n, c_pool, hh_safe, ww_safe),
                                mask=m, padding_value=0)
        pool_f = ct.astype(pool_val_bf, ct.float32) * SCALE
        return ct.astype(ct.astype(pool_f, ct.bfloat16), ct.float32)

    all_true = active
    pool = _load_pool(h, w, all_true)
    pool = pool + _load_pool(h - 1, w, up1)
    pool = pool + _load_pool(h - 2, w, up2)
    pool = pool + _load_pool(h, w - 1, left1)
    pool = pool + _load_pool(h, w - 2, left2)
    pool = pool + _load_pool(h - 1, w - 1, up1 & left1)
    pool = pool + _load_pool(h - 1, w - 2, up1 & left2)
    pool = pool + _load_pool(h - 2, w - 1, up2 & left1)
    pool = pool + _load_pool(h - 2, w - 2, up2 & left2)
    pooled = ct.astype(ct.astype(pool, ct.bfloat16), ct.float32)

    a0_bf = ct.gather(add0_ptr, (n, c_pool, h, w), mask=active, padding_value=0)
    a1_bf = ct.gather(add1_ptr, (n, c_pool, h, w), mask=active, padding_value=0)
    a2_bf = ct.gather(add2_ptr, (n, c_pool, h, w), mask=active, padding_value=0)

    add0 = ct.astype(
        ct.astype(pooled + ct.astype(a0_bf, ct.float32), ct.bfloat16),
        ct.float32,
    )
    add1 = ct.astype(
        ct.astype(add0 + ct.astype(a1_bf, ct.float32), ct.bfloat16),
        ct.float32,
    )
    source = ct.astype(
        ct.astype(add1 + ct.astype(a2_bf, ct.float32), ct.bfloat16),
        ct.float32,
    )

    x_bf = ct.gather(x_ptr, (n, c, h, w), mask=active, padding_value=0)
    x_f = ct.astype(x_bf, ct.float32)
    mean_v = ct.astype(ct.gather(mean_ptr, c, mask=active, padding_value=0), ct.float32)
    invstd_v = ct.astype(ct.gather(invstd_ptr, c, mask=active, padding_value=0), ct.float32)
    weight_v = ct.astype(ct.gather(weight_ptr, c, mask=active, padding_value=0), ct.float32)
    bias_v = ct.astype(ct.gather(bias_ptr, c, mask=active, padding_value=0), ct.float32)

    affine = (x_f - mean_v) * invstd_v * weight_v + bias_v
    affine_bf16_f32 = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    zero_scalar = ct.load(zero_scalar_ptr, index=(0,), shape=(1,))
    zero_f = ct.astype(zero_scalar, ct.float32)
    zero_bcast = ct.reshape(zero_f, (1,)) + ct.zeros((BLOCK,), dtype=ct.float32)

    grad_f = ct.where(affine_bf16_f32 <= 0.0, zero_bcast, source)
    grad_bf = ct.astype(grad_f, ct.bfloat16)
    ct.scatter(grad_ptr, (n, c, h, w), grad_bf, mask=active)


def _native_bn_branch(grad, x, mean, invstd, weight):
    grad_input, grad_weight, grad_bias = torch.ops.aten.native_batch_norm_backward.default(
        grad, x, weight, None, None,
        mean.reshape(-1), invstd.reshape(-1),
        True, 0.0, [True, True, True],
    )
    return grad_bias, grad_weight, grad_input


def _emit_grad(pool_grad, add0, add1, add2, x, mean, invstd, weight, bias,
               zero, source_offset, *, BLOCK):
    channels = int(x.shape[1])
    hw_size = HEIGHT * WIDTH
    total = int(x.shape[0]) * channels * hw_size
    grad = torch.empty_strided(
        tuple(x.shape), tuple(x.stride()),
        device=x.device, dtype=torch.bfloat16,
    )
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        ((total + BLOCK - 1) // BLOCK, 1, 1),
        _make_branch_grad_kernel,
        (pool_grad, add0, add1, add2, x,
         mean.reshape(-1), invstd.reshape(-1),
         weight.reshape(-1), bias.reshape(-1), grad,
         zero.reshape(1),
         source_offset, channels, hw_size, total, BLOCK),
    )
    return grad


@oracle_impl(hardware="B200", point="a91d4c93", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    (arg0_1, arg1_1, arg2_1, arg3_1, arg4_1,
     arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1,
     arg11_1, arg12_1, arg13_1, arg14_1, arg15_1,
     arg16_1, arg17_1, arg18_1, arg19_1, arg20_1,
     arg21_1, arg22_1, arg23_1, arg24_1, arg25_1) = inputs
    del arg1_1

    def _bn_branch(grad, x_activation, mean, invstd, weight):
        return _native_bn_branch(grad, x_activation, mean, invstd, weight)

    grad0 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, 224, BLOCK=BLOCK)
    grad1 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg10_1, 128, BLOCK=BLOCK)
    grad2 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg10_1, 64, BLOCK=BLOCK)
    grad3 = _emit_grad(arg0_1, arg2_1, arg3_1, arg4_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg10_1, 0, BLOCK=BLOCK)

    out0 = _bn_branch(grad0, arg5_1, arg6_1, arg7_1, arg8_1)
    out1 = _bn_branch(grad1, arg11_1, arg12_1, arg13_1, arg14_1)
    out2 = _bn_branch(grad2, arg16_1, arg17_1, arg18_1, arg19_1)
    out3 = _bn_branch(grad3, arg21_1, arg22_1, arg23_1, arg24_1)
    return out0 + out1 + out2 + out3
