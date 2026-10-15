"""cuTile port of sum_sum_sum_db03a2e026aa: RepVGG chained BN-backward multi-stage.

The full graph does three chained BN-backward stages sharing a producer.
We implement the elementwise BN-backward using a cuTile kernel and use
pytorch for reductions. All shapes are power-of-2-compatible.

Two shape points:
  ff6d7f0c: N=128, C=192, H=W=28 (HW=784, K=100352)
  c0fba172: N=128, C=384, H=W=14 (HW=196, K=25088)
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


INV_NHW = 9.964923469387754e-06


def _to_c_nhw(t, C, HW):
    # (N, C, H, W) contiguous or channels-last → (C, N*H*W) with C outer.
    N = t.shape[0]
    return t.permute(1, 0, 2, 3).contiguous().view(C, N * HW)


def _from_c_nhw(t_c, N, C, H, W):
    # (C, N*H*W) → (N, C, H, W) with contiguous stride.
    return t_c.view(C, N, H, W).permute(1, 0, 2, 3).contiguous()


@ct.kernel
def _bn_backward_element_kernel(
    grad_ptr,      # bf16/f32 (C, K)
    x_ptr,         # bf16/f32 (C, K)
    mean_ptr,      # f32 (C,)
    invstd_ptr,    # f32 (C,)
    weight_ptr,    # f32 (C,)
    mean_grad_ptr, # f32 (C,)  = mean(grad) per channel
    prod_mean_ptr, # f32 (C,)  = mean(grad*(x-mean)) per channel
    out_ptr,       # bf16 (C, K)
    K_C: ct.Constant[int],
    BLOCK_R: ct.Constant[int],
    GRAD_IS_F32: ct.Constant[int],
    X_IS_F32: ct.Constant[int],
):
    c = ct.bid(0)

    if GRAD_IS_F32 == 1:
        grad = ct.load(
            grad_ptr, index=(c, 0), shape=(1, BLOCK_R),
            padding_mode=ct.PaddingMode.ZERO,
        )
    else:
        grad = ct.astype(
            ct.load(
                grad_ptr, index=(c, 0), shape=(1, BLOCK_R),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            ct.float32,
        )
    if X_IS_F32 == 1:
        x = ct.load(
            x_ptr, index=(c, 0), shape=(1, BLOCK_R),
            padding_mode=ct.PaddingMode.ZERO,
        )
    else:
        x = ct.astype(
            ct.load(
                x_ptr, index=(c, 0), shape=(1, BLOCK_R),
                padding_mode=ct.PaddingMode.ZERO,
            ),
            ct.float32,
        )

    mean_1d = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd_1d = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1d = ct.load(weight_ptr, index=(c,), shape=(1,))
    mean_grad_1d = ct.load(mean_grad_ptr, index=(c,), shape=(1,))
    prod_mean_1d = ct.load(prod_mean_ptr, index=(c,), shape=(1,))

    mean_bc = ct.reshape(mean_1d, (1, 1))
    invstd_bc = ct.reshape(invstd_1d, (1, 1))
    weight_bc = ct.reshape(weight_1d, (1, 1))
    mean_grad_bc = ct.reshape(mean_grad_1d, (1, 1))
    prod_mean_bc = ct.reshape(prod_mean_1d, (1, 1))

    centered = x - mean_bc
    coeff = prod_mean_bc * (invstd_bc * invstd_bc)
    scale = invstd_bc * weight_bc

    out = (grad - centered * coeff - mean_grad_bc) * scale
    ct.store(out_ptr, index=(c, 0), tile=ct.astype(out, ct.bfloat16))


def _bn_backward(grad, x, mean_reshaped, invstd, weight, N, C, H, W, HW, K, block_r):
    """Compute BN backward: sum_grad, prod (grad * (x-mean)), and dense output.

    grad and x are (N, C, H, W); mean is (1, C, 1, 1) or (C,); invstd, weight
    are (C,).  Returns (sum_grad (C,), prod_sum (C,), dense_out (N,C,H,W) bf16).
    """
    device = x.device
    # First, compute the two channel reductions via pytorch (fast, well-optimized).
    if grad.dtype == torch.float32:
        grad_f32 = grad
    else:
        grad_f32 = grad.to(torch.float32)
    x_f32 = x.to(torch.float32)
    mean_bc = mean_reshaped.view(1, C, 1, 1)
    centered = x_f32 - mean_bc
    sum_grad = grad_f32.sum(dim=(0, 2, 3))  # (C,)
    prod_sum = (grad_f32 * centered).sum(dim=(0, 2, 3))  # (C,)

    mean_grad = sum_grad * INV_NHW
    prod_mean = prod_sum * INV_NHW

    grad_c = _to_c_nhw(grad, C, HW)
    x_c = _to_c_nhw(x, C, HW)
    out_c = torch.empty((C, K), device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _bn_backward_element_kernel,
        (
            grad_c, x_c, mean_reshaped.view(C), invstd, weight,
            mean_grad, prod_mean, out_c,
            K, block_r, 1 if grad.dtype == torch.float32 else 0,
            1 if x.dtype == torch.float32 else 0,
        ),
    )
    out = _from_c_nhw(out_c, N, C, H, W)
    return sum_grad, prod_sum, out


def _launch(inputs, *, N, C, H, W, BLOCK_R):
    (
        arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7,
        arg8, arg9, arg10, arg11, arg12, arg13, arg14, arg15,
    ) = inputs
    device = arg2.device
    HW = H * W
    K = N * HW

    # Stage 1: BN-backward on grad=arg2, x=arg3, mean=arg4, invstd=arg5, weight=arg6.
    sum_1, prod_1, dense1 = _bn_backward(
        arg2, arg3, arg4, arg5, arg6, N, C, H, W, HW, K, BLOCK_R
    )
    mul_8 = prod_1 * arg5  # sum_2 * invstd, used as vec output

    # Producer: (arg0 + arg1) + dense1  (all bf16), then relu-mask with arg7.
    residual = arg0 + arg1
    add_1 = residual + dense1
    le = arg3 <= 0
    producer = torch.where(le, arg7, add_1)  # bf16

    # Stage 2: BN-backward on grad=producer, x=arg8, mean=arg9, invstd=arg10, weight=arg11.
    sum_3, prod_3, dense2 = _bn_backward(
        producer, arg8, arg9, arg10, arg11, N, C, H, W, HW, K, BLOCK_R
    )
    mul_17 = prod_3 * arg10

    # Stage 3: BN-backward on grad=producer, x=arg12, mean=arg13, invstd=arg14, weight=arg15.
    sum_5, prod_5, dense3 = _bn_backward(
        producer, arg12, arg13, arg14, arg15, N, C, H, W, HW, K, BLOCK_R
    )
    mul_26 = prod_5 * arg14

    # Producer is bf16 but the output is convert_element_type_2 = f32 producer.
    producer_f32 = producer.to(torch.float32)
    return (
        sum_1, mul_8, producer_f32,
        sum_3, mul_17, dense2,
        sum_5, mul_26, dense3,
    )


# ff6d7f0c: N=128, C=192, H=W=28
@oracle_impl(hardware="B200", point="ff6d7f0c", H=28, W=28, C_VAL=192, BLOCK_R=131072)
# c0fba172: N=128, C=384, H=W=14
@oracle_impl(hardware="B200", point="c0fba172", H=14, W=14, C_VAL=384, BLOCK_R=32768)
def oracle_forward(inputs, *, H, W, C_VAL, BLOCK_R):
    return _launch(inputs, N=128, C=C_VAL, H=H, W=W, BLOCK_R=BLOCK_R)
