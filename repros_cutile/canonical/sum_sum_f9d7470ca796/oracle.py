"""cuTile port of sum_sum_f9d7470ca796: DenseNet BN backward with fused per-channel reductions and epilogue."""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 128
C = 184
H = 4
W = 4
HW = H * W


@ct.kernel
def _channel_reduce_epilogue_kernel(
    grad_source_ptr,  # (N, C) bf16
    x_ptr,            # (N, C, H, W) bf16
    mean_ptr,         # (C,) f32 (view of shape (1, C, 1, 1))
    invstd_ptr,       # (C,) f32
    weight_ptr,       # (C,) f32
    bias_ptr,         # (C,) f32
    full_scalar_ptr,  # () bf16 (viewed as (1,))
    sum_where_ptr,    # (C,) f32
    weight_grad_ptr,  # (C,) f32
    out_bf16_ptr,     # (N, C, H, W) bf16
    BLOCK_N: ct.Constant[int],
    BLOCK_HW: ct.Constant[int],
):
    c = ct.bid(0)

    # Load source [N,] and x [N, HW]. Since HW=16, we load [N, HW] directly.
    # Load source across all N for this c: (N,) - view input as (N, C)
    source = ct.load(grad_source_ptr, index=(0, c), shape=(BLOCK_N, 1))
    source_f = ct.astype(source, ct.float32)
    source_div = ct.astype(ct.astype(source_f * 0.0625, ct.bfloat16), ct.float32)

    # Load x[:, c, :, :] as (N, 1, H, W)
    x = ct.load(x_ptr, index=(0, c, 0, 0), shape=(BLOCK_N, 1, H, W))
    x_f = ct.astype(x, ct.float32)
    x_2d = ct.reshape(x_f, (BLOCK_N, HW))

    mean = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias = ct.load(bias_ptr, index=(c,), shape=(1,))

    mean_f = ct.astype(mean, ct.float32)
    invstd_f = ct.astype(invstd, ct.float32)
    weight_f = ct.astype(weight, ct.float32)
    bias_f = ct.astype(bias, ct.float32)

    centered = x_2d - mean_f
    normalized = centered * invstd_f
    affine = normalized * weight_f + bias_f
    affine_bf16 = ct.astype(affine, ct.bfloat16)
    affine_f_rounded = ct.astype(affine_bf16, ct.float32)

    # source_div is (N, 1); we need to broadcast to (N, HW)
    source_div_2d = ct.reshape(source_div, (BLOCK_N, 1))
    zero_2d = ct.zeros(shape=(BLOCK_N, HW), dtype=ct.float32)
    # broadcast source_div_2d to (N, HW) via broadcast
    src_bc = source_div_2d + zero_2d
    where_value = ct.where(affine_f_rounded <= 0.0, zero_2d, src_bc)

    sum_where = ct.sum(where_value)
    sum_centered = ct.sum(where_value * centered)

    mean_term = sum_where * 0.00048828125
    centered_scale = sum_centered * 0.00048828125 * invstd_f * invstd_f
    output_scale = invstd_f * weight_f

    grad = (where_value - centered * centered_scale - mean_term) * output_scale
    grad_4d = ct.reshape(grad, (BLOCK_N, 1, H, W))
    grad_bf = ct.astype(grad_4d, ct.bfloat16)

    ct.store(sum_where_ptr, index=(c,), tile=sum_where)
    ct.store(weight_grad_ptr, index=(c,), tile=sum_centered * invstd_f)
    ct.store(out_bf16_ptr, index=(0, c, 0, 0), tile=grad_bf)


@ct.kernel
def _scalar_bf16_zero_kernel(out_ptr):
    ct.store(out_ptr, index=(0,), tile=ct.zeros(shape=(1,), dtype=ct.bfloat16))


@oracle_impl(hardware="B200", point="e3c25c9d", BLOCK_N=128, BLOCK_HW=16)
def oracle_forward(inputs, *, BLOCK_N, BLOCK_HW):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, _shape_param_0, _shape_param_1 = inputs

    device = arg0_1.device
    full_scalar = torch.empty((), device=device, dtype=torch.bfloat16)
    sum_where = torch.empty((C,), device=device, dtype=torch.float32)
    weight_grad = torch.empty((C,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    # arg0_1 is (128, 184) bf16.  arg1_1 is (128, 184, 4, 4) bf16.
    # mean, invstd have shape (1, 184, 1, 1) f32 -> view as (184,)
    mean_1d = arg2_1.view(C)
    invstd_1d = arg3_1.view(C)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _channel_reduce_epilogue_kernel,
        (arg0_1, arg1_1, mean_1d, invstd_1d, arg4_1, arg5_1,
         full_scalar, sum_where, weight_grad, out_bf16,
         BLOCK_N, BLOCK_HW),
    )
    ct.launch(stream, (1, 1, 1), _scalar_bf16_zero_kernel, (full_scalar.view(1),))

    return full_scalar, sum_where, weight_grad, out_bf16, out_bf16[:, :16, :, :]
