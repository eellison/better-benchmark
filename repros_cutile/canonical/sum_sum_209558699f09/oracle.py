"""cuTile port of sum_sum_209558699f09: DenseNet-121 BN backward tail.

Matches Triton's single-kernel plan: one program per channel c, which reduces
across R = N*HW rows to compute sum_where and sum_centered, then emits the
BN-backward bf16 output plus the scale_grad vector.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


N = 4
C = 1024
H = 7
W = 7
HW = H * W
R = N * HW
INV_HW = 0.02040816326530612
INV_COUNT = 0.00510204081632653
SLICE_START = 992


@ct.kernel
def _channel_reduce_epilogue_kernel(
    grad_source_ptr,   # bf16 [N, C]
    x_ptr,             # bf16 [N, C, H, W]
    mean_ptr,          # f32 [C]
    invstd_ptr,        # f32 [C]
    weight_ptr,        # f32 [C]
    bias_ptr,          # f32 [C]
    full_scalar_ptr,   # bf16 [1]
    sum_where_ptr,     # f32 [C]
    weight_grad_ptr,   # f32 [C]
    out_bf16_ptr,      # bf16 [N, C, H, W]
    BLOCK_R: ct.Constant[int],
):
    c = ct.bid(0)
    rows = ct.arange(BLOCK_R, dtype=ct.int32)
    active = rows < R
    n = rows // HW
    spatial = rows - n * HW
    h_idx = spatial // W
    w_idx = spatial - h_idx * W

    zero_i = ct.zeros((BLOCK_R,), dtype=ct.int32)
    n_safe = ct.where(active, n, zero_i)
    h_safe = ct.where(active, h_idx, zero_i)
    w_safe = ct.where(active, w_idx, zero_i)
    c_full = ct.full((BLOCK_R,), c, dtype=ct.int32)

    source = ct.astype(
        ct.gather(grad_source_ptr, (n_safe, c_full), mask=active), ct.float32
    )
    source_div = source * INV_HW
    x = ct.astype(
        ct.gather(x_ptr, (n_safe, c_full, h_safe, w_safe), mask=active),
        ct.float32,
    )
    mean_1 = ct.load(mean_ptr, index=(c,), shape=(1,))
    invstd_1 = ct.load(invstd_ptr, index=(c,), shape=(1,))
    weight_1 = ct.load(weight_ptr, index=(c,), shape=(1,))
    bias_1 = ct.load(bias_ptr, index=(c,), shape=(1,))
    mean = ct.reshape(mean_1, (1,))
    invstd = ct.reshape(invstd_1, (1,))
    weight = ct.reshape(weight_1, (1,))
    bias = ct.reshape(bias_1, (1,))

    centered = x - mean
    normalized = centered * invstd
    affine = normalized * weight + bias
    affine_bf16 = ct.astype(ct.astype(affine, ct.bfloat16), ct.float32)
    zero_f = ct.zeros((BLOCK_R,), dtype=ct.float32)
    where_value = ct.where(active & (affine_bf16 > 0.0), source_div, zero_f)

    product = where_value * centered
    sum_where = ct.sum(where_value)
    sum_centered = ct.sum(product)

    mean_term = sum_where * INV_COUNT
    dot_scaled = sum_centered * INV_COUNT
    invstd_sq = invstd * invstd
    variance_term = dot_scaled * invstd_sq
    output_scale = invstd * weight
    after_variance = where_value - centered * variance_term
    after_mean = after_variance - mean_term
    grad = after_mean * output_scale

    sum_where_1 = ct.full((1,), sum_where, dtype=ct.float32)
    ct.store(sum_where_ptr, index=(c,), tile=sum_where_1)
    scale_grad_val = sum_centered * invstd
    ct.store(weight_grad_ptr, index=(c,), tile=scale_grad_val)

    grad_bf = ct.astype(grad, ct.bfloat16)
    ct.scatter(
        out_bf16_ptr, (n_safe, c_full, h_safe, w_safe), grad_bf, mask=active,
    )

    if c == 0:
        zero_bf_1 = ct.zeros((1,), dtype=ct.bfloat16)
        ct.store(full_scalar_ptr, index=(0,), tile=zero_bf_1)


@oracle_impl(hardware="B200", point="4d18b4bc", BLOCK_R=256)
def oracle_forward(inputs, *, BLOCK_R: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    device = arg0_1.device
    full_scalar = torch.empty((1,), device=device, dtype=torch.bfloat16)
    sum_where = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    weight_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out_bf16 = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    mean_1d = arg2_1.view(C)
    invstd_1d = arg3_1.view(C)
    weight_1d = arg4_1
    bias_1d = arg5_1

    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (C, 1, 1),
        _channel_reduce_epilogue_kernel,
        (arg0_1, arg1_1, mean_1d, invstd_1d, weight_1d, bias_1d,
         full_scalar, sum_where, weight_grad, out_bf16, BLOCK_R),
    )

    full_scalar_0d = full_scalar.view(())
    return full_scalar_0d, sum_where, weight_grad, out_bf16, out_bf16[:, SLICE_START:C, :, :]
