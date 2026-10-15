"""cuTile port of sum_sum_12fc8ebf8180: DCGAN BN-backward + leaky-ReLU.

Matches Triton's single-kernel structure: one program per channel that
fuses leaky-ReLU selection, per-channel sum + dot reductions, and the
BN-backward dense epilogue. Data is viewed as (N, C, HW) so a single
`ct.load` per input reads the entire channel-c strip.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 0.0001220703125


@ct.kernel
def _channel_reduce_epilogue_kernel(
    arg0_ptr,             # bf16 (N, C, HW)
    arg1_ptr,             # bf16 (N, C, HW)
    arg2_ptr,             # bf16 (N, C, HW)
    arg3_ptr,             # f32 (C,)   mean
    arg4_ptr,             # f32 (C,)   weight
    arg5_ptr,             # f32 (C,)   grad_weight
    sum_out_ptr,          # f32 (C,)
    dot_scaled_out_ptr,   # f32 (C,)
    out_ptr,              # bf16 (N, C, HW)
    N_: ct.Constant[int],
    HW_: ct.Constant[int],
    SCALE_VALUE: ct.Constant[float],
):
    c = ct.bid(0)
    x0 = ct.load(arg0_ptr, index=(0, c, 0), shape=(N_, 1, HW_))
    gate = ct.load(arg1_ptr, index=(0, c, 0), shape=(N_, 1, HW_))
    x2 = ct.load(arg2_ptr, index=(0, c, 0), shape=(N_, 1, HW_))
    mean = ct.load(arg3_ptr, index=(c,), shape=(1,))
    weight = ct.load(arg4_ptr, index=(c,), shape=(1,))
    grad_weight = ct.load(arg5_ptr, index=(c,), shape=(1,))

    x0_f = ct.astype(x0, ct.float32)
    gate_f = ct.astype(gate, ct.float32)
    x2_f = ct.astype(x2, ct.float32)

    leaky = x0_f * 0.2
    zero_f = ct.zeros((N_, 1, HW_), dtype=ct.float32)
    selected_pre = ct.where(gate_f > zero_f, x0_f, leaky)
    # bf16 rounding boundary
    selected = ct.astype(ct.astype(selected_pre, ct.bfloat16), ct.float32)

    mean_bcast = ct.reshape(mean, (1, 1, 1))
    centered = x2_f - mean_bcast
    dot = selected * centered

    sum_value = ct.sum(selected)   # 0-d
    dot_value = ct.sum(dot)        # 0-d

    weight_1 = ct.reshape(weight, (1,))
    grad_weight_1 = ct.reshape(grad_weight, (1,))

    sum_tile = ct.full((1,), sum_value, dtype=ct.float32)
    dot_tile = ct.full((1,), dot_value, dtype=ct.float32)
    dot_scaled_1 = dot_tile * weight_1
    ct.store(sum_out_ptr, index=(c,), tile=sum_tile)
    ct.store(dot_scaled_out_ptr, index=(c,), tile=dot_scaled_1)

    weight_sq = weight_1 * weight_1
    dot_mean = dot_tile * SCALE_VALUE
    correction_scale = ct.reshape(dot_mean * weight_sq, (1, 1, 1))
    correction = centered * correction_scale
    after_correction = selected - correction
    mean_term = sum_tile * SCALE_VALUE
    mean_term_3d = ct.reshape(mean_term, (1, 1, 1))
    centered_grad = after_correction - mean_term_3d
    output_scale_3d = ct.reshape(weight_1 * grad_weight_1, (1, 1, 1))
    out_f = centered_grad * output_scale_3d
    ct.store(out_ptr, index=(0, c, 0), tile=ct.astype(out_f, ct.bfloat16))


@oracle_impl(hardware="B200", point="a564ddd4", BLOCK_K=8192)
@oracle_impl(hardware="B200", point="49f9b4bd", BLOCK_K=2048)
@oracle_impl(hardware="B200", point="15406f9b", BLOCK_K=512)
def oracle_forward(inputs, *, BLOCK_K):
    del BLOCK_K
    arg0, arg1, arg2, arg3, arg4, arg5 = inputs
    n, c, h, w = (int(dim) for dim in arg0.shape)
    hw = h * w
    device = arg0.device

    # View NCHW as (N, C, HW). This is a metadata reshape only.
    arg0_v = arg0.view(n, c, hw)
    arg1_v = arg1.view(n, c, hw)
    arg2_v = arg2.view(n, c, hw)
    mean_1d = arg3.view(c)   # was (1, C, 1, 1) f32

    sum_out = torch.empty((c,), device=device, dtype=torch.float32)
    dot_scaled_out = torch.empty((c,), device=device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        tuple(arg0.shape),
        tuple(arg0.stride()),
        device=device,
        dtype=torch.bfloat16,
    )
    dense_out_v = dense_out.view(n, c, hw)

    stream = torch.cuda.current_stream()
    ct.launch(
        stream, (c, 1, 1), _channel_reduce_epilogue_kernel,
        (arg0_v, arg1_v, arg2_v, mean_1d, arg4, arg5,
         sum_out, dot_scaled_out, dense_out_v,
         n, hw, SCALE),
    )
    return sum_out, dot_scaled_out, dense_out
