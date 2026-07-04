"""cuTile port of sum_sum_sum_275ca61fceb8: GhostNet dual BN-backward.

Triton uses PTX .rn intrinsics (=cuTile default RTNE). The heavy indexing
(channels-last copy, slice, bf16 add) is done in torch; cuTile does an
elementwise BN-backward epilogue kernel that computes the final bf16 output.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_backward_epilogue_kernel(
    src_ptr,          # f32 [N*C*HW]
    centered_ptr,     # f32 [N*C*HW]
    mean_bc_ptr,      # f32 [N*C*HW]
    variance_bc_ptr,  # f32 [N*C*HW]
    gain_bc_ptr,      # f32 [N*C*HW]
    out_ptr,          # bf16 [N*C*HW]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    src = ct.load(src_ptr, index=(pid,), shape=(BLOCK,))
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean_bc = ct.load(mean_bc_ptr, index=(pid,), shape=(BLOCK,))
    variance_bc = ct.load(variance_bc_ptr, index=(pid,), shape=(BLOCK,))
    gain_bc = ct.load(gain_bc_ptr, index=(pid,), shape=(BLOCK,))
    sub1 = src - centered * variance_bc
    sub2 = sub1 - mean_bc
    out_f = sub2 * gain_bc
    ct.store(out_ptr, index=(pid,), tile=ct.astype(out_f, ct.bfloat16))


def _bn_backward(x_bf16, x_input_bf16, mean_1c, gamma_1c, bias, inv_count, device):
    N, C, H, W = x_bf16.shape
    numel = N * C * H * W

    x_f32 = x_bf16.to(torch.float32)
    sum_1 = x_f32.sum(dim=[0, 2, 3])
    centered = x_input_bf16.to(torch.float32) - mean_1c
    mul_x_centered = x_f32 * centered
    sum_2 = mul_x_centered.sum(dim=[0, 2, 3])
    mul1 = sum_1 * inv_count
    mul2 = sum_2 * inv_count
    gamma2 = gamma_1c.view(-1) * gamma_1c.view(-1)
    variance = mul2 * gamma2
    gain = gamma_1c.view(-1) * bias

    mean_bc = mul1.view(1, C, 1, 1).expand(N, C, H, W).contiguous().view(numel)
    variance_bc = variance.view(1, C, 1, 1).expand(N, C, H, W).contiguous().view(numel)
    gain_bc = gain.view(1, C, 1, 1).expand(N, C, H, W).contiguous().view(numel)

    src_flat = x_f32.contiguous().view(numel)
    centered_flat = centered.contiguous().view(numel)
    out_bf16 = torch.empty(numel, device=device, dtype=torch.bfloat16)

    BLOCK = 1024
    while numel % BLOCK != 0 and BLOCK > 1:
        BLOCK //= 2
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(numel, BLOCK), 1, 1),
        _bn_backward_epilogue_kernel,
        (src_flat, centered_flat, mean_bc, variance_bc, gain_bc, out_bf16, BLOCK),
    )
    mul_8 = sum_2 * gamma_1c.view(-1)
    return sum_1, mul_8, out_bf16.view(N, C, H, W)


def _forward(inputs, **kwargs):
    (
        arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1,
        arg6_1, arg7_1, arg8_1, arg9_1,
        s0, s1,
    ) = inputs

    device = arg0_1.device
    N, C24, H, W = arg0_1.shape

    inv_count = 6.228077168367346e-07

    # add = arg0 + arg1 (bf16)
    add = arg0_1 + arg1_1
    strides = tuple(int(x) for x in s1)
    shape = tuple(int(x) for x in s0)
    copy_ = torch.empty_strided(shape, strides, device=device, dtype=torch.bfloat16)
    copy_.copy_(add)
    clone = copy_.contiguous()
    copy_.copy_(clone)

    # gamma = arg4, bias(=beta) = arg5 -> gain = gamma * bias per Repro
    sum_1, mul_8, conv_bf16_2 = _bn_backward(
        clone, arg2_1, arg3_1, arg4_1.view(1, C24, 1, 1),
        arg5_1, inv_count, device,
    )

    C12 = 12
    slice_1 = copy_[:, 12:24, :, :].contiguous()
    sum_3, mul_17, conv_bf16_5 = _bn_backward(
        slice_1, arg6_1, arg7_1, arg8_1.view(1, C12, 1, 1),
        arg9_1, inv_count, device,
    )

    return copy_, sum_1, mul_8, conv_bf16_2, sum_3, mul_17, conv_bf16_5


@oracle_impl(hardware="B200", point="ccd5faf2")
def oracle_forward(inputs, **kwargs):
    return _forward(inputs, **kwargs)
