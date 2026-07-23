"""cuTile port of sum_sum_0bad56ceb78f: MobileNetV2 hardtanh BN-backward.

Hybrid plan for numerical stability across many shape points:
  * torch: mask/where + channel reductions (deterministic ordering).
  * cuTile _epilogue_kernel: fused BN-backward pointwise producer emitting
    the bf16 output in the same channels-last layout as the input. All
    per-channel scalars are precomputed as full-tensor broadcasts to avoid
    per-element channel-gather rounding hazards on non-pow2 C.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


SCALE = 6.228077168367346e-07


@ct.kernel
def _epilogue_kernel(
    where_ptr,       # bf16 [TOTAL] flat NHWC (memory order)
    centered_ptr,    # f32 [TOTAL] flat NHWC (per element x-mean)
    mean_term_ptr,   # f32 [TOTAL] (broadcast per channel: sum1*SCALE)
    correction_ptr,  # f32 [TOTAL] (broadcast per channel: sum2*SCALE*invstd^2)
    output_scale_ptr,# f32 [TOTAL] (broadcast per channel: invstd*weight)
    out_ptr,         # bf16 [TOTAL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    where_val = ct.load(where_ptr, index=(pid,), shape=(BLOCK,))
    centered = ct.load(centered_ptr, index=(pid,), shape=(BLOCK,))
    mean_term = ct.load(mean_term_ptr, index=(pid,), shape=(BLOCK,))
    correction = ct.load(correction_ptr, index=(pid,), shape=(BLOCK,))
    output_scale = ct.load(output_scale_ptr, index=(pid,), shape=(BLOCK,))

    where_f = ct.astype(where_val, ct.float32)
    grad = (where_f - centered * correction - mean_term) * output_scale
    ct.store(out_ptr, index=(pid,), tile=ct.astype(grad, ct.bfloat16))


def _compute_where_and_sums(arg0, arg1, arg2, arg3, arg4, arg5, arg6):
    """Torch reference computation of where-value and per-channel sums.
    Mirrors the eager graph's operator ordering for BF16 precision fidelity.
    """
    x_f = arg0.to(torch.float32)
    sub = x_f - arg1
    mul = sub * arg2
    weight_bc = arg3.view(1, -1, 1, 1)
    bias_bc = arg4.view(1, -1, 1, 1)
    mul1 = mul * weight_bc
    aff = mul1 + bias_bc
    aff_bf = aff.to(torch.bfloat16)
    cond = (aff_bf <= 0.0) | (aff_bf >= 6.0)
    where_bf = torch.where(cond, arg5, arg6)
    where_f = where_bf.to(torch.float32)
    sum1 = where_f.sum(dim=(0, 2, 3))
    centered = x_f - arg1
    sum2 = (where_f * centered).sum(dim=(0, 2, 3))
    return where_bf, sum1, sum2, centered


@oracle_impl(hardware="B200", point="8693ecd8")
@oracle_impl(hardware="B200", point="e4902295")
@oracle_impl(hardware="B200", point="cea53ab1")
@oracle_impl(hardware="B200", point="d3515732")
@oracle_impl(hardware="B200", point="72305b66")
@oracle_impl(hardware="B200", point="ddbc642b")
@oracle_impl(hardware="B200", point="20e7cc5d")
@oracle_impl(hardware="B200", point="8fbbedd9")
@oracle_impl(hardware="B200", point="cfc1c8b3")
@oracle_impl(hardware="B200", point="425108c3")
@oracle_impl(hardware="B200", point="47a686d2")
@oracle_impl(hardware="B200", point="1e7379f7")
@oracle_impl(hardware="B200", point="a5d13e74")
@oracle_impl(hardware="B200", point="aad9cbeb")
@oracle_impl(hardware="B200", point="2f19ce6a")
@oracle_impl(hardware="B200", point="aba87b74")
@oracle_impl(hardware="B200", point="3dce5391")
@oracle_impl(hardware="B200", point="0a38f4eb")
@oracle_impl(hardware="B200", point="5951d630")
@oracle_impl(hardware="B200", point="45529174")
@oracle_impl(hardware="B200", point="4785ab0c")
@oracle_impl(hardware="B200", point="7bc04cf2")
def oracle_forward(inputs):
    arg0, arg1, arg2, arg3, arg4, arg5, arg6 = inputs
    device = arg0.device
    n, c, h, w = int(arg0.shape[0]), int(arg0.shape[1]), int(arg0.shape[2]), int(arg0.shape[3])
    total = n * c * h * w

    # Torch reductions and where value (deterministic ordering).
    where_bf, sum1, sum2, centered_full = _compute_where_and_sums(
        arg0, arg1, arg2, arg3, arg4, arg5, arg6
    )
    scale_grad = sum2 * arg2.view(-1)

    # Precompute per-channel epilogue scalars, then broadcast to full tensor.
    invstd = arg2.view(c)
    weight = arg3
    mean_term_c = sum1 * SCALE
    correction_c = sum2 * SCALE * invstd * invstd
    output_scale_c = invstd * weight

    # Broadcast (C,) into (N, C, H, W) channels-last physical layout, then
    # view flat.
    def _bc(vec_c):
        # channels-last strides for (N, C, H, W): (C*H*W, 1, W*C, C)
        return vec_c.view(1, c, 1, 1).expand(n, c, h, w).contiguous(
            memory_format=torch.channels_last
        )

    mean_term_bc = _bc(mean_term_c)
    correction_bc = _bc(correction_c)
    output_scale_bc = _bc(output_scale_c)

    # Flat NHWC views of memory-contiguous channels-last tensors.
    where_flat = torch.as_strided(where_bf, (total,), (1,))
    centered_flat = torch.as_strided(
        centered_full.contiguous(memory_format=torch.channels_last),
        (total,), (1,),
    )
    mean_term_flat = torch.as_strided(mean_term_bc, (total,), (1,))
    correction_flat = torch.as_strided(correction_bc, (total,), (1,))
    output_scale_flat = torch.as_strided(output_scale_bc, (total,), (1,))
    out_flat = torch.empty(total, device=device, dtype=torch.bfloat16)

    BLOCK = 256
    stream = torch.cuda.current_stream()
    ct.launch(
        stream, ((total + BLOCK - 1) // BLOCK, 1, 1), _epilogue_kernel,
        (where_flat, centered_flat, mean_term_flat, correction_flat,
         output_scale_flat, out_flat, BLOCK),
    )

    # NHWC memory reinterpreted as NCHW view with channels-last strides.
    out = torch.as_strided(out_flat, (n, c, h, w), (c * h * w, 1, w * c, c))
    return sum1, scale_grad, out
