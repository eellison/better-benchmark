"""cuTile port of sum_sum_d405766683ac: MobileNetV3 BN + hard-sigmoid gate.

Reference computes BN(arg0) + relu, mul by arg5 (bf16), spatial sum, then
hard-sigmoid gate on the resulting scalar per (n,c), and finally a
per-channel sum for BN-backward. Uses cuTile for the BN + relu producer
kernel (main compute), torch for the reductions and small tensor ops.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bn_relu_kernel(
    x_ptr,       # bf16[NUMEL] (flat channels-last)
    mean_ptr,    # f32[NUMEL] broadcast per-channel
    scale_ptr,   # f32[NUMEL] broadcast per-channel
    weight_ptr,  # f32[NUMEL] broadcast per-channel
    bias_ptr,    # f32[NUMEL] broadcast per-channel
    out_ptr,     # bf16[NUMEL]
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    x = ct.astype(ct.load(x_ptr, index=(pid,), shape=(BLOCK,)), ct.float32)
    mean = ct.load(mean_ptr, index=(pid,), shape=(BLOCK,))
    scale = ct.load(scale_ptr, index=(pid,), shape=(BLOCK,))
    weight = ct.load(weight_ptr, index=(pid,), shape=(BLOCK,))
    bias = ct.load(bias_ptr, index=(pid,), shape=(BLOCK,))
    normalized = (x - mean) * scale
    affine = normalized * weight + bias
    val_bf16 = ct.astype(affine, ct.bfloat16)
    zero = ct.astype(ct.zeros((BLOCK,), dtype=ct.bfloat16), ct.bfloat16)
    relu = ct.where(val_bf16 > zero, val_bf16, zero)
    ct.store(out_ptr, index=(pid,), tile=relu)


def _run(inputs, *, BLOCK):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = inputs
    n, c, h, w = arg0_1.shape
    hw = h * w
    numel = n * c * hw
    device = arg0_1.device

    # arg0_1 is channels-last bf16. Move to (N, H, W, C) contiguous.
    x_phys = arg0_1.permute(0, 2, 3, 1).contiguous()
    x_flat = x_phys.view(numel)

    # Broadcast BN parameters to (N, H, W, C) contiguous. arg1, arg2 are
    # (1,C,1,1) f32; arg3, arg4 are (C,) f32.
    mean_bc = arg1_1.expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(numel)
    scale_bc = arg2_1.expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(numel)
    weight_bc = arg3_1.view(1, c, 1, 1).expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(numel)
    bias_bc = arg4_1.view(1, c, 1, 1).expand(n, c, h, w).permute(0, 2, 3, 1).contiguous().view(numel)
    relu_flat = torch.empty(numel, device=device, dtype=torch.bfloat16)

    stream = torch.cuda.current_stream()
    ct.launch(stream, (ct.cdiv(numel, BLOCK), 1, 1), _bn_relu_kernel,
              (x_flat, mean_bc, scale_bc, weight_bc, bias_bc, relu_flat, BLOCK))

    # Reshape relu_flat back to (N, C, H, W) with channels-last strides.
    relu_bf16 = relu_flat.view(n, h, w, c).permute(0, 3, 1, 2).contiguous(
        memory_format=torch.channels_last
    )

    # mul_2 = arg5 * relu (bf16 * bf16)
    mul_2 = arg5_1 * relu_bf16
    # sum_1 = mul_2.sum([2,3], keepdim=True, dtype=f32) -> f32[N, C, 1, 1]
    sum_1 = mul_2.sum(dim=[2, 3], keepdim=True, dtype=torch.float32)

    convert_1 = arg6_1.to(torch.float32)                # f32[N, C, 1, 1]
    convert_3 = sum_1.to(torch.bfloat16).to(torch.float32)

    mask = (convert_1 > -3.0) & (convert_1 < 3.0)
    mul_3 = convert_3 * 0.16666666666666666
    where_val = torch.where(mask, mul_3, arg7_1)
    convert_4 = where_val.to(torch.bfloat16)

    sum_2 = convert_4.sum(dim=[0, 2, 3])                 # bf16 -> f32 accum, output bf16
    convert_5 = sum_2.to(torch.float32)

    return relu_bf16, convert_1, convert_4, convert_5


@oracle_impl(hardware="B200", point="0e39883f", BLOCK=1024)
@oracle_impl(hardware="B200", point="ee700829", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK):
    return _run(inputs, BLOCK=BLOCK)
