"""cuTile port of pointwise_5a0f1ca0cc0c (SCHEDULER_FUSION): DenseNet
virtual channel-cat + BN-inference affine + NaN-preserving ReLU on bf16.

Each output element at flat position `off` corresponds to
    (batch, channel, hw) with channel < C0 sourcing x0[batch, channel, hw]
    and channel >= C0 sourcing x1[batch, channel - C0, hw]. BN parameters
    index by the concatenated channel.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


EPS = 1.0e-5


@ct.kernel
def _virtual_cat_bn_relu_kernel(
    x0_ptr,       # bf16 [batch*C0*HW]
    x1_ptr,       # bf16 [batch*C1*HW]
    mean_ptr,     # bf16 [CHANNELS]
    var_ptr,      # bf16 [CHANNELS]
    weight_ptr,   # bf16 [CHANNELS]
    bias_ptr,     # bf16 [CHANNELS]
    out_ptr,      # bf16 [batch*CHANNELS*HW]
    C0: ct.Constant[int],
    C1: ct.Constant[int],
    CHANNELS: ct.Constant[int],
    HW_C: ct.Constant[int],
    BLOCK: ct.Constant[int],
):
    pid = ct.bid(0)
    lane = ct.arange(BLOCK, dtype=ct.int64) + ct.astype(pid, ct.int64) * BLOCK

    hw = lane - (lane // HW_C) * HW_C
    tmp = lane // HW_C
    channel = tmp - (tmp // CHANNELS) * CHANNELS
    batch = tmp // CHANNELS

    in_x0 = channel < C0
    # x0[batch, channel, hw] = flat[(batch*C0 + channel)*HW + hw]
    x0_off = (batch * C0 + channel) * HW_C + hw
    x1_channel = channel - C0
    x1_off = (batch * C1 + x1_channel) * HW_C + hw

    zero_off = ct.astype(0, ct.int64)
    safe_x0_off = ct.where(in_x0, x0_off, zero_off)
    safe_x1_off = ct.where(in_x0, zero_off, x1_off)
    x0 = ct.astype(ct.gather(x0_ptr, safe_x0_off), ct.float32)
    x1 = ct.astype(ct.gather(x1_ptr, safe_x1_off), ct.float32)
    x = ct.where(in_x0, x0, x1)

    mean = ct.astype(ct.gather(mean_ptr, channel), ct.float32)
    var = ct.astype(ct.gather(var_ptr, channel), ct.float32)
    weight = ct.astype(ct.gather(weight_ptr, channel), ct.float32)
    bias = ct.astype(ct.gather(bias_ptr, channel), ct.float32)

    invstd = 1.0 / ct.sqrt(var + EPS)
    affine = (x - mean) * invstd * weight + bias
    out_bf16 = ct.astype(affine, ct.bfloat16)
    # NaN-preserving ReLU
    out_f = ct.astype(out_bf16, ct.float32)
    zero_bf = ct.astype(0, ct.bfloat16)
    keep = (out_f > 0.0) | ct.isnan(out_f)
    out_final = ct.where(keep, out_bf16, zero_bf)
    ct.store(out_ptr, index=(pid,), tile=out_final)


def _pick_block(n):
    for cand in (1024, 512, 256, 128, 64, 32, 16, 8):
        if n % cand == 0:
            return cand
    return 1


@oracle_impl(hardware="B200", point="732e3257")
@oracle_impl(hardware="B200", point="dca482b1")
@oracle_impl(hardware="B200", point="a3bd7f06")
@oracle_impl(hardware="B200", point="cfae6d5c")
@oracle_impl(hardware="B200", point="8aa233bf")
@oracle_impl(hardware="B200", point="1b91efd5")
@oracle_impl(hardware="B200", point="320237ea")
def oracle_forward(inputs):
    x0, x1, mean, var, weight, bias = inputs
    batch = int(x0.shape[0])
    c0 = int(x0.shape[1])
    c1 = int(x1.shape[1])
    height = int(x0.shape[2])
    width = int(x0.shape[3])
    channels = c0 + c1
    hw = height * width
    out = torch.empty_strided(
        (batch, channels, height, width),
        (channels * hw, hw, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    total = batch * channels * hw
    BLOCK = _pick_block(total)
    x0_flat = x0.contiguous().view(-1)
    x1_flat = x1.contiguous().view(-1)
    out_flat = out.view(-1)
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (total // BLOCK, 1, 1),
        _virtual_cat_bn_relu_kernel,
        (x0_flat, x1_flat, mean, var, weight, bias, out_flat,
         c0, c1, channels, hw, BLOCK),
    )
    return out
