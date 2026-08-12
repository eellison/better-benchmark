"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete DenseNet bf16 channel cat -> BN-inference affine -> bf16 cast -> ReLU scope in one Triton kernel, reading the concat inputs as a virtual channel layout and storing only the returned contiguous activation, whereas Inductor materializes the fixed channel concat before the downstream normalization pointwise kernel; Inductor cannot do this today because its scheduler does not model aten.cat as a virtual multi-source producer that can be inlined into channel-dependent pointwise consumers with per-channel parameter loads; the fix is SCHEDULER_FUSION: allow fixed-shape channel concat producers to feed fused pointwise consumers directly with guarded source selection instead of forcing a dense concatenated intermediate."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


EPS = 1.0e-5


@triton.jit
def _virtual_cat_bn_relu_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    x3_ptr,
    x4_ptr,
    x5_ptr,
    x6_ptr,
    x7_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    FIRST_C: tl.constexpr,
    TOTAL_C: tl.constexpr,
    HW: tl.constexpr,
    EPS_VALUE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    batch = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    spatial = tl.program_id(2) * BLOCK_HW + tl.arange(0, BLOCK_HW)
    valid_c = channels < TOTAL_C
    valid_hw = spatial < HW
    valid = valid_c[:, None] & valid_hw[None, :]

    rel = channels - FIRST_C
    use0 = channels < FIRST_C
    use1 = (rel >= 0) & (rel < 32)
    use2 = (rel >= 32) & (rel < 64)
    use3 = (rel >= 64) & (rel < 96)
    use4 = (rel >= 96) & (rel < 128)
    use5 = (rel >= 128) & (rel < 160)
    use6 = (rel >= 160) & (rel < 192)
    use7 = rel >= 192

    c0 = tl.where(use0, channels, 0)
    c1 = tl.where(use1, rel, 0)
    c2 = tl.where(use2, rel - 32, 0)
    c3 = tl.where(use3, rel - 64, 0)
    c4 = tl.where(use4, rel - 96, 0)
    c5 = tl.where(use5, rel - 128, 0)
    c6 = tl.where(use6, rel - 160, 0)
    c7 = tl.where(use7, rel - 192, 0)

    hw = spatial[None, :]
    x0_offsets = (batch * FIRST_C + c0[:, None]) * HW + hw
    tail_batch = batch * 32 * HW
    x1_offsets = tail_batch + c1[:, None] * HW + hw
    x2_offsets = tail_batch + c2[:, None] * HW + hw
    x3_offsets = tail_batch + c3[:, None] * HW + hw
    x4_offsets = tail_batch + c4[:, None] * HW + hw
    x5_offsets = tail_batch + c5[:, None] * HW + hw
    x6_offsets = tail_batch + c6[:, None] * HW + hw
    x7_offsets = tail_batch + c7[:, None] * HW + hw

    x = tl.load(x0_ptr + x0_offsets, mask=valid & use0[:, None], other=0.0).to(tl.float32)
    x += tl.load(x1_ptr + x1_offsets, mask=valid & use1[:, None], other=0.0).to(tl.float32)
    x += tl.load(x2_ptr + x2_offsets, mask=valid & use2[:, None], other=0.0).to(tl.float32)
    x += tl.load(x3_ptr + x3_offsets, mask=valid & use3[:, None], other=0.0).to(tl.float32)
    x += tl.load(x4_ptr + x4_offsets, mask=valid & use4[:, None], other=0.0).to(tl.float32)
    x += tl.load(x5_ptr + x5_offsets, mask=valid & use5[:, None], other=0.0).to(tl.float32)
    x += tl.load(x6_ptr + x6_offsets, mask=valid & use6[:, None], other=0.0).to(tl.float32)
    x += tl.load(x7_ptr + x7_offsets, mask=valid & use7[:, None], other=0.0).to(tl.float32)

    mean = tl.load(mean_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + channels, mask=valid_c, other=0.0).to(tl.float32)

    sqrt = libdevice.sqrt(var + EPS_VALUE)
    invstd = (1.0 / sqrt) * 1.0
    y = ((x - mean[:, None]) * invstd[:, None]) * weight[:, None] + bias[:, None]
    y_bf16 = y.to(tl.bfloat16)
    zero = tl.full((BLOCK_C, BLOCK_HW), 0.0, tl.bfloat16)
    relu = tl.where(y_bf16 < zero, zero, y_bf16)

    out_offsets = (batch * TOTAL_C + channels[:, None]) * HW + spatial[None, :]
    tl.store(out_ptr + out_offsets, relu, mask=valid)


def _launch(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int):
    x0, x1, x2, x3, x4, x5, x6, x7, mean, var, weight, bias = inputs
    batch, first_c, height, width = x0.shape
    total_c = mean.numel()
    hw = height * width
    out = torch.empty_strided(
        (batch, total_c, height, width),
        (total_c * hw, hw, width, 1),
        device=x0.device,
        dtype=torch.bfloat16,
    )
    _virtual_cat_bn_relu_kernel[
        (batch, triton.cdiv(total_c, BLOCK_C), triton.cdiv(hw, BLOCK_HW))
    ](
        x0,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        mean,
        var,
        weight,
        bias,
        out,
        FIRST_C=first_c,
        TOTAL_C=total_c,
        HW=hw,
        EPS_VALUE=EPS,
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


@oracle_impl(hardware="B200", point="d9ea527b", BLOCK_C=16, BLOCK_HW=64, num_warps=4)
@oracle_impl(hardware="B200", point="0046eb0d", BLOCK_C=8, BLOCK_HW=128, num_warps=4)
@oracle_impl(hardware="B200", point="55016e8d", BLOCK_C=4, BLOCK_HW=256, num_warps=4)
def oracle_forward(inputs, *, BLOCK_C: int, BLOCK_HW: int, num_warps: int):
    return _launch(inputs, BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW, num_warps=num_warps)
