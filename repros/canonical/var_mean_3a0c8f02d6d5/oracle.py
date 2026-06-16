"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 CycleGAN instance-normalization, bf16 cast, ReLU, and two-axis reflected unsafe-index scope for `[1,256,64,64]`, including the returned contiguous bf16 ReLU tensor and the returned contiguous bf16 `[1,256,66,66]` reflected layout, whereas Inductor schedules the norm-template reduction/epilogue and downstream reflected indexing as separate materialized regions; Inductor cannot do this today because its normalization scheduler does not treat fixed reflected-index consumers and bf16 activation boundaries as direct output-layout epilogues of the norm result; the fix is SCHEDULER_FUSION: fuse static reflected-index layout consumers into the normalization epilogue and emit both direct ReLU storage and reflected bf16 storage from one fixed-shape instance-norm plan."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _relu_preserve_nan(x):
    return tl.where((x > 0.0) | (x != x), x, 0.0)


@triton.jit
def _channel_norm_relu_reflect_kernel(
    x_ptr,
    relu_ptr,
    reflect_ptr,
    HW: tl.constexpr,
    OUT_HW: tl.constexpr,
    PADDED: tl.constexpr,
    WIDTH: tl.constexpr,
    STAT_BLOCK: tl.constexpr,
    OUT_BLOCK: tl.constexpr,
):
    channel = tl.program_id(0)
    stat_offsets = tl.arange(0, STAT_BLOCK)
    x = tl.load(x_ptr + channel * HW + stat_offsets).to(tl.float32)

    mean_acc = tl.zeros([STAT_BLOCK], tl.float32)
    m2_acc = tl.zeros([STAT_BLOCK], tl.float32)
    weight_acc = tl.zeros([STAT_BLOCK], tl.float32)
    mean_next, m2_next, weight_next = triton_helpers.welford_reduce(
        x,
        mean_acc,
        m2_acc,
        weight_acc,
        True,
    )
    mean, m2, _weight = triton_helpers.welford(mean_next, m2_next, weight_next, 0)
    invstd = libdevice.rsqrt((m2 / HW) + 1.0e-5)

    normalized = (x - mean) * invstd
    rounded = normalized.to(tl.bfloat16)
    relu = _relu_preserve_nan(rounded)
    tl.store(relu_ptr + channel * HW + stat_offsets, relu)

    out_base = channel * OUT_HW
    out_arange = tl.arange(0, OUT_BLOCK)
    for out_start in tl.range(0, OUT_HW, OUT_BLOCK):
        out_offsets = out_start + out_arange
        mask = out_offsets < OUT_HW
        oh = out_offsets // PADDED
        ow = out_offsets - oh * PADDED

        ih = tl.where(oh == 0, 1, tl.where(oh == PADDED - 1, WIDTH - 2, oh - 1))
        iw = tl.where(ow == 0, 1, tl.where(ow == PADDED - 1, WIDTH - 2, ow - 1))
        input_offsets = channel * HW + ih * WIDTH + iw

        y = tl.load(x_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
        reflected = (y - mean) * invstd
        reflected_rounded = reflected.to(tl.bfloat16)
        reflected_relu = _relu_preserve_nan(reflected_rounded)
        tl.store(reflect_ptr + out_base + out_offsets, reflected_relu, mask=mask)


@oracle_impl(
    hardware="B200",
    point="1b4e0bc8",
    STAT_BLOCK=4096,
    OUT_BLOCK=512,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    STAT_BLOCK: int,
    OUT_BLOCK: int,
    num_warps: int,
    num_stages: int,
):
    (arg0_1,) = inputs
    channels = int(arg0_1.shape[1])
    height = int(arg0_1.shape[2])
    width = int(arg0_1.shape[3])
    hw = height * width
    padded = width + 2
    out_hw = padded * padded

    relu = torch.empty_strided(
        (1, channels, height, width),
        (channels * hw, hw, width, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    reflected = torch.empty_strided(
        (1, channels, padded, padded),
        (channels * out_hw, out_hw, padded, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _channel_norm_relu_reflect_kernel[(channels,)](
        arg0_1,
        relu,
        reflected,
        HW=hw,
        OUT_HW=out_hw,
        PADDED=padded,
        WIDTH=width,
        STAT_BLOCK=STAT_BLOCK,
        OUT_BLOCK=OUT_BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return relu, reflected
