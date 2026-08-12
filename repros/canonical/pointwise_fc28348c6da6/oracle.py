"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete XLNet sinusoidal positional-embedding fanout scope in one Triton pointwise kernel, including the descending position iota, even-dimension frequency iota, f32 `10000 ** (i / 1024)` reciprocal, natural sin/cos, explicit bf16 rounding, the expand/clone/view/squeeze layout as fresh contiguous `[16384,1024]` tensors, and all 24 independent graph outputs. Inductor currently lowers the shared sin/cos table producer and the 24 repeated bf16 convert/view/squeeze outputs as generic pointwise/layout materializations; it cannot do this today because pointwise scheduling does not group identical shape-derived graph-output siblings while preserving distinct fresh output storages and transcendental dtype boundaries. The fix is SCHEDULER_FUSION: add graph-output sibling fusion for repeated structured positional-embedding materializations so one schedule computes the f32 sinusoid value once and emits every required bf16 output store."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 16384
HIDDEN = 1024
HALF_HIDDEN = 512
OUT_SHAPE = (ROWS, HIDDEN)
OUT_STRIDE = (HIDDEN, 1)
NUMEL = ROWS * HIDDEN


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _xlnet_positional_fanout_kernel(
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    out12,
    out13,
    out14,
    out15,
    out16,
    out17,
    out18,
    out19,
    out20,
    out21,
    out22,
    out23,
    N: tl.constexpr,
    H: tl.constexpr,
    HALF_H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    col = offsets % H
    row = offsets // H
    pos_index = row // 16
    pos = (512 - pos_index).to(tl.float32)
    freq_index = tl.where(col < HALF_H, col, col - HALF_H)

    even_dim = _f32_mul(freq_index.to(tl.float32), 2.0)
    exponent = _f32_div(even_dim, 1024.0)
    denom = libdevice.pow(tl.full((BLOCK,), 10000.0, tl.float32), exponent)
    inv_freq = _f32_div(1.0, denom)
    phase = _f32_mul(pos, inv_freq)
    f32_value = tl.where(col < HALF_H, libdevice.sin(phase), libdevice.cos(phase))
    value = f32_value.to(tl.bfloat16, fp_downcast_rounding="rtne")

    tl.store(out0 + offsets, value, mask=mask)
    tl.store(out1 + offsets, value, mask=mask)
    tl.store(out2 + offsets, value, mask=mask)
    tl.store(out3 + offsets, value, mask=mask)
    tl.store(out4 + offsets, value, mask=mask)
    tl.store(out5 + offsets, value, mask=mask)
    tl.store(out6 + offsets, value, mask=mask)
    tl.store(out7 + offsets, value, mask=mask)
    tl.store(out8 + offsets, value, mask=mask)
    tl.store(out9 + offsets, value, mask=mask)
    tl.store(out10 + offsets, value, mask=mask)
    tl.store(out11 + offsets, value, mask=mask)
    tl.store(out12 + offsets, value, mask=mask)
    tl.store(out13 + offsets, value, mask=mask)
    tl.store(out14 + offsets, value, mask=mask)
    tl.store(out15 + offsets, value, mask=mask)
    tl.store(out16 + offsets, value, mask=mask)
    tl.store(out17 + offsets, value, mask=mask)
    tl.store(out18 + offsets, value, mask=mask)
    tl.store(out19 + offsets, value, mask=mask)
    tl.store(out20 + offsets, value, mask=mask)
    tl.store(out21 + offsets, value, mask=mask)
    tl.store(out22 + offsets, value, mask=mask)
    tl.store(out23 + offsets, value, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=512, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    del inputs
    device = torch.device("cuda", 0)
    outputs = tuple(
        torch.empty_strided(OUT_SHAPE, OUT_STRIDE, device=device, dtype=torch.bfloat16)
        for _ in range(24)
    )

    _xlnet_positional_fanout_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        *outputs,
        N=NUMEL,
        H=HIDDEN,
        HALF_H=HALF_HIDDEN,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return outputs
