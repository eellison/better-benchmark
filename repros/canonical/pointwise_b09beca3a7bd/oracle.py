"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full 12-input bf16-to-fp32 ALBERT residual accumulation as one storage-linear Triton kernel, explicitly converting each bf16 input to fp32 and preserving the captured left-associated add order before writing the single contiguous fp32 output, whereas Inductor lowers the same isolated chain through generic pointwise fusion; Inductor cannot do this today because pointwise scheduling has no guarded B200 template for high-arity bf16 input accumulation that specializes register use and launch shape for this transformer matrix family; the fix is NEW_PATTERN: add a dedicated high-arity bf16-to-fp32 accumulation pointwise template or equivalent autotuned specialization for residual-add fan-in patterns."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _sum12_bf16_to_f32_kernel(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    out,
    n_elements: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    acc = tl.load(x0 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x1 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x2 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x3 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x4 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x5 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x6 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x7 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x8 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x9 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x10 + offsets, mask=mask, other=0.0).to(tl.float32)
    acc = acc + tl.load(x11 + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + offsets, acc, mask=mask)


# 26d25975: 12 x T([4096,4096], bf16)
@oracle_impl(hardware="B200", point="26d25975", BLOCK=1024, num_warps=4)
# 69f8f171: 12 x T([16384,4096], bf16)
@oracle_impl(hardware="B200", point="69f8f171", BLOCK=1024, num_warps=4)
# cf776752: 12 x T([4096,16384], bf16)
@oracle_impl(hardware="B200", point="cf776752", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    tensors = inputs
    rows, cols = tensors[0].shape
    out = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=tensors[0].device,
        dtype=torch.float32,
    )
    _sum12_bf16_to_f32_kernel[(triton.cdiv(out.numel(), BLOCK),)](
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        tensors[4],
        tensors[5],
        tensors[6],
        tensors[7],
        tensors[8],
        tensors[9],
        tensors[10],
        tensors[11],
        out,
        out.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
