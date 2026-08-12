"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Demucs bf16 slice/ReLU/add pointwise scope in one storage-linear Triton kernel, including NaN-preserving `relu(arg0)`, direct reads from `arg1[:, :, 1426:-1426]`, bf16 output rounding, and the fresh contiguous `[8,64,92844]` result. Inductor already fuses this local graph, but lowers it through the generic pointwise scheduler for a large fixed-offset slice; it cannot do better today without a guarded Demucs-style slice-plus-activation pointwise template or equivalent autotuned specialization that strips the generic slice indexing overhead while preserving bf16 dtype boundaries. The fix is NEW_PATTERN: add a shape-specialized slice/ReLU/add pointwise lowering for this dense offset-slice family, and otherwise record the generic fused pointwise path as the memory-bandwidth floor."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 64
TIME = 92844
ARG_TIME = 95696
SLICE_START = 1426
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@triton.jit
def _slice_relu_add_kernel(
    conv_ptr,
    arg_ptr,
    out_ptr,
    N: tl.constexpr,
    T: tl.constexpr,
    ARG_T: tl.constexpr,
    SLICE_START_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    row = offsets // T
    col = offsets - row * T
    offsets = row * T + col

    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    sliced = tl.load(
        arg_ptr + row * ARG_T + SLICE_START_ + col,
        mask=mask,
        other=0.0,
    )
    relu = tl.where(conv != conv, conv, tl.maximum(conv, 0.0))
    tl.store(out_ptr + offsets, relu + sliced, mask=mask)


@oracle_impl(hardware="B200", point="042ea2d0", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    conv, arg = inputs
    out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bfloat16,
    )
    _slice_relu_add_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        conv,
        arg,
        out,
        N=NUMEL,
        T=TIME,
        ARG_T=ARG_TIME,
        SLICE_START_=SLICE_START,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
