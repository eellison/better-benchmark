"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs bf16 slice/ReLU/add pointwise scope in one storage-linear Triton kernel, including NaN-preserving `relu(arg0)`, direct reads from `arg1[:, :, 87:-88]`, bf16 add/output rounding, and the fresh contiguous `[8, 256, 5804]` result. Inductor already lowers this local graph as one fused pointwise expression, and cannot materially reduce the remaining work through scheduler fusion, scatter/reduce, split-K, or recomputation because the exact output contract requires two dense bf16 input reads, the fixed-offset slice index, and one dense bf16 output store. The fix is BANDWIDTH_BOUND: record this as a pointwise memory floor unless broader pointwise indexing, bandwidth, or launch-overhead changes move both paths."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 8
CHANNELS = 256
TIME = 5804
ARG_TIME = 5979
SLICE_START = 87
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@triton.jit
def _slice_relu_add_kernel(
    conv_ptr,
    arg_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    t: tl.constexpr,
    arg_t: tl.constexpr,
    slice_start: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    row = offsets // t
    col = offsets - row * t

    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    sliced = tl.load(arg_ptr + row * arg_t + slice_start + col, mask=mask, other=0.0)
    relu = tl.where(conv != conv, conv, tl.maximum(conv, 0.0))
    tl.store(out_ptr + offsets, relu + sliced, mask=mask)


@oracle_impl(hardware="B200", point="839c4ad7", BLOCK=1024, num_warps=4)
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
        n_elements=NUMEL,
        t=TIME,
        arg_t=ARG_TIME,
        slice_start=SLICE_START,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=4,
    )
    return out
