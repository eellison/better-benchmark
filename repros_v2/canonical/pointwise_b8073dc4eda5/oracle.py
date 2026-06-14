"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete Demucs bf16 ReLU/slice/add/mask scope in one storage-linear Triton kernel, including NaN-preserving `relu(arg0)`, direct reads from `arg1[:, :, 20:-21]`, bf16 add/output rounding, and the returned `relu <= 0` bool mask folded to the equivalent `arg0 <= 0` predicate while sharing the single activation load. Inductor already lowers this local scope to one fused pointwise kernel that reads both dense inputs and writes both dense outputs, so there is no reduction, scatter, split-K, or recomputation opportunity left inside the repro. The fix is BANDWIDTH_BOUND: record this as a pointwise memory floor unless broader pointwise bandwidth, launch, or allocation changes move both paths together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 512
TIME = 1452
ARG_TIME = 1493
SLICE_START = 20
NUMEL = BATCH * CHANNELS * TIME
OUT_SHAPE = (BATCH, CHANNELS, TIME)
OUT_STRIDE = (CHANNELS * TIME, TIME, 1)


@triton.jit
def _relu_slice_add_mask_kernel(
    conv_ptr,
    arg_ptr,
    add_out_ptr,
    mask_out_ptr,
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
    arg_offsets = row * arg_t + slice_start + col

    conv = tl.load(conv_ptr + offsets, mask=mask, other=0.0)
    sliced = tl.load(arg_ptr + arg_offsets, mask=mask, other=0.0)
    relu = tl.where(conv != conv, conv, tl.maximum(conv, 0.0))
    tl.store(add_out_ptr + offsets, relu + sliced, mask=mask)
    tl.store(mask_out_ptr + offsets, conv <= 0.0, mask=mask)


@oracle_impl(hardware="B200", point="46ecc6c3", BLOCK=8192, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    conv, arg = inputs
    add_out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bfloat16,
    )
    mask_out = torch.empty_strided(
        OUT_SHAPE,
        OUT_STRIDE,
        device=conv.device,
        dtype=torch.bool,
    )
    _relu_slice_add_mask_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        conv,
        arg,
        add_out,
        mask_out,
        n_elements=NUMEL,
        t=TIME,
        arg_t=ARG_TIME,
        slice_start=SLICE_START,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return add_out, mask_out
