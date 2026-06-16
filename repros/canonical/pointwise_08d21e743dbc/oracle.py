"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Demucs bf16 ReLU/slice/add/mask scope in one storage-linear Triton kernel, including NaN-preserving `relu(arg0_1)`, direct reads from `arg1_1[:, :, 355:-356]`, bf16 add/output rounding, and the returned `relu <= 0` bool mask folded to the equivalent `arg0_1 <= 0` predicate while sharing the single activation load, whereas Inductor lowers this local scope through its generic fused pointwise scheduler; Inductor cannot specialize this fixed-offset dense slice family today because pointwise codegen has no guarded Demucs ReLU-plus-centered-slice template that strips generic slice indexing while preserving the side mask and bf16 dtype boundaries; the fix is NEW_PATTERN: add a shape-specialized slice/ReLU/add/mask pointwise lowering for this Demucs family."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 4
CHANNELS = 128
TIME = 23212
ARG_TIME = 23923
SLICE_START = 355
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


@oracle_impl(hardware="B200", point="10dba222", BLOCK=8192, num_warps=4)
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
