"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete isolated bf16 two-input `aten.add.Tensor` from `Repro.forward` as one Triton pointwise kernel while preserving eager's fresh dense output layout, using a storage-linear path for the channels-last NCHW points and a separate Demucs tiled path for the mixed contiguous/transposed input strides; Inductor already lowers this one-op repro to a single pointwise kernel with the same mandatory two-read/one-write memory traffic, and there is no producer/consumer fusion, algebraic elimination, split-K, or scatter-reduce opportunity available inside this local graph; the fix is BANDWIDTH_BOUND: record the pointwise memory-bandwidth floor unless broader pointwise codegen or launch-overhead improvements move both implementations."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _add_dense_kernel(x_ptr, y_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@triton.jit
def _add_demucs_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    t = tl.program_id(2) * BLOCK_T + tl.arange(0, BLOCK_T)
    mask = (c[:, None] < C) & (t[None, :] < T)

    out_offsets = b * C + c[:, None] + t[None, :] * (B * C)
    y_offsets = b * C * T + c[:, None] * T + t[None, :]
    x = tl.load(x_ptr + out_offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + y_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, x + y, mask=mask)


@oracle_impl(hardware="B200", point="092aaddd", BLOCK=256, num_warps=4)
@oracle_impl(hardware="B200", point="fd885e3f", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="fe14aec8", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="f30f68d5", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="b4ab9b24", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="8bac800c", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="8b187706", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="7c24d0c8", BLOCK=2048, num_warps=8)
@oracle_impl(hardware="B200", point="d0800b9f", BLOCK=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    x, y = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in x.shape),
        tuple(int(stride) for stride in x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    _add_dense_kernel[(triton.cdiv(x.numel(), BLOCK),)](
        x,
        y,
        out,
        N=x.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out


@oracle_impl(hardware="B200", point="79c39c98", B=8, C=2048, T=92, BLOCK_C=128, BLOCK_T=8, num_warps=4)
@oracle_impl(hardware="B200", point="1ffb91db", B=4, C=2048, T=92, BLOCK_C=128, BLOCK_T=8, num_warps=4)
def oracle_forward_demucs(
    inputs,
    *,
    B: int,
    C: int,
    T: int,
    BLOCK_C: int,
    BLOCK_T: int,
    num_warps: int,
):
    x, y = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in x.shape),
        tuple(int(stride) for stride in x.stride()),
        device=x.device,
        dtype=x.dtype,
    )
    _add_demucs_kernel[(B, triton.cdiv(C, BLOCK_C), triton.cdiv(T, BLOCK_T))](
        x,
        y,
        out,
        B=B,
        C=C,
        T=T,
        BLOCK_C=BLOCK_C,
        BLOCK_T=BLOCK_T,
        num_warps=num_warps,
    )
    return out
