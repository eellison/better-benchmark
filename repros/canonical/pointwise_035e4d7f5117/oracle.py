"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full shape-param `aten.full.default([204, 204, 26], 1, dtype=bfloat16)` scope by allocating the fresh contiguous CUDA bf16 output and writing the exact packed bf16 `1.0` bit pattern with one Triton fill kernel, whereas Inductor lowers the standalone full through its generic pointwise constant-store scheduler; Inductor cannot do this today because codegen has no dedicated static constant-full lowering that bypasses generic elementwise indexing and emits packed final-dtype stores for bf16 fills; the fix is NEW_PATTERN: add a guarded constant-full template for static CUDA `aten.full` that writes the required output layout directly."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ONE_BF16X2_BITS = 0x3F803F80


@triton.jit
def _fill_one_bf16x2_kernel(
    out_bits_ptr,
    n_words: tl.constexpr,
    bits: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_words
    values = tl.full((BLOCK,), bits, tl.uint32)
    tl.store(out_bits_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    shape = tuple(inputs[0])
    stride = (shape[1] * shape[2], shape[2], 1)
    n_words = (shape[0] * shape[1] * shape[2]) // 2
    out = torch.empty_strided(
        shape,
        stride,
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    out_bits = out.reshape(-1).view(torch.uint32)
    grid = (triton.cdiv(n_words, BLOCK),)
    _fill_one_bf16x2_kernel[grid](
        out_bits,
        n_words=n_words,
        bits=ONE_BF16X2_BITS,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return out
