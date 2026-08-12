"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete XLNet relative-shift bf16 attention softmax scope, including the singleton view/permute aliases, sliced relative-position index, bf16 add and scale rounding boundaries, finite-row `any` guard, both fp32 stable amax paths, natural exp/sum/div, final bf16 cast, and contiguous `[256, 512, 512]` output view in one Triton row kernel, whereas Inductor lowers the decomposed view/permute/slice/index/add/cast/finite-check/amax/exp/sum/div/cast/view graph as generic indexing, pointwise, and reduction work; Inductor cannot do this today because its pattern library does not recognize XLNet's relative-shift attention-score construction as a structured producer that can be sunk into the row-softmax schedule while preserving bf16 rounding and exceptional-value semantics; the fix is NEW_PATTERN: add an Inductor lowering for XLNet relative-shift attention softmax that fuses the shifted relative-position load, score add, row-validity test, normalization, and output-layout epilogue into one generated kernel."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


@triton.jit
def _xlnet_relative_shift_softmax_kernel(
    content_ptr,
    rel_ptr,
    out_ptr,
    N_ROWS: tl.constexpr,
    K_LEN: tl.constexpr,
    REL_WIDTH: tl.constexpr,
    REL_STRIDE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < N_ROWS
    col_mask = cols < K_LEN
    mask = row_mask[:, None] & col_mask[None, :]

    group = rows // K_LEN
    query = rows - group * K_LEN
    content_offsets = rows[:, None] * K_LEN + cols[None, :]
    rel_offsets = group[:, None] * REL_STRIDE + K_LEN + query[:, None] * REL_WIDTH + cols[None, :]

    content = tl.load(content_ptr + content_offsets, mask=mask, other=0.0).to(tl.float32)
    rel = tl.load(rel_ptr + rel_offsets, mask=mask, other=0.0).to(tl.float32)

    added_bf16 = (content + rel).to(tl.bfloat16)
    unscaled = added_bf16.to(tl.float32)
    scaled_bf16 = (added_bf16 * SCALE).to(tl.bfloat16)
    scaled = scaled_bf16.to(tl.float32)

    abs_scaled = tl.abs(scaled)
    finite = (scaled == scaled) & (abs_scaled != float("inf")) & mask
    invalid = (~finite) & mask
    has_invalid = tl.max(tl.where(invalid, 1, 0), axis=1) != 0

    unscaled_for_max = tl.where(mask, unscaled, -float("inf"))
    scaled_for_max = tl.where(mask, scaled, -float("inf"))
    unscaled_max = tl.max(unscaled_for_max, axis=1)
    scaled_max = tl.max(scaled_for_max, axis=1)

    shifted_unscaled = (unscaled - unscaled_max[:, None]) * SCALE
    shifted_scaled = scaled - scaled_max[:, None]
    shifted = tl.where(has_invalid[:, None], shifted_scaled, shifted_unscaled)
    shifted = tl.where(mask, shifted, -float("inf"))

    numer = libdevice.exp(shifted)
    numer = tl.where(mask, numer, 0.0)
    denom = tl.sum(numer, axis=1)
    probs = numer / denom[:, None]

    tl.store(out_ptr + content_offsets, probs.to(tl.bfloat16), mask=mask)


# (T([256,512,512], bf16), T([256,512,1024], bf16), S([16,16,512,1,512]), ...)
@oracle_impl(hardware="B200", point="99deda8b", BLOCK_M=4, BLOCK_N=512, num_warps=8)
def oracle_forward(inputs, *, BLOCK_M: int, BLOCK_N: int, num_warps: int):
    content, rel, _shape0, _shape1, _shape2, _shape3, _shape4, _shape5, out_shape = inputs
    out_shape = tuple(int(dim) for dim in out_shape)
    out = torch.empty_strided(
        out_shape,
        (out_shape[1] * out_shape[2], out_shape[2], 1),
        device=content.device,
        dtype=torch.bfloat16,
    )
    n_rows = out_shape[0] * out_shape[1]

    _xlnet_relative_shift_softmax_kernel[(triton.cdiv(n_rows, BLOCK_M),)](
        content,
        rel,
        out,
        N_ROWS=n_rows,
        K_LEN=out_shape[2],
        REL_WIDTH=1023,
        REL_STRIDE=512 * 1024,
        SCALE=0.125,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
