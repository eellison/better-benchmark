"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 XLNet layout materialization scope by writing the required fresh contiguous `[1024, 1024]` output directly from the contiguous `[1024, 16, 64]` input with the captured `out[d * 16 + h, n] = input[n, h, d]` indexing, whereas Inductor lowers the unsqueeze/permute/permute/clone/view/squeeze chain through generic layout-copy code; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this XLNet relative-position head layout as a guarded direct materialization template with simplified affine indexing; the fix is NEW_PATTERN: add a shape-specialized layout-copy lowering for this permute-clone-view family."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _xlnet_layout_materialize_kernel(
    in_ptr,
    out_ptr,
    LOCAL_SIZE: tl.constexpr,
    COLS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_LOCAL: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    local = tl.program_id(0) * BLOCK_LOCAL + tl.arange(0, BLOCK_LOCAL)
    cols = tl.program_id(1) * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = (local[:, None] < LOCAL_SIZE) & (cols[None, :] < COLS)

    h = local // HEAD_DIM
    d = local - h * HEAD_DIM
    rows = d * HEADS + h
    in_offsets = cols[None, :] * (HEADS * HEAD_DIM) + local[:, None]
    out_offsets = rows[:, None] * COLS + cols[None, :]

    values = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, values, mask=mask)


# T([1024,16,64], bf16), S([1,1024,1024]) -> T([1024,1024], bf16)
@oracle_impl(hardware="B200", point="a1c20a52", BLOCK_LOCAL=32, BLOCK_COLS=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_LOCAL: int, BLOCK_COLS: int, num_warps: int):
    x, shape = inputs
    out_shape = (int(shape[1]), int(shape[2]))
    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=x.device,
        dtype=x.dtype,
    )
    cols = out_shape[1]
    heads = int(x.shape[1])
    head_dim = int(x.shape[2])
    _xlnet_layout_materialize_kernel[
        (triton.cdiv(heads * head_dim, BLOCK_LOCAL), triton.cdiv(cols, BLOCK_COLS))
    ](
        x,
        out,
        LOCAL_SIZE=heads * head_dim,
        COLS=cols,
        HEADS=heads,
        HEAD_DIM=head_dim,
        BLOCK_LOCAL=BLOCK_LOCAL,
        BLOCK_COLS=BLOCK_COLS,
        num_warps=num_warps,
        num_stages=3,
    )
    return out
