"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete ConvBERT bf16 attention-head layout scope by materializing the final contiguous `[16384,768]` output directly from the two captured dense sources: the first 384 columns come from the already-storage-contiguous `permute(...).clone()` view of `arg1`, and the second 384 columns come from the `arg0` flattened view; Inductor lowers the view/permute/clone/cat/view chain through generic layout and concatenation materialization, so it cannot currently recognize that both inputs are compact row-major `[B,S,6,64]` halves feeding one final `[B*S,12*64]` buffer; the fix is NEW_PATTERN: add a guarded layout-concat materialization template that emits a single row-tiled copy into the final flattened output while preserving the required bf16 values and contiguous output scope.
"""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _convbert_layout_cat_kernel(
    arg0_ptr,
    arg1_ptr,
    out_ptr,
    ROWS: tl.constexpr,
    HALF: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    mask = (rows[:, None] < ROWS) & (cols[None, :] < HALF)
    src_offsets = rows[:, None] * HALF + cols[None, :]
    out_offsets = rows[:, None] * (HALF * 2) + cols[None, :]

    left = tl.load(arg1_ptr + src_offsets, mask=mask, other=0.0)
    right = tl.load(arg0_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, left, mask=mask)
    tl.store(out_ptr + out_offsets + HALF, right, mask=mask)


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


@oracle_impl(
    hardware="B200",
    point="add2068b",
    BLOCK_M=8,
    BLOCK_C=512,
    num_warps=4,
    num_stages=3,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_M: int,
    BLOCK_C: int,
    num_warps: int,
    num_stages: int,
):
    arg0_1, arg1_1, _shape0, _shape1, _shape2, shape3 = inputs
    out_shape = _shape_tuple(shape3)
    output = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    rows = out_shape[0]
    half = out_shape[1] // 2
    _convbert_layout_cat_kernel[(triton.cdiv(rows, BLOCK_M),)](
        arg0_1,
        arg1_1,
        output,
        ROWS=rows,
        HALF=half,
        BLOCK_M=BLOCK_M,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
