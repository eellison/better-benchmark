"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete transformer attention-head layout materialization by copying the captured contiguous bf16 [B*S,H*D] projection directly into the final contiguous [B*H,S,D] clone/view layout with a shape-specialized Triton layout-copy kernel, whereas Inductor lowers the view/permute/expand/clone/view chain through its generic layout materialization path; Inductor cannot do this today because its scheduler/codegen does not recognize this attention-head reshape-permute-clone family as a dedicated layout-copy template with the final view folded into the store indexing; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that emits the direct [B*S,H*D] to [B*H,S,D] copy while preserving the fresh contiguous output, bf16 dtype, and aliasing behavior."""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)[:, None, None]
    seq = tl.program_id(2) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)[None, :, None]
    dim = tl.arange(0, BLOCK_D)[None, None, :]

    mask = (head < H) & (seq < S) & (dim < D)
    input_offsets = ((batch * S + seq) * H + head) * D + dim
    output_offsets = ((batch * H + head) * S + seq) * D + dim
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@oracle_impl(
    hardware="B200",
    point="631b8e39",
    BLOCK_HEADS=4,
    BLOCK_SEQ=8,
    num_warps=8,
)
def oracle_forward(inputs, *, BLOCK_HEADS=4, BLOCK_SEQ=8, num_warps=8):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    b, h, s, d = (int(dim) for dim in _shape_param_2)
    output_shape = tuple(int(dim) for dim in _shape_param_3)
    output = torch.empty_strided(
        output_shape,
        (s * d, d, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    grid = (
        b,
        triton.cdiv(h, BLOCK_HEADS),
        triton.cdiv(s, BLOCK_SEQ),
    )
    _head_layout_kernel[grid](
        arg0_1,
        output,
        B=b,
        H=h,
        S=s,
        D=d,
        BLOCK_D=_next_power_of_2(d),
        BLOCK_HEADS=BLOCK_HEADS,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=num_warps,
    )
    return output


_SHAPES = json.loads((Path(__file__).with_name("shapes.json")).read_text())
for _POINT in _SHAPES["points"][1:]:
    oracle_impl(
        hardware="B200",
        point=_POINT["shape_hash"],
        BLOCK_HEADS=4,
        BLOCK_SEQ=8,
        num_warps=8,
    )(oracle_forward)
