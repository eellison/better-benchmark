"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16-to-fp32 transformer attention-head layout materialization by writing the fresh contiguous `[B*H, S, D]` clone/view output directly from the contiguous `[B*S, H*D]` projection and applying the captured bf16-to-float32 cast at the store boundary, whereas Inductor lowers the view/view/permute/cast/expand/clone/view chain through generic pointwise layout materialization; Inductor cannot do this today because its pointwise/layout scheduler does not recognize this attention-head reshape-permute-cast-clone family as a dedicated layout-copy template with shape-specialized affine indexing and the dtype conversion folded into the materialization; the fix is NEW_PATTERN: add a guarded attention-head layout materialization lowering that emits the direct `[B*S, H*D]` bf16 load to `[B*H, S, D]` fp32 store while preserving the fresh contiguous output and exact cast boundary."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_layout_cast_kernel(
    input_ptr,
    output_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_head = tl.program_id(0)
    seq_offsets = tl.program_id(1) * BLOCK_S + tl.arange(0, BLOCK_S)[:, None]
    dim_offsets = tl.arange(0, BLOCK_D)[None, :]

    batch = batch_head // H
    head = batch_head - batch * H
    mask = (seq_offsets < S) & (dim_offsets < D)

    input_offsets = ((batch * S + seq_offsets) * H + head) * D + dim_offsets
    output_offsets = (batch_head * S + seq_offsets) * D + dim_offsets
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(output_ptr + output_offsets, values, mask=mask)


@oracle_impl(
    hardware="B200",
    point="af0c9f46",
    BLOCK_S=16,
    BLOCK_D=128,
    num_warps=8,
    num_stages=3,
)
def oracle_forward(inputs, *, BLOCK_S, BLOCK_D, num_warps, num_stages):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3 = inputs
    del _shape_param_0, _shape_param_1

    batch = int(_shape_param_2[0])
    heads = int(_shape_param_2[1])
    seq = int(_shape_param_2[2])
    head_dim = int(_shape_param_2[3])
    output = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_3),
        (seq * head_dim, head_dim, 1),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    grid = (batch * heads, triton.cdiv(seq, BLOCK_S))
    _head_layout_cast_kernel[grid](
        arg0_1,
        output,
        S=seq,
        H=heads,
        D=head_dim,
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
