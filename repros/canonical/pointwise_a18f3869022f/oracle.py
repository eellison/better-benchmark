"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the full bf16 scaled attention head-splitting layout scope by writing the required fresh contiguous `[B*H, S, D]` output directly from the contiguous `[B*S, H*D]` input with the captured fp32 scalar multiply and final bf16 store in one Triton transpose-copy kernel, whereas Inductor lowers the view/permute/mul/expand/clone/view chain through generic pointwise layout materialization; Inductor cannot do this today because its scheduler does not recognize this common scaled transformer head-layout materialization as a specialized copy pattern with tuned launch geometry; the fix is NEW_PATTERN: add a guarded scaled attention head-layout materialization template for `view(B,S,H,D).permute(0,2,1,3).mul(scale).contiguous().view(B*H,S,D)`."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SCALE = 0.3535533905932738


@triton.jit
def _scaled_head_split_flat_kernel(
    input_ptr,
    output_ptr,
    seq: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dim = offsets % head_dim
    seq_idx = (offsets // head_dim) % seq
    batch_head = offsets // (seq * head_dim)
    head = batch_head % heads
    batch = batch_head // heads

    input_offsets = ((batch * seq + seq_idx) * heads + head) * head_dim + dim
    values = tl.load(input_ptr + input_offsets).to(tl.float32)
    tl.store(output_ptr + offsets, values * scale)


def _resolve_shape(shape, numel):
    out = [int(dim) for dim in shape]
    infer = -1
    known = 1
    for index, dim in enumerate(out):
        if dim == -1:
            infer = index
        else:
            known *= dim
    if infer >= 0:
        out[infer] = numel // known
    return tuple(out)


@oracle_impl(hardware="B200", point="d87997ca", BLOCK_SIZE=4096, num_warps=4)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_SIZE=1024, num_warps=4)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    x, shape0, _shape1, _shape2, shape3 = inputs
    output_shape = _resolve_shape(shape3, x.numel())
    batch = int(shape0[0])
    seq = int(shape0[1])
    head_dim = int(output_shape[2])
    heads = int(output_shape[0]) // batch

    output = torch.empty_strided(
        output_shape,
        (seq * head_dim, head_dim, 1),
        device=x.device,
        dtype=x.dtype,
    )
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)
    _scaled_head_split_flat_kernel[grid](
        x,
        output,
        seq,
        heads,
        head_dim,
        SCALE,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return output
