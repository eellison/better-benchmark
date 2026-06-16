"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the full bf16 attention head-splitting layout scope by writing the required fresh contiguous `[B*H, S, D]` output directly from the contiguous `[B*S, H*D]` input with a B200-tuned Triton transpose-copy kernel, whereas Inductor lowers the view/permute/clone/view chain through generic layout materialization; Inductor cannot do this today because its scheduler does not recognize this common transformer head-splitting transpose as a specialized copy pattern with tuned launch geometry; the fix is NEW_PATTERN: add a guarded attention head-layout materialization template for `view(B,S,H,D).permute(0,2,1,3).contiguous().view(B*H,S,D)`."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_split_transpose_kernel(
    input_ptr,
    output_ptr,
    rows: tl.constexpr,
    seq: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row_offsets = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dim_offsets = tl.arange(0, BLOCK_D)
    mask = (row_offsets[:, None] < rows) & (dim_offsets[None, :] < head_dim)

    seq_idx = row_offsets % seq
    tmp = row_offsets // seq
    head = tmp % heads
    batch = tmp // heads

    input_offsets = ((batch[:, None] * seq + seq_idx[:, None]) * heads + head[:, None]) * head_dim + dim_offsets[None, :]
    output_offsets = row_offsets[:, None] * head_dim + dim_offsets[None, :]
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + output_offsets, values, mask=mask)


def _resolve_shape(shape, numel):
    dims = [int(dim) for dim in shape]
    if -1 in dims:
        known = 1
        for dim in dims:
            if dim != -1:
                known *= dim
        dims[dims.index(-1)] = numel // known
    return tuple(dims)


@oracle_impl(hardware="B200", point="631b8e39", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="bd432928", BLOCK_ROWS=8, BLOCK_D=128, num_warps=4)
@oracle_impl(hardware="B200", point="981155f5", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="1a8eaeba", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="d87997ca", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="d20f46e2", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b8160d07", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="af0c9f46", BLOCK_ROWS=4, BLOCK_D=128, num_warps=4)
@oracle_impl(hardware="B200", point="ad7b2a2c", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="cb7fdfdf", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b63e0b0f", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="b642f4d6", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="4fa33397", BLOCK_ROWS=8, BLOCK_D=64, num_warps=4)
@oracle_impl(hardware="B200", point="3ab46e72", BLOCK_ROWS=16, BLOCK_D=32, num_warps=4)
def oracle_forward(inputs, *, BLOCK_ROWS=None, BLOCK_D=None, num_warps=None):
    x, shape0, _shape1, shape2 = inputs
    output_shape = _resolve_shape(shape2, x.numel())
    batch = int(shape0[0])
    seq = int(shape0[1])
    head_dim = int(output_shape[2])
    heads = int(output_shape[0]) // batch
    rows = int(output_shape[0]) * seq
    if BLOCK_D is None:
        BLOCK_D = triton.next_power_of_2(head_dim)
    if BLOCK_ROWS is None:
        BLOCK_ROWS = 16 if head_dim <= 32 else (4 if head_dim >= 128 else 8)
    if num_warps is None:
        num_warps = 4

    output = torch.empty_strided(
        output_shape,
        (seq * head_dim, head_dim, 1),
        device=x.device,
        dtype=x.dtype,
    )
    grid = (triton.cdiv(rows, BLOCK_ROWS),)
    _head_split_transpose_kernel[grid](
        x,
        output,
        rows,
        seq,
        heads,
        head_dim,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return output
