"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Longformer bf16 view/permute/clone/view layout chain by directly materializing the required fresh contiguous `[8192, 768]` output from the original contiguous `[384, 256, 64]` input with one shape-specialized Triton layout-copy kernel, whereas Inductor lowers the captured two-clone layout materialization through generic view/permute/clone scheduling; Inductor cannot do this today because its scheduler/codegen does not recognize this Longformer chunk/head layout chain as a direct affine copy template with the final view folded into the store indexing; the fix is NEW_PATTERN: add a guarded Longformer layout materialization lowering that emits the direct input-to-output copy while preserving the fresh contiguous output, bf16 dtype, and non-aliasing clone semantics."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _longformer_layout_kernel(
    input_ptr,
    output_ptr,
    BLOCK_ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < 8192

    batch = rows // 1024
    token = rows - batch * 1024
    chunk = token // 256
    seq = token - chunk * 256

    cols0 = tl.arange(0, 512)
    head0 = cols0 // 64
    dim0 = cols0 - head0 * 64
    input_offsets0 = ((batch[:, None] * 48 + head0[None, :] * 4 + chunk[:, None]) * 256 + seq[:, None]) * 64 + dim0[None, :]
    output_offsets0 = rows[:, None] * 768 + cols0[None, :]
    values0 = tl.load(input_ptr + input_offsets0, mask=row_mask[:, None], other=0.0)
    tl.store(output_ptr + output_offsets0, values0, mask=row_mask[:, None])

    cols1 = 512 + tl.arange(0, 256)
    head1 = cols1 // 64
    dim1 = cols1 - head1 * 64
    input_offsets1 = ((batch[:, None] * 48 + head1[None, :] * 4 + chunk[:, None]) * 256 + seq[:, None]) * 64 + dim1[None, :]
    output_offsets1 = rows[:, None] * 768 + cols1[None, :]
    values1 = tl.load(input_ptr + input_offsets1, mask=row_mask[:, None], other=0.0)
    tl.store(output_ptr + output_offsets1, values1, mask=row_mask[:, None])


@oracle_impl(hardware="B200", point="ced82531", BLOCK_ROWS=32, num_warps=8)
def oracle_forward(inputs, *, BLOCK_ROWS, num_warps):
    arg0_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3

    output = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_4),
        (768, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _longformer_layout_kernel[(triton.cdiv(8192, BLOCK_ROWS),)](
        arg0_1,
        output,
        BLOCK_ROWS=BLOCK_ROWS,
        num_warps=num_warps,
    )
    return output
