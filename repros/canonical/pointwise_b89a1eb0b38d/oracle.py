"""Gap diagnosis (classification: NEW_PATTERN): this oracle materializes the complete bf16 attention-head `permute.clone.view.view` scope directly into the returned contiguous `[B*S, H*D]` storage using the captured input strides, whereas Inductor lowers the same required clone through generic pointwise/layout-copy indexing; Inductor cannot do this today because its pointwise/layout scheduler has no guarded transformer head-major to token-major materialization template that specializes the affine input strides and final view store; the fix is NEW_PATTERN: add a shape-specialized attention-head layout-copy lowering that writes the final clone storage directly for this permute/view family."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _head_to_token_layout_kernel(
    in_ptr,
    out_ptr,
    TOTAL_VECTORS: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    STRIDE_B: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_S: tl.constexpr,
    STRIDE_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    vectors = tl.program_id(0) * BLOCK_V + tl.arange(0, BLOCK_V)
    dim = tl.arange(0, BLOCK_D)
    vector_mask = vectors < TOTAL_VECTORS

    h = vectors % H
    tmp = vectors // H
    s = tmp % S
    b = tmp // S

    in_offsets = (
        b[:, None] * STRIDE_B
        + h[:, None] * STRIDE_H
        + s[:, None] * STRIDE_S
        + dim[None, :] * STRIDE_D
    )
    out_offsets = vectors[:, None] * D + dim[None, :]
    mask = vector_mask[:, None] & (dim[None, :] < D)
    values = tl.load(in_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + out_offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="9fbac4b8", BLOCK_V=128, num_warps=8)
@oracle_impl(hardware="B200", point="903ae292", BLOCK_V=128, num_warps=8)
@oracle_impl(hardware="B200", point="976e9b58", BLOCK_V=128, num_warps=4)
def oracle_forward(inputs, *, BLOCK_V, num_warps):
    arg0_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0

    bsz = int(arg0_1.shape[0])
    heads = int(arg0_1.shape[1])
    seq = int(arg0_1.shape[2])
    head_dim = int(arg0_1.shape[3])
    out_shape = tuple(int(dim) for dim in _shape_param_1)

    out = torch.empty_strided(
        out_shape,
        (out_shape[1], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total_vectors = bsz * seq * heads
    grid = (triton.cdiv(total_vectors, BLOCK_V),)
    _head_to_token_layout_kernel[grid](
        arg0_1,
        out,
        TOTAL_VECTORS=total_vectors,
        S=seq,
        H=heads,
        D=head_dim,
        STRIDE_B=int(arg0_1.stride(0)),
        STRIDE_H=int(arg0_1.stride(1)),
        STRIDE_S=int(arg0_1.stride(2)),
        STRIDE_D=int(arg0_1.stride(3)),
        BLOCK_V=BLOCK_V,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=num_warps,
        num_stages=3,
    )
    return out
