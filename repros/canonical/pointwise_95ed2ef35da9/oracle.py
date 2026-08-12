"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete bf16 OPT/Whisper scaled attention-head view/permute scope by materializing the scaled contiguous backing storage once and returning the exact non-contiguous `[B, H, S, D]` metadata view with stride `(S*H*D, D, H*D, 1)`, whereas Inductor lowers the decomposed view/mul/view/permute graph through a generic pointwise/layout schedule; Inductor cannot do this today because its scheduler does not recognize this scaled head-split permute as a guarded backing-storage materialization template that can fold the scalar multiply into the layout output while preserving the returned view strides; the fix is NEW_PATTERN: add a scaled attention-head view/permute lowering that writes the backing storage directly and returns the aliasing metadata view."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _scale_storage_kernel(
    input_ptr,
    output_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, values * 0.125, mask=mask)


@oracle_impl(hardware="B200", point="07bfd41e", BLOCK=1024, num_warps=4)
@oracle_impl(hardware="B200", point="e6fa82b2", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, shape0, shape1 = inputs
    batch = int(shape0[0])
    seq = int(shape0[1])
    dim = int(shape1[3])
    heads = arg0_1.shape[1] // dim
    hidden = heads * dim

    out = torch.empty_strided(
        (batch, heads, seq, dim),
        (seq * hidden, dim, hidden, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _scale_storage_kernel[(triton.cdiv(arg0_1.numel(), BLOCK),)](
        arg0_1,
        out,
        N=arg0_1.numel(),
        BLOCK=BLOCK,
        num_warps=num_warps,
    )
    return out
