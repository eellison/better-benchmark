"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete Swin attention-output layout scope, including the live `slice(..., 1, 0, -7)` from the captured `[B*H, 56, 32]` bf16 input and the required fresh contiguous `[B*49, H*32]` output, by directly materializing the sliced head/sequence transpose with a shape-specialized Triton copy, whereas Inductor lowers the captured slice/view/permute/clone/view chain through its generic layout-materialization scheduler; Inductor cannot do this today because it does not recognize this sliced Swin attention layout as a guarded copy template with the fixed 56-to-49 sequence contraction; the fix is NEW_PATTERN: add a Swin attention output layout materialization pattern that folds the slice and final contiguous clone into one tuned copy schedule."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _swin_slice_layout_kernel(
    x_ptr,
    out_ptr,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    IN_S: tl.constexpr,
    OUT_S: tl.constexpr,
    HIDDEN: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    dim = offsets % D
    head = (offsets // D) % H
    seq = (offsets // HIDDEN) % OUT_S
    batch = offsets // BATCH_STRIDE

    src_offsets = dim + D * seq + (IN_S * D) * head + (IN_S * D * H) * batch
    values = tl.load(x_ptr + src_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="fecae95d", BLOCK_SIZE=1024, num_warps=8, num_stages=3)
@oracle_impl(hardware="B200", point="31e78315", BLOCK_SIZE=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="c6f70802", BLOCK_SIZE=1024, num_warps=4, num_stages=3)
@oracle_impl(hardware="B200", point="28712c8b", BLOCK_SIZE=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps, num_stages):
    x, shape0, _shape1, shape2 = inputs
    h = int(shape0[1])
    out_s = int(shape0[2])
    d = int(shape0[3])
    in_s = int(x.shape[1])
    hidden = h * d
    out_shape = tuple(int(dim) for dim in shape2)
    out = torch.empty_strided(out_shape, (hidden, 1), device=x.device, dtype=x.dtype)

    n_elements = out_shape[0] * out_shape[1]
    _swin_slice_layout_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        x,
        out,
        N=n_elements,
        H=h,
        D=d,
        IN_S=in_s,
        OUT_S=out_s,
        HIDDEN=hidden,
        BATCH_STRIDE=out_s * hidden,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
