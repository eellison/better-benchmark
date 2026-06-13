"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ConvBert local-window gather layout scope by directly materializing the final contiguous `[98304,64,9]` bf16 backing tensor from the `[16384,384]` projection and `[9,512,1,1]` padded-position index tensor, returning both the contiguous view and its `[98304,9,64]` permuted alias, whereas Inductor lowers the view/permute/clone/constant_pad_nd/index/permute/view/clone/view/expand/permute chain as generic layout and gather work; Inductor cannot do this today because its scheduler treats the constant padding, advanced indexing, and clone layout materialization as separate barriers instead of one affine-plus-indexed output map with sibling view returns; the fix is SCHEDULER_FUSION: teach layout/gather scheduling to sink constant padding and indexed loads into the final clone materialization while preserving the aliasing output views."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _convbert_window_gather_linear_kernel(
    projection_ptr,
    index_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL

    k = offsets % 9
    channel = (offsets // 9) % 384
    seq = (offsets // 3456) % 512
    batch = offsets // 1769472

    padded_seq = tl.load(index_ptr + k * 512 + seq, mask=mask, other=0)
    padded_seq = tl.where(padded_seq < 0, padded_seq + 520, padded_seq)
    source_seq = padded_seq - 4
    valid = mask & (source_seq >= 0) & (source_seq < 512)
    values = tl.load(
        projection_ptr + (batch * 512 + source_seq) * 384 + channel,
        mask=valid,
        other=0.0,
    )
    tl.store(out_ptr + offsets, values, mask=mask)


@oracle_impl(hardware="B200", point="3f2521b1", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    arg0_1, arg1_1, arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5 = inputs
    del arg2_1, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_5

    output_shape = tuple(int(dim) for dim in _shape_param_4)
    out = torch.empty_strided(
        output_shape,
        (output_shape[1] * output_shape[2], output_shape[2], 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    total = output_shape[0] * output_shape[1] * output_shape[2]
    _convbert_window_gather_linear_kernel[(triton.cdiv(total, BLOCK),)](
        arg0_1,
        arg1_1,
        out,
        TOTAL=total,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out, out.permute(0, 2, 1)
