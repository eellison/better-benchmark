"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete Longformer permute-clone-view scope by materializing the three required fresh contiguous bf16 `[8192,768]` outputs in one Triton layout-copy kernel that shares each `arg0_1[batch,seq,dim]` load across all three stores, whereas Inductor lowers the repeated clone/view branches as generic layout materializations; Inductor cannot do this today because pointwise/layout scheduling does not group identical sibling layout-copy outputs while preserving three distinct returned storages; the fix is SCHEDULER_FUSION: add sibling layout-materialization fusion so repeated clone consumers of the same permute can share address decoding and input loads while emitting every visible output."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _triple_layout_copy_kernel(
    x_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    rows = offsets // 768
    cols = offsets - rows * 768
    batch = rows % 8
    seq = rows // 8
    src = batch * 786432 + seq * 768 + cols
    vals = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(out0_ptr + offsets, vals, mask=mask)
    tl.store(out1_ptr + offsets, vals, mask=mask)
    tl.store(out2_ptr + offsets, vals, mask=mask)


# 2973ba0c: (T([8,1024,768], bf16), S([8192,768]), S([8192,768]), S([8192,768]))
@oracle_impl(hardware="B200", point="2973ba0c", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    x, shape0, _shape1, _shape2 = inputs
    out_shape = tuple(int(dim) for dim in shape0)
    out_stride = (out_shape[1], 1)

    out0 = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)
    out1 = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)
    out2 = torch.empty_strided(out_shape, out_stride, device=x.device, dtype=x.dtype)

    total = out_shape[0] * out_shape[1]
    _triple_layout_copy_kernel[(triton.cdiv(total, BLOCK),)](
        x,
        out0,
        out1,
        out2,
        total=total,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return out0, out1, out2
