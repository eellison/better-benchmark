"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 MobileViT view/view/permute/slice_scatter scope by directly materializing the fresh contiguous `[512,4,256,40]` result, loading columns `0:36` from the compact `[131072,144]` source using the folded `[512,256,4,36] -> [512,4,256,36]` layout map and preserving columns `36:40` from the padded base tensor. Inductor lowers the metadata-only views/permute and functional slice update through generic pointwise/layout scheduling around the same data movement, with avoidable generic index decoding and an explicit scatter abstraction. Inductor cannot express this exact floor today because its scheduler does not sink static view/permute algebra through a fixed-width slice_scatter into one direct materialization. The fix is SCHEDULER_FUSION: add a guarded MobileViT padded-head slice_scatter lowering that emits the compact-source and base-tail stores from one output-space loop."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


BATCH = 512
HEADS = 4
SEQ = 256
ACTIVE_DIM = 36
PADDED_DIM = 40
HIDDEN = HEADS * ACTIVE_DIM
TOTAL = BATCH * HEADS * SEQ * PADDED_DIM


@triton.jit
def _mobilevit_slice_scatter_kernel(
    src_ptr,
    base_ptr,
    out_ptr,
    total: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    active_dim: tl.constexpr,
    padded_dim: tl.constexpr,
    hidden: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total

    dim = offsets % padded_dim
    tmp = offsets // padded_dim
    seq = tmp % seq_len
    tmp = tmp // seq_len
    head = tmp % heads
    batch = tmp // heads

    from_src = dim < active_dim
    src_offsets = (batch * seq_len + seq) * hidden + head * active_dim + dim
    src_values = tl.load(src_ptr + src_offsets, mask=active & from_src, other=0.0)
    base_values = tl.load(base_ptr + offsets, mask=active & (dim >= active_dim), other=0.0)
    values = tl.where(from_src, src_values, base_values)
    tl.store(out_ptr + offsets, values, mask=active)


# (T([131072,144], bf16), T([512,4,256,40], bf16), S([512,256,144]), S([512,256,4,36]))
@oracle_impl(hardware="B200", point="259ee10a", BLOCK=1024, num_warps=4, num_stages=3)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int, num_stages: int):
    src, base, _shape0, _shape1 = inputs
    out = torch.empty_strided(
        (BATCH, HEADS, SEQ, PADDED_DIM),
        (HEADS * SEQ * PADDED_DIM, SEQ * PADDED_DIM, PADDED_DIM, 1),
        device=base.device,
        dtype=base.dtype,
    )
    _mobilevit_slice_scatter_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        src,
        base,
        out,
        total=TOTAL,
        heads=HEADS,
        seq_len=SEQ,
        active_dim=ACTIVE_DIM,
        padded_dim=PADDED_DIM,
        hidden=HIDDEN,
        block=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
