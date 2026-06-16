"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bf16 bias add, divide-by-8 scaling, head/batch layout rewrite, overlapping three-window as_strided stencil clone, and final contiguous `[288,512,64]` return into one Triton materialization kernel, whereas Inductor currently handles the cheap pointwise producer and the overlapping clone/layout materialization as separate generic scheduling work with extra memory traffic; Inductor cannot do this today because fixed-overlap as_strided clone materialization is treated as a scheduler fusion barrier even when the producer is affine pointwise arithmetic and the consumer is only a view; the fix is SCHEDULER_FUSION: teach layout clone codegen to sink affine pointwise producers through fixed-overlap as_strided window indexing and write the final contiguous backing storage directly while preserving bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _longformer_scaled_stencil_kernel(
    input_ptr,
    bias_ptr,
    output_ptr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    head_batch = chunk // 3
    window = chunk - head_batch * 3
    batch = head_batch // 12
    head = head_batch - batch * 12

    source_seq = window * 256 + pos[:, None]
    source_feature = head * 64 + dim[None, :]
    load_offsets = (source_seq * 8 + batch) * 768 + source_feature
    store_offsets = chunk * 512 * 64 + pos[:, None] * 64 + dim[None, :]
    mask = (pos[:, None] < 512) & (dim[None, :] < 64)

    values = tl.load(input_ptr + load_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + source_feature, mask=dim[None, :] < 64, other=0.0).to(tl.float32)
    added = (values + bias).to(tl.bfloat16).to(tl.float32)
    scaled = (added * 0.125).to(tl.bfloat16)
    tl.store(output_ptr + store_offsets, scaled, mask=mask)


@oracle_impl(hardware="B200", point="5fa3702b", BLOCK_P=32, BLOCK_D=64, num_warps=8)
def oracle_forward(inputs, *, BLOCK_P: int, BLOCK_D: int, num_warps: int):
    arg0, arg1, _shape0, _shape1, _shape2, _shape3, _shape4, _stride4, shape5 = inputs
    out = torch.empty_strided(
        tuple(int(dim) for dim in shape5),
        (512 * 64, 64, 1),
        device=arg0.device,
        dtype=arg0.dtype,
    )
    _longformer_scaled_stencil_kernel[
        (288, triton.cdiv(512, BLOCK_P), triton.cdiv(64, BLOCK_D))
    ](
        arg0,
        arg1,
        out,
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
    )
    return out
