"""Gap diagnosis (classification: BANDWIDTH_BOUND): this oracle computes the complete BERT bf16 metadata-view/add/select/view scope by materializing the shared `[16,128,768]` backing storage with one storage-linear Triton add kernel, then returning the eager-compatible `select(dim=1,index=0)` alias and flattened `[2048,768]` alias from that same allocation; Inductor already lowers the required work to the same mandatory two bf16 reads, one bf16-rounded store, and metadata-only output views, so the remaining delta is launch/allocation overhead and memory bandwidth rather than a missed fusion or layout transformation; the fix is BANDWIDTH_BOUND: classify this repro as a view-alias pointwise floor unless broader launch-overhead or allocation improvements move both paths together."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


TOTAL = 2048 * 768


@triton.jit
def _bert_add_backing_kernel(
    flat_ptr,
    view_ptr,
    out_ptr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    flat = tl.load(flat_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    view = tl.load(view_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offsets, (view + flat).to(tl.bfloat16), mask=mask)


@oracle_impl(hardware="B200", point="043f71e9", BLOCK=1024)
def oracle_forward(inputs, *, BLOCK: int):
    arg0_1, arg1_1, _shape_param_0, _shape_param_1 = inputs
    del _shape_param_0, _shape_param_1

    backing = torch.empty_strided(
        (16, 128, 768),
        (98304, 768, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )
    _bert_add_backing_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        arg0_1,
        arg1_1,
        backing,
        total=TOTAL,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return backing.select(1, 0), backing.view(2048, 768)
