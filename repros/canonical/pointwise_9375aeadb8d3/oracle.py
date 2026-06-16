"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete shape-param Repro.forward scope by materializing the f32 full(1) storage once, returning the two unsqueeze metadata views from that storage, and filling the visible bf16 cast tensor plus the `1.0 - cast` positive-zero tensor in the same Triton kernel, whereas Inductor currently lowers the full/unsqueeze/unsqueeze/cast/scalar-sub multi-output expression through generic scheduled pointwise materialization; Inductor cannot do this today because its scheduler/codegen simplifier does not propagate scalar constants through shape-parameter fulls, metadata-only views, dtype conversion, scalar-sub, and sibling returned outputs before lowering; the fix is ALGEBRAIC_ELIMINATION: add shape-aware constant propagation and output-alias planning for full/view/cast/scalar-sub chains so constant materializations and metadata views are emitted directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _fill_constant_mask_scope(
    full_ptr,
    cast_ptr,
    sub_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    one_f32 = tl.full((BLOCK,), 1.0, tl.float32)
    one_bf16 = one_f32.to(tl.bfloat16)
    zero_bf16 = tl.full((BLOCK,), 0.0, tl.float32).to(tl.bfloat16)
    tl.store(full_ptr + offsets, one_f32)
    tl.store(cast_ptr + offsets, one_bf16)
    tl.store(sub_ptr + offsets, zero_bf16)


# d7517139: (S([8,1024]))
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    shape = inputs[0]
    rows = int(shape[0])
    cols = int(shape[1])
    numel = rows * cols

    full = torch.empty_strided(
        (rows, cols),
        (cols, 1),
        device=torch.device("cuda", 0),
        dtype=torch.float32,
    )
    convert_element_type = torch.empty_strided(
        (rows, 1, 1, cols),
        (cols, cols, cols, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )
    sub = torch.empty_strided(
        (rows, 1, 1, cols),
        (cols, cols, cols, 1),
        device=torch.device("cuda", 0),
        dtype=torch.bfloat16,
    )

    _fill_constant_mask_scope[(triton.cdiv(numel, BLOCK),)](
        full,
        convert_element_type,
        sub,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )

    unsqueeze = full.unsqueeze(1)
    unsqueeze_1 = unsqueeze.unsqueeze(2)
    return full, unsqueeze, unsqueeze_1, convert_element_type, sub
