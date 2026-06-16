"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete `prims.iota(128)` plus `aten.embedding.default` scope in one Triton launch, returning the observable contiguous `i64[128]` iota and materializing the identity embedding gather from the f32 `[128, 2560]` table into a fresh contiguous f32 `[128, 2560]` output. Inductor treats the generated iota as an embedding index tensor and lowers the region through generic index construction plus gather/copy code; it cannot produce this exact floor today because its algebraic simplifier does not canonicalize identity iota embedding into the required fresh clone/copy while preserving the returned iota. The fix is ALGEBRAIC_ELIMINATION: add a guarded embedding-iota rewrite that removes identity index address arithmetic from the gather and emits the mandatory fresh contiguous copy plus iota output directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 128
HIDDEN = 2560
TOTAL = ROWS * HIDDEN


@triton.jit
def _embedding_iota_copy_kernel(
    table_ptr,
    iota_ptr,
    out_ptr,
    IN_S0: tl.constexpr,
    IN_S1: tl.constexpr,
    ROWS_: tl.constexpr,
    HIDDEN_: tl.constexpr,
    TOTAL_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < TOTAL_

    rows = offsets // HIDDEN_
    cols = offsets - rows * HIDDEN_
    values = tl.load(table_ptr + rows * IN_S0 + cols * IN_S1, mask=valid, other=0.0)
    tl.store(out_ptr + offsets, values, mask=valid)

    iota_mask = offsets < ROWS_
    tl.store(iota_ptr + offsets, offsets.to(tl.int64), mask=iota_mask)


# 800c9b3b: (T([128,2560], f32))
@oracle_impl(hardware="B200", point="800c9b3b", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK: int, num_warps: int):
    (arg0_1,) = inputs
    iota = torch.empty((ROWS,), device=arg0_1.device, dtype=torch.int64)
    embedding = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=arg0_1.device,
        dtype=arg0_1.dtype,
    )

    _embedding_iota_copy_kernel[(triton.cdiv(TOTAL, BLOCK),)](
        arg0_1,
        iota,
        embedding,
        IN_S0=arg0_1.stride(0),
        IN_S1=arg0_1.stride(1),
        ROWS_=ROWS,
        HIDDEN_=HIDDEN,
        TOTAL_=TOTAL,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return iota, embedding
