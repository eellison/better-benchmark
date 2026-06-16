"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete DeBERTaV2 constant attention-mask fanout by materializing the visible f32 full/view aliases, the f32 broadcast product, and all twenty-four fresh bool outputs directly as constants, whereas Inductor lowers the graph through a float full, metadata views, broadcast multiply, and repeated bool conversions; Inductor cannot do this today because its simplifier does not prove that full(1) multiplied by its broadcasted view is nonzero everywhere while preserving the returned full/view aliases and duplicated bool materializations; the fix is ALGEBRAIC_ELIMINATION: canonicalize nonzero constant full/broadcast arithmetic feeding bool conversions into direct constant materialization with sibling-output CSE before scheduling."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _constant_mask_fanout_kernel(
    full_ptr,
    mul_ptr,
    out0,
    out1,
    out2,
    out3,
    out4,
    out5,
    out6,
    out7,
    out8,
    out9,
    out10,
    out11,
    out12,
    out13,
    out14,
    out15,
    out16,
    out17,
    out18,
    out19,
    out20,
    out21,
    out22,
    out23,
    total: tl.constexpr,
    full_total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total
    true_values = offsets == offsets

    tl.store(mul_ptr + offsets, tl.full((BLOCK,), 1.0, tl.float32), mask=mask)
    tl.store(out0 + offsets, true_values, mask=mask)
    tl.store(out1 + offsets, true_values, mask=mask)
    tl.store(out2 + offsets, true_values, mask=mask)
    tl.store(out3 + offsets, true_values, mask=mask)
    tl.store(out4 + offsets, true_values, mask=mask)
    tl.store(out5 + offsets, true_values, mask=mask)
    tl.store(out6 + offsets, true_values, mask=mask)
    tl.store(out7 + offsets, true_values, mask=mask)
    tl.store(out8 + offsets, true_values, mask=mask)
    tl.store(out9 + offsets, true_values, mask=mask)
    tl.store(out10 + offsets, true_values, mask=mask)
    tl.store(out11 + offsets, true_values, mask=mask)
    tl.store(out12 + offsets, true_values, mask=mask)
    tl.store(out13 + offsets, true_values, mask=mask)
    tl.store(out14 + offsets, true_values, mask=mask)
    tl.store(out15 + offsets, true_values, mask=mask)
    tl.store(out16 + offsets, true_values, mask=mask)
    tl.store(out17 + offsets, true_values, mask=mask)
    tl.store(out18 + offsets, true_values, mask=mask)
    tl.store(out19 + offsets, true_values, mask=mask)
    tl.store(out20 + offsets, true_values, mask=mask)
    tl.store(out21 + offsets, true_values, mask=mask)
    tl.store(out22 + offsets, true_values, mask=mask)
    tl.store(out23 + offsets, true_values, mask=mask)

    full_mask = offsets < full_total
    tl.store(full_ptr + offsets, tl.full((BLOCK,), 1.0, tl.float32), mask=full_mask)


# d7517139: (S([8,512]))
@oracle_impl(hardware="B200", point="d7517139", BLOCK=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK, num_warps):
    (shape_param,) = inputs
    batch = int(shape_param[0])
    seq = int(shape_param[1])
    device = torch.device("cuda", 0)

    full = torch.empty_strided((batch, seq), (seq, 1), device=device, dtype=torch.float32)
    unsqueeze = torch.as_strided(full, (batch, seq, 1), (seq, 1, 1))
    unsqueeze_1 = torch.as_strided(full, (batch, 1, seq), (seq, seq, 1))
    unsqueeze_2 = torch.as_strided(full, (batch, 1, 1, seq), (seq, seq, seq, 1))
    squeeze = torch.as_strided(full, (batch, 1, seq), (seq, seq, 1))
    unsqueeze_3 = torch.as_strided(full, (batch, 1, seq, 1), (seq, seq, 1, 1))

    big_shape = (batch, 1, seq, seq)
    big_stride = (seq * seq, seq * seq, seq, 1)
    mul = torch.empty_strided(big_shape, big_stride, device=device, dtype=torch.float32)
    bool_outputs = tuple(
        torch.empty_strided(big_shape, big_stride, device=device, dtype=torch.bool)
        for _ in range(24)
    )

    total = batch * seq * seq
    _constant_mask_fanout_kernel[(triton.cdiv(total, BLOCK),)](
        full,
        mul,
        *bool_outputs,
        total=total,
        full_total=batch * seq,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )
    return (
        full,
        unsqueeze,
        unsqueeze_1,
        unsqueeze_2,
        squeeze,
        unsqueeze_3,
        mul,
        *bool_outputs,
    )
