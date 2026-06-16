"""Gap diagnosis (classification: ALGEBRAIC_ELIMINATION): this oracle computes the complete bf16 MT5 attention-mask producer scope, including returned iota/add tensors, unsqueeze view aliases, the always-true `ge` predicate and expanded bool alias, copied input scalar, min-finite bf16 scalar, and contiguous `[32,1,128,128]` where tensor, by proving `iota(128) >= 0` is tautologically true and filling the live outputs directly in one Triton kernel; Inductor lowers the decomposed iota/add/unsqueeze/ge/expand/scalar/where graph as generic pointwise and metadata work and does not eliminate the dead min-finite branch; the fix is ALGEBRAIC_ELIMINATION: add shape-aware simplification for nonnegative generated iota predicates before scheduling structured attention-mask materialization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _mask_outputs_kernel(
    input_scalar_ptr,
    iota_ptr,
    add_ptr,
    ge_ptr,
    lift_ptr,
    neg_ptr,
    where_ptr,
    total: tl.constexpr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_range = offsets < total
    x = tl.load(input_scalar_ptr)

    tl.store(where_ptr + offsets, x, mask=in_range)

    seq_mask = offsets < seq_len
    values = offsets.to(tl.int64)
    tl.store(iota_ptr + offsets, values, mask=seq_mask)
    tl.store(add_ptr + offsets, values, mask=seq_mask)
    tl.store(ge_ptr + offsets, seq_mask, mask=seq_mask)

    first = offsets == 0
    tl.store(lift_ptr + offsets, x, mask=first)
    tl.store(
        neg_ptr + offsets,
        tl.full((BLOCK_SIZE,), -3.3895313892515355e38, tl.float32),
        mask=first,
    )


# 2a837a19: (T([], bf16), S([32,-1,128,128]))
@oracle_impl(hardware="B200", point="2a837a19", BLOCK_SIZE=1024, num_warps=4)
def oracle_forward(inputs, *, BLOCK_SIZE, num_warps):
    arg0_1, shape_param = inputs
    device = arg0_1.device
    dtype = arg0_1.dtype
    batch = int(shape_param[0])
    seq_len = int(shape_param[2])

    iota = torch.empty_strided((seq_len,), (1,), device=device, dtype=torch.int64)
    add = torch.empty_strided((seq_len,), (1,), device=device, dtype=torch.int64)
    unsqueeze = torch.as_strided(add, (1, seq_len), (seq_len, 1))
    unsqueeze_1 = torch.as_strided(add, (1, 1, seq_len), (seq_len, seq_len, 1))
    unsqueeze_2 = torch.as_strided(add, (1, 1, seq_len, 1), (seq_len, seq_len, 1, 1))

    ge = torch.empty_strided(
        (1, 1, seq_len, 1),
        (seq_len, seq_len, 1, 1),
        device=device,
        dtype=torch.bool,
    )
    expand = torch.as_strided(ge, (batch, 1, seq_len, seq_len), (0, seq_len, 1, 0))

    lift_fresh_copy = torch.empty_strided((), (), device=device, dtype=dtype)
    scalar_tensor = torch.empty_strided((), (), device=device, dtype=dtype)
    where = torch.empty_strided(
        (batch, 1, seq_len, seq_len),
        (seq_len * seq_len, seq_len * seq_len, seq_len, 1),
        device=device,
        dtype=dtype,
    )

    total = where.numel()
    _mask_outputs_kernel[(triton.cdiv(total, BLOCK_SIZE),)](
        arg0_1,
        iota,
        add,
        ge,
        lift_fresh_copy,
        scalar_tensor,
        where,
        total=total,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return (
        iota,
        add,
        unsqueeze,
        unsqueeze_1,
        unsqueeze_2,
        ge,
        expand,
        lift_fresh_copy,
        scalar_tensor,
        where,
    )
