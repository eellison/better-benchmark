"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete returned bf16 zero scalar, materialized bf16 `where(arg0 <= 0, 0, arg1)`, and returned f32 channel sum in one Triton pass over the two inputs, whereas Inductor schedules the visible `where` producer and its channel-reduction consumer as separate generic regions; Inductor cannot do this today because its scheduler does not fuse a returned full-layout pointwise tensor with a sibling fixed-axis reduction while preserving the observable bf16 side output and compiled f32 reduction envelope; the fix is SCHEDULER_FUSION: add a guarded multi-output producer/reduction fusion that emits the materialized pointwise result and reduced channel side output from one loop nest."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


@triton.jit
def _masked_where_channel_sum_kernel(
    pred_ptr,
    value_ptr,
    full_out_ptr,
    where_out_ptr,
    sum_out_ptr,
    n_size: tl.constexpr,
    c_size: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    rows = tl.arange(0, BLOCK_N)
    mask = (rows[:, None] < n_size) & (cols[None, :] < c_size)
    offsets = rows[:, None] * c_size + cols[None, :]

    pred = tl.load(pred_ptr + offsets, mask=mask, other=0.0)
    values = tl.load(value_ptr + offsets, mask=mask, other=0.0)
    selected = tl.where(pred <= 0.0, 0.0, values).to(tl.bfloat16)

    tl.store(where_out_ptr + offsets, selected, mask=mask)

    totals = tl.sum(tl.where(mask, selected.to(tl.float32), 0.0), axis=0)
    tl.store(sum_out_ptr + cols, totals, mask=cols < c_size)
    tl.store(full_out_ptr, 0.0, mask=tl.program_id(0) == 0)


def _oracle_forward_impl(inputs, *, BLOCK_N: int, BLOCK_C: int, num_warps: int):
    arg0_1, arg1_1 = inputs
    n_size = int(arg0_1.shape[0])
    c_size = int(arg0_1.shape[1])

    full = torch.empty_strided(
        (),
        (),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    where = torch.empty_strided(
        tuple(arg1_1.shape),
        tuple(arg1_1.stride()),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )
    converted_sum = torch.empty_strided(
        (c_size,),
        (1,),
        device=arg0_1.device,
        dtype=torch.float32,
    )

    _masked_where_channel_sum_kernel[(triton.cdiv(c_size, BLOCK_C),)](
        arg0_1,
        arg1_1,
        full,
        where,
        converted_sum,
        n_size=n_size,
        c_size=c_size,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    return full, where, converted_sum


# f0206885: (T([128,768,1,1], bf16), T([128,768,1,1], bf16))
@oracle_impl(hardware="B200", point="f0206885", BLOCK_N=128, BLOCK_C=8, num_warps=4)
# 32b702a1: (T([512,240,1,1], bf16), T([512,240,1,1], bf16))
@oracle_impl(hardware="B200", point="32b702a1", BLOCK_N=512, BLOCK_C=8, num_warps=8)
# 09bcbc35: (T([128,384,1,1], bf16), T([128,384,1,1], bf16))
@oracle_impl(hardware="B200", point="09bcbc35", BLOCK_N=128, BLOCK_C=8, num_warps=4)
# f4357794: (T([32,240,1,1], bf16), T([32,240,1,1], bf16))
@oracle_impl(hardware="B200", point="f4357794", BLOCK_N=32, BLOCK_C=16, num_warps=4)
def oracle_forward(inputs, *, BLOCK_N: int, BLOCK_C: int, num_warps: int):
    return _oracle_forward_impl(
        inputs,
        BLOCK_N=BLOCK_N,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
