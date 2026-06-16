"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete XLNet large-vocabulary loss-backward fragment by materializing the returned bf16 `[8192, 32000]` gradient, returning its `[32000, 8192]` transpose as an alias view, and split-K reducing the same bf16 output into the returned bf16-rounded f32 column sum. Inductor currently lowers the sparse label one-hot producer, bf16-rounded shifted-logit exp, dense bf16 add, transpose metadata, and column reduction as separate generic regions around a very large materialized tensor; it cannot do this today because scheduler/codegen does not form one full-scope plan that shares the dense producer with the reduction while preserving the sparse label path, libdevice exp, bf16 cast boundaries, and output alias. The fix is COOPERATIVE_SPLIT_K: add a guarded large-vocabulary loss-backward template that writes the observable bf16 tensor while accumulating per-column partials for the sibling reduction and returns metadata aliases from the same backing storage."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


ROWS = 8192
VOCAB = 32000


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16_add(a, b):
    return (a.to(tl.float32) + b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _materialize_partials_kernel(
    numerator_ptr,
    denominator_ptr,
    labels_ptr,
    logits_ptr,
    shift0_ptr,
    shift1_ptr,
    residual_ptr,
    out_ptr,
    partial_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    row_block = tl.program_id(0)
    col_block = tl.program_id(1)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (rows[:, None] < 8192) & (cols[None, :] < 32000)
    offsets = rows[:, None] * 32000 + cols[None, :]

    scale = _f32_div(
        tl.load(numerator_ptr).to(tl.float32),
        tl.load(denominator_ptr).to(tl.float32),
    )
    label = tl.load(labels_ptr + rows, mask=rows < 8192, other=-100)
    valid = label != -100
    target = tl.where(valid, label, 0)
    row_grad = tl.where(
        valid,
        _f32_sub(0.0, scale).to(tl.bfloat16, fp_downcast_rounding="rtne").to(
            tl.float32
        ),
        0.0,
    )

    logits = tl.load(logits_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    row_shift0 = tl.load(shift0_ptr + rows, mask=rows < 8192, other=0.0).to(
        tl.float32
    )
    row_shift1 = tl.load(shift1_ptr + rows, mask=rows < 8192, other=0.0).to(
        tl.float32
    )
    shifted = _f32_sub(_f32_sub(logits, row_shift0[:, None]), row_shift1[:, None])
    shifted = shifted.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    exp_value = libdevice.exp(shifted)

    sparse = tl.where(
        valid[:, None] & (cols[None, :] == target[:, None]),
        row_grad[:, None],
        0.0,
    )
    dense_delta = _f32_sub(sparse, _f32_mul(exp_value, row_grad[:, None]))
    delta_bf16 = dense_delta.to(tl.bfloat16, fp_downcast_rounding="rtne")
    residual = tl.load(residual_ptr + offsets, mask=active, other=0.0)
    out_bf16 = _bf16_add(residual, delta_bf16)

    tl.store(out_ptr + offsets, out_bf16, mask=active)
    partial = tl.sum(tl.where(active, out_bf16.to(tl.float32), 0.0), axis=0)
    partial_offsets = row_block * 32000 + cols
    tl.store(partial_ptr + partial_offsets, partial, mask=cols < 32000)


@triton.jit
def _finalize_sum_kernel(
    partial_ptr,
    sum_out_ptr,
    NUM_ROW_BLOCKS: tl.constexpr,
    BLOCK_RB: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_blocks = tl.arange(0, BLOCK_RB)
    cols = col_block * BLOCK_C + tl.arange(0, BLOCK_C)
    active = (row_blocks[:, None] < NUM_ROW_BLOCKS) & (cols[None, :] < 32000)
    offsets = row_blocks[:, None] * 32000 + cols[None, :]
    values = tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    reduced = tl.sum(values, axis=0)
    rounded = reduced.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    tl.store(sum_out_ptr + cols, rounded, mask=cols < 32000)


# hf XLNetLMHeadModel train, labels [8192], dense vocab [8192, 32000].
@oracle_impl(
    hardware="B200",
    point="011490da",
    BLOCK_R=64,
    BLOCK_C=32,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_R: int, BLOCK_C: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        *_shape_params,
    ) = inputs

    num_row_blocks = triton.cdiv(ROWS, BLOCK_R)
    device = arg3_1.device
    out = torch.empty_strided(
        (ROWS, VOCAB), (VOCAB, 1), device=device, dtype=torch.bfloat16
    )
    partial = torch.empty_strided(
        (num_row_blocks, VOCAB), (VOCAB, 1), device=device, dtype=torch.float32
    )
    sum_out = torch.empty_strided((VOCAB,), (1,), device=device, dtype=torch.float32)

    _materialize_partials_kernel[(num_row_blocks, triton.cdiv(VOCAB, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        out,
        partial,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_sum_kernel[(triton.cdiv(VOCAB, BLOCK_C),)](
        partial,
        sum_out,
        NUM_ROW_BLOCKS=num_row_blocks,
        BLOCK_RB=triton.next_power_of_2(num_row_blocks),
        BLOCK_C=BLOCK_C,
        num_warps=8,
        num_stages=3,
    )

    return out, torch.as_strided(out, (VOCAB, ROWS), (1, VOCAB)), sum_out
