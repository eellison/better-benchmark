"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete XLNet bf16 layernorm-backward/dropout return tuple by reading the `[16,512,1024]` view through the captured `[512,16,1024]` permute, preserving the first dropout clone, sharing each row's hidden-dimension reductions, storing the returned f32 gradient tensor plus the bf16 dropout-product view and its metadata-only transpose, and cooperatively finalizing the two f32 clone column reductions plus the bf16-rounded side-output sum from common row-tile partials, whereas Inductor schedules the view/permute/clone, row reductions, dense epilogues, bf16 materialization, transpose view, and sibling reductions as separate generic kernels over materialized intermediates; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output template that keeps row-local layernorm scalars live while emitting required dense side outputs and compatible hidden-column partial reductions across explicit bf16 cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a row-tiled transformer layernorm-backward lowering that fuses the permuted source read, dropout epilogues, view-equivalent side outputs, and sibling column-sum finalization."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 512
SEGMENTS = 16
ROWS = SEQ * SEGMENTS
HIDDEN = 1024
DROP_SCALE = 1.1111111111111112


@triton.jit
def _mul_rn(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _row_partials_kernel(
    source_ptr,
    keep0_ptr,
    weight_ptr,
    rhs_ptr,
    scale_ptr,
    keep1_ptr,
    grad_out_ptr,
    bf16_out_ptr,
    partials_ptr,
    ROWS_PER_GROUP: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HIDDEN_: tl.constexpr,
    SEQ_: tl.constexpr,
    SEGMENTS_: tl.constexpr,
    DROP_SCALE_: tl.constexpr,
):
    group = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < HIDDEN_
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    drop_scale = tl.full((BLOCK_R, BLOCK_C), DROP_SCALE_, tl.float32)

    acc_clone_rhs = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_clone = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bf16 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for local_start in tl.range(0, ROWS_PER_GROUP, BLOCK_R):
        rows = group * ROWS_PER_GROUP + local_start + tl.arange(0, BLOCK_R)
        row_mask = rows < (SEQ_ * SEGMENTS_)
        seq = rows // SEGMENTS_
        segment = rows - seq * SEGMENTS_
        source_rows = segment * SEQ_ + seq
        offsets = rows[:, None] * HIDDEN_ + cols[None, :]
        source_offsets = source_rows[:, None] * HIDDEN_ + cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :]

        source = tl.load(source_ptr + source_offsets, mask=mask, other=0.0).to(tl.float32)
        keep0 = tl.load(keep0_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep0_scaled = _mul_rn(keep0, drop_scale)
        clone = _mul_rn(source, keep0_scaled)

        rhs = tl.load(rhs_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + rows, mask=row_mask, other=0.0).to(tl.float32)
        weighted = _mul_rn(clone, weight[None, :])
        row_sum = tl.sum(tl.where(mask, weighted, 0.0), axis=1)
        row_dot = tl.sum(tl.where(mask, _mul_rn(weighted, rhs), 0.0), axis=1)
        centered = _sub_rn(_mul_rn(weighted, tl.full((BLOCK_R, BLOCK_C), HIDDEN_, tl.float32)), row_sum[:, None])
        centered = _sub_rn(centered, _mul_rn(rhs, row_dot[:, None]))
        grad = _mul_rn(scale[:, None], centered)

        keep1 = tl.load(keep1_ptr + offsets, mask=mask, other=0).to(tl.float32)
        keep_scaled = _mul_rn(keep1, drop_scale)
        keep_scaled_bf16 = keep_scaled.to(tl.bfloat16)
        grad_bf16 = grad.to(tl.bfloat16)
        dropped = _mul_rn(grad, keep_scaled).to(tl.bfloat16)
        dropped_for_sum = _mul_rn(
            grad_bf16.to(tl.float32),
            keep_scaled_bf16.to(tl.float32),
        ).to(tl.bfloat16)

        tl.store(grad_out_ptr + offsets, grad, mask=mask)
        tl.store(bf16_out_ptr + offsets, dropped, mask=mask)

        acc_clone_rhs += tl.sum(tl.where(mask, _mul_rn(clone, rhs), 0.0), axis=0)
        acc_clone += tl.sum(tl.where(mask, clone, 0.0), axis=0)
        acc_bf16 += tl.sum(tl.where(mask, dropped_for_sum.to(tl.float32), 0.0), axis=0)

    partial_base = group * 3 * HIDDEN_ + cols
    tl.store(partials_ptr + partial_base, acc_clone_rhs, mask=col_mask)
    tl.store(partials_ptr + partial_base + HIDDEN_, acc_clone, mask=col_mask)
    tl.store(partials_ptr + partial_base + 2 * HIDDEN_, acc_bf16, mask=col_mask)


@triton.jit
def _finalize_partials_kernel(
    partials_ptr,
    out_clone_rhs_ptr,
    out_clone_ptr,
    out_bf16_sum_ptr,
    NUM_GROUPS: tl.constexpr,
    HIDDEN_: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    groups = tl.arange(0, GROUP_BLOCK)
    col_mask = cols < HIDDEN_
    mask = (groups[:, None] < NUM_GROUPS) & col_mask[None, :]
    offsets = groups[:, None] * 3 * HIDDEN_ + cols[None, :]

    clone_rhs = tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    clone = tl.load(partials_ptr + offsets + HIDDEN_, mask=mask, other=0.0).to(tl.float32)
    side = tl.load(partials_ptr + offsets + 2 * HIDDEN_, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_clone_rhs_ptr + cols, tl.sum(clone_rhs, axis=0), mask=col_mask)
    tl.store(out_clone_ptr + cols, tl.sum(clone, axis=0), mask=col_mask)
    tl.store(
        out_bf16_sum_ptr + cols,
        tl.sum(side, axis=0).to(tl.bfloat16).to(tl.float32),
        mask=col_mask,
    )


@oracle_impl(
    hardware="B200",
    point="f3ef90ca",
    ROWS_PER_GROUP=16,
    BLOCK_R=1,
    BLOCK_C=1024,
    FINAL_BLOCK_C=8,
    num_warps=8,
    final_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROWS_PER_GROUP: int,
    BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    num_warps: int,
    final_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        *_shape_params,
    ) = inputs

    device = arg0_1.device
    num_groups = triton.cdiv(ROWS, ROWS_PER_GROUP)
    grad_out = torch.empty_strided(
        (SEQ, SEGMENTS, HIDDEN),
        (SEGMENTS * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )
    bf16_out = torch.empty_strided(
        (ROWS, HIDDEN),
        (HIDDEN, 1),
        device=device,
        dtype=torch.bfloat16,
    )
    out_clone_rhs = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_clone = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    out_bf16_sum = torch.empty((HIDDEN,), device=device, dtype=torch.float32)
    partials = torch.empty_strided(
        (num_groups, 3, HIDDEN),
        (3 * HIDDEN, HIDDEN, 1),
        device=device,
        dtype=torch.float32,
    )

    _row_partials_kernel[(num_groups,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        grad_out,
        bf16_out,
        partials,
        ROWS_PER_GROUP=ROWS_PER_GROUP,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        HIDDEN_=HIDDEN,
        SEQ_=SEQ,
        SEGMENTS_=SEGMENTS,
        DROP_SCALE_=DROP_SCALE,
        num_warps=num_warps,
    )

    group_block = 1 << (num_groups - 1).bit_length()
    _finalize_partials_kernel[(triton.cdiv(HIDDEN, FINAL_BLOCK_C),)](
        partials,
        out_clone_rhs,
        out_clone,
        out_bf16_sum,
        NUM_GROUPS=num_groups,
        HIDDEN_=HIDDEN,
        GROUP_BLOCK=group_block,
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=final_warps,
    )

    return grad_out, out_clone_rhs, out_clone, bf16_out, bf16_out.permute(1, 0), out_bf16_sum
