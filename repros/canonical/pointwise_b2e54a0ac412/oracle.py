"""Gap diagnosis (classification: NEW_PATTERN): this oracle computes the complete GPT-J bf16 indexed RoPE gather/layout scope, including the deterministic iota/repeat side outputs, gathered bf16 sine/cosine coefficient aliases, both rounded bf16 rotary branches, and the returned strided aliasing layout views, whereas Inductor lowers the decomposed gather, split, expand/clone, rotate-half arithmetic, cat, dtype-convert, and permute/view graph through generic pointwise and layout scheduling; Inductor cannot do this today because its scheduler/codegen has no guarded GPT-J RoPE gather-layout template that preserves the visible side outputs and aliasing view envelope while fusing the repeated coefficient gather and two rotary consumers; the fix is NEW_PATTERN: add a GPT-J indexed RoPE lowering that materializes required side outputs and emits both branch layouts directly with exact bf16 rounding boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


SEQ = 128
HEADS = 16
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_PAIRS = 32


@triton.jit
def _round_to_bf16_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _side_outputs_kernel(
    table_ptr,
    add_ptr,
    ne_ptr,
    repeat_ptr,
    coeff_ptr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    meta_mask = offsets < 128
    seq_meta = offsets.to(tl.int64)
    tl.store(add_ptr + offsets, seq_meta, mask=meta_mask)
    tl.store(ne_ptr + offsets, offsets < 0, mask=meta_mask)

    coeff_mask = offsets < 8192
    seq = offsets // 64
    dim = offsets - seq * 64
    coeff = tl.load(table_ptr + seq * 64 + dim, mask=coeff_mask, other=0.0)
    tl.store(
        coeff_ptr + offsets,
        coeff.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=coeff_mask,
    )
    tl.store(repeat_ptr + offsets, seq.to(tl.int64), mask=coeff_mask)


@triton.jit
def _rope_layout_kernel(
    x0_ptr,
    x1_ptr,
    coeff_ptr,
    out0_ptr,
    out1_ptr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dims = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)

    row_mask = rows < 2048
    elem_mask = row_mask[:, None] & (dims[None, :] < 256)

    seq = rows // 16
    head = rows - seq * 16
    in_base = seq[:, None] * 4096 + head[:, None] * 256
    offsets = in_base + dims[None, :]

    rotary = dims < 64
    rotary_mask = elem_mask & rotary[None, :]
    pair = dims // 2
    pair_dim = tl.where((dims & 1) == 0, dims + 1, dims - 1)
    sign = tl.where((dims & 1) == 0, -1.0, 1.0)

    coeff_base = tl.load(
        coeff_ptr + seq[:, None] * 64 + 32 + pair[None, :],
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    coeff_rotate = tl.load(
        coeff_ptr + seq[:, None] * 64 + pair[None, :],
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    x0 = tl.load(x0_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
    x0_pair = tl.load(
        x0_ptr + in_base + pair_dim[None, :],
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    x0_mul = _round_to_bf16_f32(x0 * coeff_base)
    x0_rot = _round_to_bf16_f32((x0_pair * sign[None, :]) * coeff_rotate)
    x0_value = _round_to_bf16_f32(x0_mul + x0_rot)
    x0_value = tl.where(rotary[None, :], x0_value, x0)

    x1 = tl.load(x1_ptr + offsets, mask=elem_mask, other=0.0).to(tl.float32)
    x1_pair = tl.load(
        x1_ptr + in_base + pair_dim[None, :],
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    x1_mul = _round_to_bf16_f32(x1 * coeff_base)
    x1_rot = _round_to_bf16_f32((x1_pair * sign[None, :]) * coeff_rotate)
    x1_value = _round_to_bf16_f32(x1_mul + x1_rot)
    x1_value = tl.where(rotary[None, :], x1_value, x1)

    out_offsets = seq[:, None] * 4096 + head[:, None] * 256 + dims[None, :]
    tl.store(
        out0_ptr + out_offsets,
        x0_value.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=elem_mask,
    )
    tl.store(
        out1_ptr + out_offsets,
        x1_value.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=elem_mask,
    )


@oracle_impl(
    hardware="B200",
    point="589b9793",
    META_BLOCK=8192,
    BLOCK_ROWS=4,
    BLOCK_D=64,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    META_BLOCK: int,
    BLOCK_ROWS: int,
    BLOCK_D: int,
    num_warps: int,
):
    arg0_1, arg1_1, arg2_1, *_shape_params = inputs

    add = torch.empty_strided((SEQ,), (1,), device=arg0_1.device, dtype=torch.int64)
    ne = torch.empty_strided((1, SEQ), (SEQ, 1), device=arg0_1.device, dtype=torch.bool)
    repeat = torch.empty_strided(
        (1, SEQ, ROTARY_DIM),
        (SEQ * ROTARY_DIM, ROTARY_DIM, 1),
        device=arg0_1.device,
        dtype=torch.int64,
    )
    coeff = torch.empty_strided(
        (1, SEQ, ROTARY_DIM),
        (SEQ * ROTARY_DIM, ROTARY_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _side_outputs_kernel[(1,)](
        arg2_1,
        add,
        ne,
        repeat,
        coeff,
        BLOCK=META_BLOCK,
        num_warps=num_warps,
        num_stages=3,
    )

    out0_base = torch.empty_strided(
        (1, HEADS, SEQ, HEAD_DIM),
        (HEADS * SEQ * HEAD_DIM, HEAD_DIM, HEADS * HEAD_DIM, 1),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )
    out1_base = torch.empty_strided(
        (1, HEADS, HEAD_DIM, SEQ),
        (HEADS * SEQ * HEAD_DIM, HEAD_DIM, 1, HEADS * HEAD_DIM),
        device=arg1_1.device,
        dtype=torch.bfloat16,
    )

    grid = (triton.cdiv(SEQ * HEADS, BLOCK_ROWS), triton.cdiv(HEAD_DIM, BLOCK_D))
    _rope_layout_kernel[grid](
        arg0_1,
        arg1_1,
        coeff,
        out0_base,
        out1_base,
        BLOCK_ROWS=BLOCK_ROWS,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )

    unsqueeze = add.unsqueeze(0)
    coeff_lo = coeff[:, :, :ROTARY_PAIRS].unsqueeze(2).unsqueeze(4)
    coeff_hi = coeff[:, :, ROTARY_PAIRS:].unsqueeze(2).unsqueeze(4)
    view_8 = out0_base.view(HEADS, SEQ, HEAD_DIM)
    view_9 = out1_base.view(HEADS, HEAD_DIM, SEQ)
    return (
        unsqueeze,
        ne,
        repeat,
        coeff_lo,
        coeff_hi,
        view_8,
        view_9,
        view_8.permute(0, 2, 1),
        view_9.permute(0, 2, 1),
    )
