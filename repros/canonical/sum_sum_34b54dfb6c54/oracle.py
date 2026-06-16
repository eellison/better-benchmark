"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ALBERT bf16 layernorm-backward row-sum fragment with one row-resident Triton program per `[4096]` hidden row, sharing the `arg0.to(f32) * arg1` producer across both row reductions, the returned f32 converted activation, the returned f32 epilogue tensor, and the returned bf16 view plus transpose alias, whereas Inductor currently schedules the conversion, sibling row reductions, dependent dense epilogue, bf16 cast, and aliasing transpose through generic fused regions around materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form a full-scope row-resident multi-output plan that keeps both reductions and all visible dtype/view boundaries in one specialized producer; the fix is SCHEDULER_FUSION: add a fixed-hidden layernorm-backward row-reduction template that sinks the f32 side outputs and bf16 alias-producing store from the same row summaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


ROWS = 4096
HIDDEN = 4096


@triton.jit
def _mul_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _sub_rn_f32(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _albert_row_kernel(
    arg0_ptr,
    scale_ptr,
    normed_ptr,
    row_scale_ptr,
    convert_out_ptr,
    f32_out_ptr,
    bf16_out_ptr,
    HIDDEN_: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)
    mask = cols < HIDDEN_
    offsets = row * HIDDEN_ + cols

    converted = tl.load(arg0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    normed = tl.load(normed_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    weighted = _mul_rn_f32(converted, scale)

    sum_weighted = tl.sum(tl.where(mask, weighted, 0.0), axis=0)
    weighted_normed = _mul_rn_f32(weighted, normed)
    sum_weighted_normed = tl.sum(tl.where(mask, weighted_normed, 0.0), axis=0)

    row_scale = tl.load(row_scale_ptr + row).to(tl.float32)
    hidden_value = tl.full((BLOCK_H,), HIDDEN_, tl.float32)
    scaled_weighted = _mul_rn_f32(weighted, hidden_value)
    centered = _sub_rn_f32(scaled_weighted, sum_weighted)
    correction = _mul_rn_f32(normed, sum_weighted_normed)
    out = _mul_rn_f32(row_scale, _sub_rn_f32(centered, correction))

    tl.store(convert_out_ptr + offsets, converted, mask=mask)
    tl.store(f32_out_ptr + offsets, out, mask=mask)
    tl.store(
        bf16_out_ptr + offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )


@oracle_impl(hardware="B200", point="77702286", BLOCK_H=4096, num_warps=8)
def oracle_forward(inputs, *, BLOCK_H: int, num_warps: int):
    arg0_1, arg1_1, arg2_1, arg3_1, _shape_param_0, _shape_param_1 = inputs

    convert_out = torch.empty(
        (8, 512, HIDDEN), device=arg0_1.device, dtype=torch.float32
    )
    f32_out = torch.empty(
        (8, 512, HIDDEN), device=arg0_1.device, dtype=torch.float32
    )
    bf16_out = torch.empty(
        (ROWS, HIDDEN), device=arg0_1.device, dtype=torch.bfloat16
    )

    _albert_row_kernel[(ROWS,)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        convert_out,
        f32_out,
        bf16_out,
        HIDDEN_=HIDDEN,
        BLOCK_H=BLOCK_H,
        num_warps=num_warps,
        num_stages=3,
    )
    return convert_out, f32_out, bf16_out, bf16_out.permute(1, 0)
