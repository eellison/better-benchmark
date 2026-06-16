"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle fuses the complete Longformer bf16 bias-add/scale producer, head/batch layout rewrite, overlapping three-window `as_strided` stencil clone, and returned aliasing `[288,512,64]` plus `[288,64,512]` views into one Triton materialization kernel; Inductor lowers the cheap pointwise producer and the fixed-overlap clone/layout materialization as generic separate scheduling work, so it rereads or materializes data around a pure affine stencil; the fix is SCHEDULER_FUSION: sink bf16 pointwise producers through affine overlapping layout-clone codegen and write the final backing storage directly while preserving the view alias returned by the permute."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _longformer_bf16_layout_stencil_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    chunk = tl.program_id(0)
    pos = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    dim = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)

    head_batch = chunk // 3
    window = chunk - head_batch * 3
    batch = head_batch // 12
    head = head_batch - batch * 12

    source_seq = window * 256 + pos[:, None]
    source_feature = head * 64 + dim[None, :]
    mask = (pos[:, None] < 512) & (dim[None, :] < 64)

    load_offsets = (source_seq * 8 + batch) * 768 + source_feature
    store_offsets = chunk * 512 * 64 + pos[:, None] * 64 + dim[None, :]

    x = tl.load(x_ptr + load_offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + source_feature, mask=dim[None, :] < 64, other=0.0).to(tl.float32)
    bias_bf16 = _round_to_bf16_f32(bias)
    added = _round_to_bf16_f32(x + bias_bf16)
    scaled = added * 0.125
    tl.store(out_ptr + store_offsets, scaled.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="53c69788", BLOCK_P=16, BLOCK_D=64, num_warps=4)
def oracle_forward(inputs, *, BLOCK_P, BLOCK_D, num_warps):
    bias, x, _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6 = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5

    base = torch.empty_strided(
        tuple(int(dim) for dim in _shape_param_6),
        (512 * 64, 64, 1),
        device=x.device,
        dtype=torch.bfloat16,
    )
    _longformer_bf16_layout_stencil_kernel[
        (288, triton.cdiv(512, BLOCK_P), triton.cdiv(64, BLOCK_D))
    ](
        x,
        bias,
        base,
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=3,
    )
    return base, base.permute(0, 2, 1)
