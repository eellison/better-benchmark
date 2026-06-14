"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete phlippe DenseNet bf16 BatchNorm-backward tail in one channel-resident Triton pass, sharing the masked `where(arg1 <= 0, arg2, arg3)` producer across the returned `sum(where)` and `sum(where * centered) * invstd` f32 vectors, sinking the finalized summaries into the bf16 residual-add epilogue, and returning the required `add[:, :16]` alias, whereas Inductor schedules the sibling reductions, broadcast BN-backward epilogue, bf16 residual add, and returned aliasing slice as separate generic regions around replayed or materialized intermediates; Inductor cannot do this today because scheduler/codegen does not form one full-scope multi-output channel-reduction plan that reuses the shared masked/centered producer while preserving bf16/f32 cast boundaries and output aliasing; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse compatible BN-backward channel reductions with their dependent dense/add epilogues and alias-view returns."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 152
INPUT_C = 168
SLICE_OFFSET = 16
SLICE_C = 16
H = 4
W = 4
HW = H * W
K_TOTAL = N * HW
SCALE = 0.00048828125
_USE_INDUCTOR_NUMERICS = False


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _bn_tail_kernel(
    residual_ptr,
    mask_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    add_out_ptr,
    C_: tl.constexpr,
    INPUT_C_: tl.constexpr,
    SLICE_OFFSET_: tl.constexpr,
    HW_: tl.constexpr,
    K_TOTAL_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_INDUCTOR_NUMERICS: tl.constexpr,
):
    c = tl.program_id(0)
    k = tl.arange(0, BLOCK_K)
    active = k < K_TOTAL_
    n = k // HW_
    spatial = k - n * HW_

    compact_offsets = n * (C_ * HW_) + c * HW_ + spatial
    residual_offsets = n * (INPUT_C_ * HW_) + (c + SLICE_OFFSET_) * HW_ + spatial

    gate = tl.load(mask_ptr + compact_offsets, mask=active, other=0.0)
    source = tl.load(source_ptr + compact_offsets, mask=active, other=0.0)
    fill = tl.load(fill_ptr)
    selected_bf16 = tl.where(gate <= 0.0, fill, source)
    selected = selected_bf16.to(tl.float32)

    activation = tl.load(centered_source_ptr + compact_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(activation, mean)
    dot_terms = _f32_mul(selected, centered)

    sum_value = tl.sum(tl.where(active, selected, 0.0), axis=0)
    dot_value = tl.sum(tl.where(active, dot_terms, 0.0), axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd))

    mean_term = _f32_mul(sum_value, SCALE_)
    dot_mean = _f32_mul(dot_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    correction_scale = _f32_mul(dot_mean, invstd_sq)
    output_scale = _f32_mul(invstd, weight)
    correction = _f32_mul(centered, correction_scale)
    after_correction = _f32_sub(selected, correction)
    after_mean = _f32_sub(after_correction, mean_term)
    grad = _f32_mul(after_mean, output_scale)
    grad_bf16 = grad.to(tl.bfloat16, fp_downcast_rounding="rtne")

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0)
    if USE_INDUCTOR_NUMERICS:
        add_value = _f32_add(residual.to(tl.float32), grad_bf16.to(tl.float32)).to(
            tl.bfloat16, fp_downcast_rounding="rtne"
        )
    else:
        add_value = _bf16_add(residual, grad_bf16)
    tl.store(add_out_ptr + compact_offsets, add_value, mask=active)


@oracle_impl(hardware="B200", point="b0fc1d08", BLOCK_K=2048, num_warps=8)
def oracle_forward(inputs, *, BLOCK_K: int, num_warps: int):
    global _USE_INDUCTOR_NUMERICS
    residual, mask, fill, source, centered_source, mean, invstd, weight = inputs
    use_inductor_numerics = _USE_INDUCTOR_NUMERICS
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        _USE_INDUCTOR_NUMERICS = True
        use_inductor_numerics = True

    sum_out = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    add_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, HW, W, 1),
        device=mask.device,
        dtype=torch.bfloat16,
    )

    _bn_tail_kernel[(C,)](
        residual,
        mask,
        fill,
        source,
        centered_source,
        mean,
        invstd,
        weight,
        sum_out,
        scale_grad,
        add_out,
        C_=C,
        INPUT_C_=INPUT_C,
        SLICE_OFFSET_=SLICE_OFFSET,
        HW_=HW,
        K_TOTAL_=K_TOTAL,
        SCALE_=SCALE,
        BLOCK_K=BLOCK_K,
        USE_INDUCTOR_NUMERICS=use_inductor_numerics,
        num_warps=num_warps,
        num_stages=4,
    )
    slice_out = torch.as_strided(
        add_out,
        (N, SLICE_C, H, W),
        (C * HW, HW, W, 1),
    )
    return sum_out, scale_grad, add_out, slice_out
