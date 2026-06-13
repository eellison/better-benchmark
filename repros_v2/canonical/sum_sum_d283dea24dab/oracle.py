"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 BatchNorm-backward fragment by evaluating the bf16-rounded BN/ReLU mask producer, selecting between the scalar fill and high-channel slice, co-reducing both `[0, 2, 3]` channel sums, and using the finalized summaries to write the returned channels-last bf16 dense gradient plus both returned f32 vectors. Inductor currently schedules the mask producer, sibling reductions, vector epilogues, and dependent dense epilogue as generic pointwise/reduction regions over materialized intermediates; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan that preserves the bf16 cast and where boundaries while sharing the masked producer across both reductions and the downstream BN-backward epilogue. The fix is SCHEDULER_FUSION: add a guarded channel-reduction lowering that shares compatible split-K partials and sinks their finalized summaries into the dependent tensor/vector stores."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 200
C = 100
SLICE_START = 100
H = 14
W = 14
HW = H * W
R = N * HW
NUMEL = N * C * HW
INV_R = 9.964923469387754e-06
ARG0_STRIDE_N = C_IN * HW
ARG0_STRIDE_H = C_IN * W
ARG1_STRIDE_N = C * HW
ARG1_STRIDE_H = C * W


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
def _masked_where_value(
    src_ptr,
    mask_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    src_offsets,
    dense_offsets,
    cols,
    active,
    C_: tl.constexpr,
):
    x = tl.load(mask_input_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=cols < C_, other=0.0).to(tl.float32)
    centered = _f32_sub(x, mean)
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd), weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    take_fill = affine_bf16 <= 0.0
    fill = tl.load(fill_ptr).to(tl.bfloat16)
    source = tl.load(src_ptr + src_offsets, mask=active, other=0.0)
    selected = tl.where(take_fill, fill, source)
    return selected.to(tl.float32), centered


@triton.jit
def _partial_reduce_kernel(
    src_ptr,
    mask_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    R_: tl.constexpr,
    C_: tl.constexpr,
    C_IN_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    ARG0_STRIDE_N_: tl.constexpr,
    ARG0_STRIDE_H_: tl.constexpr,
    ARG1_STRIDE_N_: tl.constexpr,
    ARG1_STRIDE_H_: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = tile * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_active = rows < R_
    col_active = cols < C_
    active = row_active[:, None] & col_active[None, :]

    n = rows // HW_
    spatial = rows - n * HW_
    h = spatial // W_
    w = spatial - h * W_
    dense_offsets = n[:, None] * ARG1_STRIDE_N_ + h[:, None] * ARG1_STRIDE_H_ + w[:, None] * C_ + cols[None, :]
    src_offsets = n[:, None] * ARG0_STRIDE_N_ + h[:, None] * ARG0_STRIDE_H_ + w[:, None] * C_IN_ + (cols[None, :] + SLICE_START_)

    where_value, centered = _masked_where_value(
        src_ptr,
        mask_input_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        src_offsets,
        dense_offsets,
        cols[None, :],
        active,
        C_,
    )
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)
    dot = _f32_mul(where_value, centered)

    out_offsets = tile * C_ + cols
    tl.store(partial_sum_ptr + out_offsets, tl.sum(where_value, axis=0), mask=col_active)
    tl.store(partial_dot_ptr + out_offsets, tl.sum(dot, axis=0), mask=col_active)


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    C_: tl.constexpr,
    INV_R_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    partial_offsets = tiles * C_ + c
    sum_values = tl.load(partial_sum_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
    dot_values = tl.load(partial_dot_ptr + partial_offsets, mask=active, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_values, axis=0)
    dot_value = tl.sum(dot_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    dot_scaled = _f32_mul(dot_value, INV_R_)
    invstd_sq = _f32_mul(invstd, invstd)
    tl.store(sum_out_ptr + c, sum_value)
    tl.store(vec_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, INV_R_))
    tl.store(dot_coeff_ptr + c, _f32_mul(dot_scaled, invstd_sq))
    tl.store(out_scale_ptr + c, _f32_mul(invstd, weight))


@triton.jit
def _epilogue_kernel(
    src_ptr,
    mask_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    C_IN_: tl.constexpr,
    SLICE_START_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    ARG0_STRIDE_N_: tl.constexpr,
    ARG0_STRIDE_H_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < NUMEL_
    c = linear % C_
    w = (linear // C_) % W_
    h = (linear // (C_ * W_)) % (HW_ // W_)
    n = linear // (C_ * HW_)

    dense_offsets = linear
    src_offsets = n * ARG0_STRIDE_N_ + h * ARG0_STRIDE_H_ + w * C_IN_ + (c + SLICE_START_)
    where_value, centered = _masked_where_value(
        src_ptr,
        mask_input_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        src_offsets,
        dense_offsets,
        c,
        active,
        C_,
    )
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_coeff = tl.load(dot_coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    adjusted = _f32_sub(_f32_sub(where_value, _f32_mul(centered, dot_coeff)), mean_term)
    out = _f32_mul(adjusted, out_scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + linear, out, mask=active)


def _ceil_pow2(value):
    return 1 << (int(value) - 1).bit_length()


def _launch(inputs, *, BLOCK_R: int, BLOCK_C: int, BLOCK: int, reduce_warps: int, final_warps: int, epilogue_warps: int):
    src, mask_input, mean, invstd, weight, bias, fill = inputs
    num_tiles = triton.cdiv(R, BLOCK_R)
    partial_sum = torch.empty((num_tiles, C), device=src.device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=src.device, dtype=torch.float32)
    sum_out = torch.empty_strided((C,), (1,), device=src.device, dtype=torch.float32)
    vec_out = torch.empty_strided((C,), (1,), device=src.device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=src.device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=src.device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=src.device, dtype=torch.float32)
    dense_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=src.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(num_tiles, triton.cdiv(C, BLOCK_C))](
        src,
        mask_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        partial_sum,
        partial_dot,
        R_=R,
        C_=C,
        C_IN_=C_IN,
        SLICE_START_=SLICE_START,
        W_=W,
        HW_=HW,
        ARG0_STRIDE_N_=ARG0_STRIDE_N,
        ARG0_STRIDE_H_=ARG0_STRIDE_H,
        ARG1_STRIDE_N_=ARG1_STRIDE_N,
        ARG1_STRIDE_H_=ARG1_STRIDE_H,
        BLOCK_R=BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        sum_out,
        vec_out,
        mean_term,
        dot_coeff,
        out_scale,
        C_=C,
        INV_R_=INV_R,
        NUM_TILES=num_tiles,
        BLOCK_TILES=_ceil_pow2(num_tiles),
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, BLOCK),)](
        src,
        mask_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        mean_term,
        dot_coeff,
        out_scale,
        dense_out,
        NUMEL_=NUMEL,
        C_=C,
        C_IN_=C_IN,
        SLICE_START_=SLICE_START,
        W_=W,
        HW_=HW,
        ARG0_STRIDE_N_=ARG0_STRIDE_N,
        ARG0_STRIDE_H_=ARG0_STRIDE_H,
        BLOCK=BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_out, vec_out, dense_out


# d56aace4: GhostNet train bf16 masked BN-backward reductions, C=100, H=W=14.
@oracle_impl(
    hardware="B200",
    point="d56aace4",
    BLOCK_R=512,
    BLOCK_C=16,
    BLOCK=1024,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(inputs, **kwargs):
    return _launch(inputs, **kwargs)
