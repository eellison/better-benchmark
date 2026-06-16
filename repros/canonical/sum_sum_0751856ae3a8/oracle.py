"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete GhostNet bf16 BatchNorm-backward fragment by sharing the sliced bf16 add producer, its f32 promotion, both `[0, 2, 3]` channel reductions, the raw-sum and scale-gradient vector outputs, and the returned bf16 dense epilogue, whereas Inductor schedules the sliced add, sibling reductions, vector epilogue, and dependent full-tensor epilogue as separate generic pointwise/reduction regions over replayed large producers; Inductor cannot do this today because its scheduler/codegen lacks a full-scope multi-output channel-reduction template that keeps compatible reductions, exact bf16/f32 cast boundaries, and finalized scalars live across the dependent bf16 tensor store; the fix is SCHEDULER_FUSION: add scheduler support for paired BN-backward channel reductions with shared producer finalization and fused vector/dense epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_IN = 16
C = 8
H = 112
W = 112
HW = H * W
R = N * HW
NUMEL = N * C * HW
ARG0_STRIDE_N = C_IN * HW
ARG0_STRIDE_H = C_IN * W
ARG1_STRIDE_N = C * HW
ARG1_STRIDE_H = C * W
REDUCE_SCALE = 1.5570192920918366e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    NUM_TILES: tl.constexpr,
    R_: tl.constexpr,
    C_: tl.constexpr,
    C_IN_: tl.constexpr,
    H_: tl.constexpr,
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
    arg0_offsets = (
        n[:, None] * ARG0_STRIDE_N_
        + h[:, None] * ARG0_STRIDE_H_
        + w[:, None] * C_IN_
        + cols[None, :]
    )
    dense_offsets = (
        n[:, None] * ARG1_STRIDE_N_
        + h[:, None] * ARG1_STRIDE_H_
        + w[:, None] * C_
        + cols[None, :]
    )

    source = _f32_add(
        tl.load(arg0_ptr + arg0_offsets, mask=active, other=0.0).to(tl.float32),
        tl.load(arg1_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32),
    ).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    centered = _f32_sub(
        tl.load(arg2_ptr + dense_offsets, mask=active, other=0.0).to(tl.float32),
        tl.load(mean_ptr + cols[None, :], mask=col_active[None, :], other=0.0).to(tl.float32),
    )
    dot = _f32_mul(source, centered)

    out_offsets = tile * C_ + cols
    tl.store(partial_sum_ptr + out_offsets, tl.sum(source, axis=0), mask=col_active)
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
    NUM_TILES: tl.constexpr,
    C_: tl.constexpr,
    REDUCE_SCALE_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = tiles * C_ + c

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    dot_scaled = _f32_mul(dot_value, REDUCE_SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(vec_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, REDUCE_SCALE_))
    tl.store(dot_coeff_ptr + c, _f32_mul(dot_scaled, invstd_sq))
    tl.store(out_scale_ptr + c, _f32_mul(invstd, weight))


@triton.jit
def _epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_ptr,
    mean_term_ptr,
    dot_coeff_ptr,
    out_scale_ptr,
    out_ptr,
    NUMEL_: tl.constexpr,
    C_: tl.constexpr,
    C_IN_: tl.constexpr,
    H_: tl.constexpr,
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
    h = (linear // (C_ * W_)) % H_
    n = linear // (C_ * HW_)
    arg0_offsets = n * ARG0_STRIDE_N_ + h * ARG0_STRIDE_H_ + w * C_IN_ + c

    source = _f32_add(
        tl.load(arg0_ptr + arg0_offsets, mask=active, other=0.0).to(tl.float32),
        tl.load(arg1_ptr + linear, mask=active, other=0.0).to(tl.float32),
    ).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    centered = _f32_sub(
        tl.load(arg2_ptr + linear, mask=active, other=0.0).to(tl.float32),
        tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32),
    )
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_coeff = tl.load(dot_coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=active, other=0.0).to(tl.float32)

    adjusted = _f32_sub(_f32_sub(source, _f32_mul(centered, dot_coeff)), mean_term)
    result = _f32_mul(adjusted, out_scale).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(out_ptr + linear, result, mask=active)


# ffa43910: GhostNet train bf16 sliced-add BN-backward, C=8, H=W=112.
@oracle_impl(
    hardware="B200",
    point="ffa43910",
    BLOCK_R=1024,
    BLOCK_C=8,
    reduce_warps=8,
    final_warps=8,
    epilogue_warps=4,
)
def oracle_forward(inputs, *, BLOCK_R, BLOCK_C, reduce_warps, final_warps, epilogue_warps):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = inputs
    num_tiles = triton.cdiv(R, BLOCK_R)
    block_tiles = _next_power_of_2(num_tiles)

    partial_sum = torch.empty((num_tiles, C), device=arg0_1.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    sum_1 = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    mul_8 = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    dot_coeff = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=arg0_1.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, C * W, C),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(num_tiles, triton.cdiv(C, BLOCK_C))](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        partial_sum,
        partial_dot,
        NUM_TILES=num_tiles,
        R_=R,
        C_=C,
        C_IN_=C_IN,
        H_=H,
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
        arg4_1,
        arg5_1,
        sum_1,
        mul_8,
        mean_term,
        dot_coeff,
        out_scale,
        NUM_TILES=num_tiles,
        C_=C,
        REDUCE_SCALE_=REDUCE_SCALE,
        BLOCK_TILES=block_tiles,
        num_warps=final_warps,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(NUMEL, EPILOGUE_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        mean_term,
        dot_coeff,
        out_scale,
        out,
        NUMEL_=NUMEL,
        C_=C,
        C_IN_=C_IN,
        H_=H,
        W_=W,
        HW_=HW,
        ARG0_STRIDE_N_=ARG0_STRIDE_N,
        ARG0_STRIDE_H_=ARG0_STRIDE_H,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_1, mul_8, out
