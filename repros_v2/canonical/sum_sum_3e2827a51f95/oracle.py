"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 BN-backward-style return tuple by preserving the sliced channels-last bf16 add boundary, cooperatively reducing the shared `[N,H,W]` domain into both returned f32 channel summaries, and using the finalized summaries in a dense channels-last bf16 epilogue; whereas Inductor schedules the strided slice/add producer, sibling channel reductions, and dependent tensor epilogue as generic reduction and pointwise regions; Inductor cannot do this today because scheduler/codegen has no cooperative split-K multi-output lowering that coordinates bf16 producer rounding, two compatible per-channel reductions, and a dependent full-tensor side output with channels-last strides; the fix is COOPERATIVE_SPLIT_K: split the N/H/W reduction into deterministic channel partials, finalize the summaries once, and feed the exact epilogue store directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C0 = 80
C = 40
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
TOTAL = N * C * HW
SCALE = 9.964923469387754e-06
N_TILES = 98


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
def _round_bf16_to_f32(x):
    return tl.inline_asm_elementwise(
        "{ .reg .b16 t; cvt.rn.bf16.f32 t, $1; cvt.f32.bf16 $0, t; }",
        constraints="=f,f",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _partial_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    mean_ptr,
    partial_ptr,
    K_TOTAL_: tl.constexpr,
    C0_: tl.constexpr,
    C_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    N_TILES_: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    tile = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    valid = (k[:, None] < K_TOTAL_) & (c[None, :] < C_)

    n = k // HW_
    hw = k - n * HW_
    h = hw // W_
    w = hw - h * W_

    x0_offsets = n[:, None] * (C0_ * HW_) + h[:, None] * (W_ * C0_) + w[:, None] * C0_ + c[None, :]
    x1_offsets = n[:, None] * (C_ * HW_) + h[:, None] * (W_ * C_) + w[:, None] * C_ + c[None, :]

    x0 = tl.load(x0_ptr + x0_offsets, mask=valid, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=valid, other=0.0).to(tl.float32)
    add_value = _round_bf16_to_f32(_f32_add(x0, x1))

    x2 = tl.load(x2_ptr + x1_offsets, mask=valid, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < C_, other=0.0).to(tl.float32)
    centered = _f32_sub(x2, mean[None, :])
    prod = _f32_mul(add_value, centered)

    values = tl.where(valid, add_value, 0.0)
    products = tl.where(valid, prod, 0.0)
    partial_offsets = tile * C_ + c
    tl.store(partial_ptr + partial_offsets, tl.sum(values, axis=0), mask=c < C_)
    tl.store(partial_ptr + N_TILES_ * C_ + partial_offsets, tl.sum(products, axis=0), mask=c < C_)


@triton.jit
def _finalize_kernel(
    partial_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    C_: tl.constexpr,
    N_TILES_: tl.constexpr,
    SCALE_: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < N_TILES_
    partial_offsets = tiles * C_ + c

    sum_values = tl.load(partial_ptr + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    prod_values = tl.load(partial_ptr + N_TILES_ * C_ + partial_offsets, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_values, axis=0)
    prod_value = tl.sum(prod_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    scaled_prod = _f32_mul(prod_value, SCALE_)
    invstd_sq = _f32_mul(invstd, invstd)
    prod_coeff = _f32_mul(scaled_prod, invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scale_grad_ptr + c, _f32_mul(prod_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, SCALE_))
    tl.store(prod_coeff_ptr + c, prod_coeff)
    tl.store(output_scale_ptr + c, output_scale)


@triton.jit
def _epilogue_kernel(
    x0_ptr,
    x1_ptr,
    x2_ptr,
    mean_ptr,
    mean_term_ptr,
    prod_coeff_ptr,
    output_scale_ptr,
    out_ptr,
    TOTAL_: tl.constexpr,
    C0_: tl.constexpr,
    C_: tl.constexpr,
    H_: tl.constexpr,
    W_: tl.constexpr,
    HW_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL_
    c = offsets % C_
    w = (offsets // C_) % W_
    h = (offsets // (C_ * W_)) % H_
    n = offsets // (C_ * HW_)

    x0_offsets = n * (C0_ * HW_) + h * (W_ * C0_) + w * C0_ + c
    x1_offsets = n * (C_ * HW_) + h * (W_ * C_) + w * C_ + c
    x0 = tl.load(x0_ptr + x0_offsets, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    add_value = _round_bf16_to_f32(_f32_add(x0, x1))

    x2 = tl.load(x2_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(x2, mean)
    prod_coeff = tl.load(prod_coeff_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=mask, other=0.0).to(tl.float32)
    output_scale = tl.load(output_scale_ptr + c, mask=mask, other=0.0).to(tl.float32)

    out = _f32_mul(
        _f32_sub(_f32_sub(add_value, _f32_mul(centered, prod_coeff)), mean_term),
        output_scale,
    )
    out_bf16 = _round_bf16_to_f32(out)
    tl.store(out_ptr + offsets, out_bf16, mask=mask)


@oracle_impl(
    hardware="B200",
    point="453e4b44",
    BLOCK_K=1024,
    BLOCK_C=8,
    BLOCK_TILES=128,
    EPILOGUE_BLOCK=512,
    reduce_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    BLOCK_C: int,
    BLOCK_TILES: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    x0, x1, x2, mean, invstd, weight = inputs
    partial = torch.empty((2, N_TILES, C), device=x0.device, dtype=torch.float32)
    sum_out = torch.empty((C,), device=x0.device, dtype=torch.float32)
    scale_grad = torch.empty((C,), device=x0.device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=x0.device, dtype=torch.float32)
    prod_coeff = torch.empty((C,), device=x0.device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=x0.device, dtype=torch.float32)
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=x0.device,
        dtype=torch.bfloat16,
    )

    _partial_kernel[(N_TILES, triton.cdiv(C, BLOCK_C))](
        x0,
        x1,
        x2,
        mean,
        partial,
        K_TOTAL_=K_TOTAL,
        C0_=C0,
        C_=C,
        W_=W,
        HW_=HW,
        N_TILES_=N_TILES,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _finalize_kernel[(C,)](
        partial,
        invstd,
        weight,
        sum_out,
        scale_grad,
        mean_term,
        prod_coeff,
        output_scale,
        C_=C,
        N_TILES_=N_TILES,
        SCALE_=SCALE,
        BLOCK_TILES=BLOCK_TILES,
        num_warps=4,
        num_stages=3,
    )
    _epilogue_kernel[(triton.cdiv(TOTAL, EPILOGUE_BLOCK),)](
        x0,
        x1,
        x2,
        mean,
        mean_term,
        prod_coeff,
        output_scale,
        out,
        TOTAL_=TOTAL,
        C0_=C0,
        C_=C,
        H_=H,
        W_=W,
        HW_=HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=4,
    )
    return sum_out, scale_grad, out
