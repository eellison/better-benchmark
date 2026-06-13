"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete ConvNeXtV2 GRN backward plus structured average-pool-backward return scope by keeping the `[128,640] -> [128,1,1,640]` permute virtual, computing the per-row channel summaries once, emitting the exact bf16 zero-fill/as_strided_scatter/as_strided/expand/div result as a dense `[128,640,7,7]` tensor, and deriving the three returned `[640]` reductions from the compact row producer, whereas Inductor lowers the row reductions, dependent input-gradient epilogue, scatter-style broadcast construction, expanded bf16 div, and sibling channel reductions as generic scheduled fragments; Inductor cannot do this today because its scheduler/codegen does not model this dense as_strided_scatter/expand average-pool-backward idiom as a structured scatter-reduce producer that can feed the sibling reductions while preserving bf16 cast boundaries; the fix is SCATTER_REDUCE: add a guarded ConvNeXtV2 GRN/average-pool-backward lowering that shares row summaries, collapses the scatter broadcast to direct indexed stores, and finalizes compatible channel reductions from the compact producer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 640
HW = 49
INV_C = 1.0 / 640.0
INV_HW = 1.0 / 49.0


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
def _row_base_kernel(
    grad_flat_ptr,
    weight_ptr,
    x_nhwc_ptr,
    mean_ptr,
    invstd_ptr,
    base_bf16_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    C_: tl.constexpr,
    INV_C_: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C_
    offsets = n * C_ + cols

    grad = tl.load(
        grad_flat_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    weight = tl.load(
        weight_ptr + cols,
        mask=mask,
        other=0.0,
        eviction_policy="evict_last",
    ).to(tl.float32)
    x = tl.load(
        x_nhwc_ptr + offsets,
        mask=mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    mean = tl.load(mean_ptr + n).to(tl.float32)
    invstd = tl.load(invstd_ptr + n).to(tl.float32)

    weighted = _f32_mul(grad, weight)
    centered = _f32_sub(x, mean)
    normalized = _f32_mul(centered, invstd)
    weighted_for_sum = tl.where(mask, weighted, 0.0)
    dot_for_sum = tl.where(mask, _f32_mul(weighted, normalized), 0.0)
    weighted_sum = tl.sum(weighted_for_sum, axis=0)
    dot_sum = tl.sum(dot_for_sum, axis=0)

    term = weighted * C_ - weighted_sum
    term = term - normalized * dot_sum
    base = (invstd * INV_C_) * term
    tl.store(
        base_bf16_ptr + offsets,
        base.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=mask,
    )
    zero_mask = mask & (n == 0)
    tl.store(out0_ptr + cols, 0.0, mask=zero_mask)
    tl.store(out1_ptr + cols, 0.0, mask=zero_mask)
    tl.store(out3_ptr + cols, 0.0, mask=zero_mask)


@triton.jit
def _materialize_expand_div_kernel(
    base_bf16_ptr,
    grad_flat_ptr,
    x_nhwc_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    out0_ptr,
    out1_ptr,
    out3_ptr,
    C_: tl.constexpr,
    HW_: tl.constexpr,
    INV_HW_: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    hw = tl.arange(0, BLOCK_HW)
    c_mask = cols < C_
    hw_mask = hw < HW_
    mask = c_mask[:, None] & hw_mask[None, :]
    base_idx = n * C_ + cols
    idx = base_idx[:, None] * HW_ + hw[None, :]
    base = tl.load(
        base_bf16_ptr + base_idx,
        mask=c_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    out = (base * INV_HW_).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(
        out_ptr + idx,
        out[:, None],
        mask=mask,
    )

    grad = tl.load(
        grad_flat_ptr + base_idx,
        mask=c_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = tl.load(
        x_nhwc_ptr + base_idx,
        mask=c_mask,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    mean = tl.load(mean_ptr + n).to(tl.float32)
    invstd = tl.load(invstd_ptr + n).to(tl.float32)
    normalized = _f32_mul(_f32_sub(x, mean), invstd)
    tl.atomic_add(out0_ptr + cols, _f32_mul(grad, normalized), sem="relaxed", mask=c_mask)
    tl.atomic_add(out1_ptr + cols, grad, sem="relaxed", mask=c_mask)
    tl.atomic_add(out3_ptr + cols, out.to(tl.float32) * HW_, sem="relaxed", mask=c_mask)


def _contiguous_stride(shape):
    stride = []
    running = 1
    for dim in reversed(shape):
        stride.append(running)
        running *= int(dim)
    return tuple(reversed(stride))


def _as_shape(shape):
    return tuple(int(dim) for dim in shape)


# fd237320: (T([128,640], bf16), T([640], f32), T([128,1,1,640], bf16), T([128,1,1,1], f32), ...)
@oracle_impl(
    hardware="B200",
    point="fd237320",
    ROW_BLOCK_C=1024,
    OUT_BLOCK=64,
    REDUCE_BLOCK_C=8,
    num_warps=8,
)
def oracle_forward(
    inputs,
    *,
    ROW_BLOCK_C: int,
    OUT_BLOCK: int,
    REDUCE_BLOCK_C: int,
    num_warps: int,
):
    (
        grad_flat,
        weight,
        x_nhwc,
        mean,
        invstd,
        _shape0,
        _shape1,
        _shape2,
        _shape3,
        _shape4,
        _shape5,
        out_shape_param,
    ) = inputs
    device = grad_flat.device
    out_shape = _as_shape(out_shape_param)

    base_bf16 = torch.empty_strided((N, C), (C, 1), device=device, dtype=torch.bfloat16)
    out0 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out1 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    out2 = torch.empty_strided(
        out_shape,
        _contiguous_stride(out_shape),
        device=device,
        dtype=torch.bfloat16,
    )
    out3 = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)

    _row_base_kernel[(N,)](
        grad_flat,
        weight,
        x_nhwc,
        mean,
        invstd,
        base_bf16,
        out0,
        out1,
        out3,
        C_=C,
        INV_C_=INV_C,
        BLOCK_C=ROW_BLOCK_C,
        num_warps=num_warps,
    )
    _materialize_expand_div_kernel[(N, triton.cdiv(C, REDUCE_BLOCK_C))](
        base_bf16,
        grad_flat,
        x_nhwc,
        mean,
        invstd,
        out2,
        out0,
        out1,
        out3,
        C_=C,
        HW_=HW,
        INV_HW_=INV_HW,
        BLOCK_C=REDUCE_BLOCK_C,
        BLOCK_HW=OUT_BLOCK,
        num_warps=4,
    )

    return out0, out1, out2, out3
