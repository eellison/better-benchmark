"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 MobileNet-style BN-backward scope by split-K reducing the shared `N,H,W` domain for both sibling channel sums, finalizing the two f32 vector outputs, and using the same finalized summaries to write the returned channels-last bf16 gradient tensor; whereas Inductor schedules the bf16 add producer, sibling `sum([0,2,3])` reductions, and dependent full-tensor epilogue as generic reduction/pointwise work; Inductor cannot do this today because its scheduler/codegen lacks a cooperative split-K multi-output channel-reduction template with a dependent materialized epilogue that preserves bf16 cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a guarded BN-backward lowering that splits compatible channel reductions across `N,H,W`, finalizes shared summaries once, and fuses the tensor/vector epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


INV_NHW_CAPTURED = 2.4912308673469386e-06


@triton.jit
def _f32_add(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_sub(a, b):
    return tl.inline_asm_elementwise(
        "sub.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _f32_mul(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
        [a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _bf16(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _partial_bn_bwd_kernel(
    grad0_ptr,
    grad1_ptr,
    activation_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_centered_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < TOTAL_K
    offset = k * C + c

    lhs = tl.load(grad0_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(grad1_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + offset, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)

    grad = _bf16(_f32_add(lhs, rhs))
    centered = _f32_sub(activation, mean)
    product = _f32_mul(grad, centered)
    active_grad = tl.where(mask, grad, 0.0)
    active_product = tl.where(mask, product, 0.0)

    partial_offset = c * tl.num_programs(1) + tile
    tl.store(partial_sum_ptr + partial_offset, tl.sum(active_grad, axis=0))
    tl.store(partial_centered_ptr + partial_offset, tl.sum(active_product, axis=0))


@triton.jit
def _finalize_bn_bwd_kernel(
    partial_sum_ptr,
    partial_centered_ptr,
    invstd_ptr,
    stats_ptr,
    out_sum_ptr,
    out_weight_ptr,
    C: tl.constexpr,
    N_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    INV_NHW: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < N_TILES
    partial_offset = c * N_TILES + tiles

    sum_parts = tl.load(partial_sum_ptr + partial_offset, mask=mask, other=0.0).to(tl.float32)
    centered_parts = tl.load(partial_centered_ptr + partial_offset, mask=mask, other=0.0).to(tl.float32)
    sum_grad = tl.sum(sum_parts, axis=0)
    sum_centered = tl.sum(centered_parts, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    invstd_sq = _f32_mul(invstd, invstd)
    mean_term = _f32_mul(sum_grad, INV_NHW)
    var_term = _f32_mul(_f32_mul(sum_centered, INV_NHW), invstd_sq)
    weight_grad = _f32_mul(sum_centered, invstd)

    tl.store(stats_ptr + c, mean_term)
    tl.store(stats_ptr + C + c, var_term)
    tl.store(out_sum_ptr + c, sum_grad)
    tl.store(out_weight_ptr + c, weight_grad)


@triton.jit
def _write_bn_bwd_kernel(
    grad0_ptr,
    grad1_ptr,
    activation_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    stats_ptr,
    out_ptr,
    C: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    mask = offsets < TOTAL
    c = offsets % C

    lhs = tl.load(grad0_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(grad1_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var_term = tl.load(stats_ptr + C + c, mask=mask, other=0.0).to(tl.float32)

    grad = _bf16(_f32_add(lhs, rhs))
    centered = _f32_sub(activation, mean)
    correction = _f32_mul(centered, var_term)
    residual = _f32_sub(_f32_sub(grad, correction), mean_term)
    scale = _f32_mul(invstd, weight)
    out = _f32_mul(residual, scale)
    tl.store(out_ptr + offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=mask)


@oracle_impl(hardware="B200", point="f3fd9151", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=8, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="65b876e3", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="1b9feebb", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="3726f4ca", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="9793b43e", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="3d50e493", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=8, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="399aa3e2", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=8, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="39a9326e", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="86ef280c", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="b6f518ab", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="a45e6340", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=8, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="da738408", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="6f1023fc", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="b5264010", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="315c2b3e", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="0dc5b6bd", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="864b3c6f", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="cf15f756", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="409c8bd3", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="727b7028", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="4d254913", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="3edd6c00", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="ee318906", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="1592ce3d", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="ebb56431", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="f63ebc76", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
@oracle_impl(hardware="B200", point="f35ade00", BLOCK_K=2048, BLOCK_E=256, num_warps_reduce=8, num_warps_final=4, num_warps_epilogue=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K,
    BLOCK_E,
    num_warps_reduce,
    num_warps_final,
    num_warps_epilogue,
):
    grad0, grad1, activation, mean, invstd, weight = inputs
    n, c, h, w = grad0.shape
    hw = h * w
    nhw = n * hw
    total = nhw * c
    n_tiles = triton.cdiv(nhw, BLOCK_K)
    block_tiles = 1 << (n_tiles - 1).bit_length()

    partial_sum = torch.empty_strided((c, n_tiles), (n_tiles, 1), device=grad0.device, dtype=torch.float32)
    partial_centered = torch.empty_strided((c, n_tiles), (n_tiles, 1), device=grad0.device, dtype=torch.float32)
    stats = torch.empty_strided((2, c), (c, 1), device=grad0.device, dtype=torch.float32)
    out_sum = torch.empty_strided((c,), (1,), device=grad0.device, dtype=torch.float32)
    out_weight = torch.empty_strided((c,), (1,), device=grad0.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(grad0.shape), tuple(grad0.stride()), device=grad0.device, dtype=torch.bfloat16)

    _partial_bn_bwd_kernel[(c, n_tiles)](
        grad0,
        grad1,
        activation,
        mean,
        partial_sum,
        partial_centered,
        C=c,
        HW=hw,
        TOTAL_K=nhw,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps_reduce,
        num_stages=4,
    )
    _finalize_bn_bwd_kernel[(c,)](
        partial_sum,
        partial_centered,
        invstd,
        stats,
        out_sum,
        out_weight,
        C=c,
        N_TILES=n_tiles,
        BLOCK_TILES=block_tiles,
        INV_NHW=INV_NHW_CAPTURED,
        num_warps=num_warps_final,
        num_stages=4,
    )
    _write_bn_bwd_kernel[(triton.cdiv(total, BLOCK_E),)](
        grad0,
        grad1,
        activation,
        mean,
        invstd,
        weight,
        stats,
        out,
        C=c,
        TOTAL=total,
        BLOCK_E=BLOCK_E,
        num_warps=num_warps_epilogue,
        num_stages=4,
    )
    return out_sum, out_weight, out
