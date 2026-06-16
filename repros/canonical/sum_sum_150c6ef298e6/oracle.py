"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete bf16 DenseNet BN/ReLU-backward tail by split-K reducing the shared masked producer into `sum(where)` and `sum(where * centered)` per-channel f32 accumulators, finalizing the returned `[128]` vectors once, and using those finalized summaries in the dependent bf16 full-tensor epilogue while preserving the returned `[128,16,H,W]` slice as an alias of the returned `[128,128,H,W]` add storage, whereas Inductor currently schedules the paired reductions, broadcast scalar epilogue, bf16 cast/add, and returned slice as generic work around materialized intermediates; Inductor cannot do this today because its scheduler/codegen does not form one coordinated multi-output reduction plan with finalized channel scalars feeding a required full-layout side output and alias view while preserving bf16/f32 cast boundaries; the fix is COOPERATIVE_SPLIT_K: add a split-K multi-accumulator reduction template that sinks the dependent BN-backward epilogue and returned alias into the same full-scope plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 128
SRC_C = 144
SLICE_C = 16
SCALE = 3.0517578125e-05


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise(
        "add.rn.f32 $0, $1, $2;",
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
def _reduce_partials_kernel(
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    partial_where_ptr,
    partial_mul_ptr,
    c_size: tl.constexpr,
    hw: tl.constexpr,
    total_spatial: tl.constexpr,
    num_tiles: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < total_spatial

    n = k // hw
    spatial = k - n * hw
    offsets = n * (c_size * hw) + c * hw + spatial

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0).to(tl.bfloat16)
    source = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0)
    full_value = tl.load(full_ptr)
    where_bf16 = tl.where(mask_value <= 0.0, full_value, source)
    where_f32 = where_bf16.to(tl.float32)

    centered_src = tl.load(centered_src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _sub_rn(centered_src, mean)
    product = _mul_rn(where_f32, centered)

    partial_offsets = c * num_tiles + tile
    tl.store(partial_where_ptr + partial_offsets, tl.sum(tl.where(active, where_f32, 0.0), axis=0))
    tl.store(partial_mul_ptr + partial_offsets, tl.sum(tl.where(active, product, 0.0), axis=0))


@triton.jit
def _finalize_vectors_kernel(
    partial_where_ptr,
    partial_mul_ptr,
    invstd_ptr,
    sum_where_out_ptr,
    sum_mul_ptr,
    mul8_out_ptr,
    num_tiles: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < num_tiles
    offsets = c * num_tiles + tiles

    where_values = tl.load(partial_where_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mul_values = tl.load(partial_mul_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_where = tl.sum(where_values, axis=0)
    sum_mul = tl.sum(mul_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_where_out_ptr + c, sum_where)
    tl.store(sum_mul_ptr + c, sum_mul)
    tl.store(mul8_out_ptr + c, _mul_rn(sum_mul, invstd))


@triton.jit
def _bn_add_epilogue_kernel(
    residual_ptr,
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_ptr,
    sum_mul_ptr,
    out_ptr,
    c_size: tl.constexpr,
    src_c: tl.constexpr,
    slice_c: tl.constexpr,
    scale: tl.constexpr,
    hw: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total
    spatial = offsets % hw
    c = (offsets // hw) % c_size
    n = offsets // (c_size * hw)
    input_offsets = n * (c_size * hw) + c * hw + spatial
    residual_offsets = n * (src_c * hw) + (c + slice_c) * hw + spatial

    mask_value = tl.load(mask_ptr + input_offsets, mask=active, other=0.0).to(tl.bfloat16)
    source = tl.load(where_rhs_ptr + input_offsets, mask=active, other=0.0)
    full_value = tl.load(full_ptr)
    where_bf16 = tl.where(mask_value <= 0.0, full_value, source)
    where_f32 = where_bf16.to(tl.float32)

    centered_src = tl.load(centered_src_ptr + input_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = _sub_rn(centered_src, mean)

    sum_where = tl.load(sum_where_ptr + c, mask=active, other=0.0).to(tl.float32)
    sum_mul = tl.load(sum_mul_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)

    mean_term = _mul_rn(sum_where, scale)
    sum_mul_scaled = _mul_rn(sum_mul, scale)
    invstd_sq = _mul_rn(invstd, invstd)
    variance_term = _mul_rn(sum_mul_scaled, invstd_sq)
    centered_scaled = _mul_rn(centered, variance_term)
    sub1 = _sub_rn(where_f32, centered_scaled)
    sub2 = _sub_rn(sub1, mean_term)
    out_weight = _mul_rn(invstd, weight)
    grad = _mul_rn(sub2, out_weight).to(tl.bfloat16)

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0).to(tl.bfloat16)
    added = _add_rn(residual.to(tl.float32), grad.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + offsets, added, mask=active)


def _launch(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    final_warps: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    residual, mask, full, where_rhs, centered_src, mean, invstd, weight = inputs
    h = int(mask.shape[2])
    w = int(mask.shape[3])
    hw = h * w
    total_spatial = N * hw
    total = N * C * hw
    num_tiles = triton.cdiv(total_spatial, BLOCK_K)

    partial_where = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=mask.device, dtype=torch.float32)
    partial_mul = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=mask.device, dtype=torch.float32)
    sum_where = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    sum_mul = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    mul8 = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    out = torch.empty_strided((N, C, h, w), (C * hw, hw, w, 1), device=mask.device, dtype=torch.bfloat16)

    _reduce_partials_kernel[(C, num_tiles)](
        mask,
        full,
        where_rhs,
        centered_src,
        mean,
        partial_where,
        partial_mul,
        c_size=C,
        hw=hw,
        total_spatial=total_spatial,
        num_tiles=num_tiles,
        BLOCK_K=BLOCK_K,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_vectors_kernel[(C,)](
        partial_where,
        partial_mul,
        invstd,
        sum_where,
        sum_mul,
        mul8,
        num_tiles=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        num_warps=final_warps,
        num_stages=3,
    )
    _bn_add_epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        residual,
        mask,
        full,
        where_rhs,
        centered_src,
        mean,
        invstd,
        weight,
        sum_where,
        sum_mul,
        out,
        c_size=C,
        src_c=SRC_C,
        slice_c=SLICE_C,
        scale=SCALE,
        hw=hw,
        total=total,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return sum_where, mul8, out, torch.as_strided(out, (N, SLICE_C, h, w), (C * hw, hw, w, 1))


# a758b9fd: H=W=16
@oracle_impl(hardware="B200", point="a758b9fd", BLOCK_K=1024, EPILOGUE_BLOCK=256, reduce_warps=4, final_warps=4, epilogue_warps=4)
# c976107e: H=W=8
@oracle_impl(hardware="B200", point="c976107e", BLOCK_K=1024, EPILOGUE_BLOCK=256, reduce_warps=4, final_warps=4, epilogue_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
):
    return _launch(
        inputs,
        BLOCK_K=BLOCK_K,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
        reduce_warps=reduce_warps,
        final_warps=final_warps,
        epilogue_warps=epilogue_warps,
    )
