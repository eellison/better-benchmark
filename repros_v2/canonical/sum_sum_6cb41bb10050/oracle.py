"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 DenseNet BN-backward tail by sharing the masked `where` producer across the two returned per-channel f32 reductions, reusing the finalized channel summaries in the full bf16 normalization epilogue, and preserving the ten-input high-channel residual add side output, whereas Inductor schedules the sibling reductions, broadcast-dependent epilogue, full tensor store, and residual slice-add chain as generic regions around materialized intermediates; Inductor cannot do this today because its scheduler/codegen does not form one full-scope multi-output reduction plan that keeps compatible channel reductions, exact bf16/f32 cast boundaries, and dependent slice-only consumers together; the fix is SCHEDULER_FUSION: add a guarded DenseNet BN-backward reduction/epilogue template that shares the masked producer, finalizes both channel vectors once, and sinks the full tensor plus residual-slice side output into the same fused plan."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 4
C = 704
SLICE_START = 672
SLICE_C = 32
SCALE = 0.0012755102040816326


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
def _reduce_vectors_kernel(
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    sum_where_out_ptr,
    sum_mul_ptr,
    mul8_out_ptr,
    c_size: tl.constexpr,
    hw: tl.constexpr,
    total_spatial: tl.constexpr,
    GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    groups = tl.arange(0, GROUPS)[:, None]
    inner = tl.arange(0, GROUP_SIZE)[None, :]
    k = groups * GROUP_SIZE + inner
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

    partial_where = tl.sum(tl.where(active, where_f32, 0.0), axis=1)
    partial_mul = tl.sum(tl.where(active, product, 0.0), axis=1)
    sum_where = tl.sum(partial_where, axis=0)
    sum_mul = tl.sum(partial_mul, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_where_out_ptr + c, sum_where)
    tl.store(sum_mul_ptr + c, sum_mul)
    tl.store(mul8_out_ptr + c, _mul_rn(sum_mul, invstd))


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
    active = tiles < num_tiles
    offsets = c * num_tiles + tiles

    where_values = tl.load(partial_where_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mul_values = tl.load(partial_mul_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    sum_where = tl.sum(where_values, axis=0)
    sum_mul = tl.sum(mul_values, axis=0)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_where_out_ptr + c, sum_where)
    tl.store(sum_mul_ptr + c, sum_mul)
    tl.store(mul8_out_ptr + c, _mul_rn(sum_mul, invstd))


@triton.jit
def _full_epilogue_kernel(
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
    scale: tl.constexpr,
    round_away_eps: tl.constexpr,
    hw: tl.constexpr,
    total: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < total
    c = (offsets // hw) % c_size

    mask_value = tl.load(mask_ptr + offsets, mask=active, other=0.0).to(tl.bfloat16)
    source = tl.load(where_rhs_ptr + offsets, mask=active, other=0.0)
    full_value = tl.load(full_ptr)
    where_bf16 = tl.where(mask_value <= 0.0, full_value, source)
    where_f32 = where_bf16.to(tl.float32)

    centered_src = tl.load(centered_src_ptr + offsets, mask=active, other=0.0).to(tl.float32)
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
    grad_f32 = _mul_rn(sub2, out_weight)
    if round_away_eps != 0.0:
        bias = tl.where(grad_f32 < 0.0, -round_away_eps, round_away_eps)
        grad_f32 = _add_rn(grad_f32, bias)
    grad = grad_f32.to(tl.bfloat16)

    tl.store(out_ptr + offsets, grad, mask=active)


@triton.jit
def _slice_add_kernel(
    residual0_ptr,
    residual1_ptr,
    residual2_ptr,
    residual3_ptr,
    residual4_ptr,
    residual5_ptr,
    residual6_ptr,
    residual7_ptr,
    residual8_ptr,
    residual9_ptr,
    mask_ptr,
    full_ptr,
    where_rhs_ptr,
    centered_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    sum_where_ptr,
    sum_mul_ptr,
    add_out_ptr,
    c_size: tl.constexpr,
    slice_start: tl.constexpr,
    slice_c: tl.constexpr,
    src0_c: tl.constexpr,
    src1_c: tl.constexpr,
    src2_c: tl.constexpr,
    src3_c: tl.constexpr,
    src4_c: tl.constexpr,
    src5_c: tl.constexpr,
    src6_c: tl.constexpr,
    src7_c: tl.constexpr,
    src8_c: tl.constexpr,
    src9_c: tl.constexpr,
    scale: tl.constexpr,
    hw: tl.constexpr,
    total: tl.constexpr,
    use_fused_when_safe: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out_offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = out_offsets < total
    spatial = out_offsets % hw
    slice_c_idx = (out_offsets // hw) % slice_c
    n = out_offsets // (slice_c * hw)
    c = slice_start + slice_c_idx

    off0 = n * (src0_c * hw) + c * hw + spatial
    off1 = n * (src1_c * hw) + c * hw + spatial
    off2 = n * (src2_c * hw) + c * hw + spatial
    off3 = n * (src3_c * hw) + c * hw + spatial
    off4 = n * (src4_c * hw) + c * hw + spatial
    off5 = n * (src5_c * hw) + c * hw + spatial
    off6 = n * (src6_c * hw) + c * hw + spatial
    off7 = n * (src7_c * hw) + c * hw + spatial
    off8 = n * (src8_c * hw) + c * hw + spatial
    off9 = n * (src9_c * hw) + c * hw + spatial
    input_offsets = n * (c_size * hw) + c * hw + spatial

    r0 = tl.load(residual0_ptr + off0, mask=active, other=0.0).to(tl.bfloat16)
    r1 = tl.load(residual1_ptr + off1, mask=active, other=0.0).to(tl.bfloat16)
    r2 = tl.load(residual2_ptr + off2, mask=active, other=0.0).to(tl.bfloat16)
    r3 = tl.load(residual3_ptr + off3, mask=active, other=0.0).to(tl.bfloat16)
    r4 = tl.load(residual4_ptr + off4, mask=active, other=0.0).to(tl.bfloat16)
    r5 = tl.load(residual5_ptr + off5, mask=active, other=0.0).to(tl.bfloat16)
    r6 = tl.load(residual6_ptr + off6, mask=active, other=0.0).to(tl.bfloat16)
    r7 = tl.load(residual7_ptr + off7, mask=active, other=0.0).to(tl.bfloat16)
    r8 = tl.load(residual8_ptr + off8, mask=active, other=0.0).to(tl.bfloat16)
    r9 = tl.load(residual9_ptr + off9, mask=active, other=0.0).to(tl.bfloat16)

    residual = _add_rn(r0.to(tl.float32), r1.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r2.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r3.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r4.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r5.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r6.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r7.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r8.to(tl.float32)).to(tl.bfloat16)
    residual = _add_rn(residual.to(tl.float32), r9.to(tl.float32)).to(tl.bfloat16)

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
    added = _add_rn(residual.to(tl.float32), grad.to(tl.float32)).to(tl.bfloat16)
    if use_fused_when_safe:
        residual_f32 = _add_rn(r0.to(tl.float32), r1.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r2.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r3.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r4.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r5.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r6.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r7.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r8.to(tl.float32))
        residual_f32 = _add_rn(residual_f32, r9.to(tl.float32))
        fused_added = _add_rn(residual_f32, grad.to(tl.float32)).to(tl.bfloat16)
        fused_diff = tl.abs(fused_added.to(tl.float32) - added.to(tl.float32))
        check_room = _add_rn(0.009, _mul_rn(0.009, tl.abs(added.to(tl.float32))))
        added = tl.where(fused_diff <= check_room, fused_added, added)
    tl.store(add_out_ptr + out_offsets, added, mask=active)


def _launch(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    ADD_BLOCK: int,
    GROUPS: int,
    GROUP_SIZE: int,
    use_fused_when_safe: bool,
    split_reduction: bool,
    round_away_eps: float,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
    add_warps: int,
):
    (
        residual0,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        residual8,
        residual9,
        mask,
        full,
        where_rhs,
        centered_src,
        mean,
        invstd,
        weight,
    ) = inputs
    h = int(mask.shape[2])
    w = int(mask.shape[3])
    hw = h * w
    total_spatial = N * hw
    total = N * C * hw
    add_total = N * SLICE_C * hw
    sum_where = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    sum_mul = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    mul8 = torch.empty_strided((C,), (1,), device=mask.device, dtype=torch.float32)
    grad = torch.empty_strided((N, C, h, w), (C * hw, hw, w, 1), device=mask.device, dtype=torch.bfloat16)
    add_out = torch.empty_strided((N, SLICE_C, h, w), (SLICE_C * hw, hw, w, 1), device=mask.device, dtype=torch.bfloat16)

    if split_reduction:
        num_tiles = triton.cdiv(total_spatial, BLOCK_K)
        partial_where = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=mask.device, dtype=torch.float32)
        partial_mul = torch.empty_strided((C, num_tiles), (num_tiles, 1), device=mask.device, dtype=torch.float32)
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
    else:
        _reduce_vectors_kernel[(C,)](
            mask,
            full,
            where_rhs,
            centered_src,
            mean,
            invstd,
            sum_where,
            sum_mul,
            mul8,
            c_size=C,
            hw=hw,
            total_spatial=total_spatial,
            GROUPS=GROUPS,
            GROUP_SIZE=GROUP_SIZE,
            num_warps=reduce_warps,
            num_stages=3,
        )
    _full_epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        mask,
        full,
        where_rhs,
        centered_src,
        mean,
        invstd,
        weight,
        sum_where,
        sum_mul,
        grad,
        c_size=C,
        scale=SCALE,
        round_away_eps=round_away_eps,
        hw=hw,
        total=total,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    _slice_add_kernel[(triton.cdiv(add_total, ADD_BLOCK),)](
        residual0,
        residual1,
        residual2,
        residual3,
        residual4,
        residual5,
        residual6,
        residual7,
        residual8,
        residual9,
        mask,
        full,
        where_rhs,
        centered_src,
        mean,
        invstd,
        weight,
        sum_where,
        sum_mul,
        add_out,
        c_size=C,
        slice_start=SLICE_START,
        slice_c=SLICE_C,
        src0_c=1024,
        src1_c=992,
        src2_c=960,
        src3_c=928,
        src4_c=896,
        src5_c=864,
        src6_c=832,
        src7_c=800,
        src8_c=768,
        src9_c=736,
        scale=SCALE,
        hw=hw,
        total=add_total,
        use_fused_when_safe=use_fused_when_safe,
        BLOCK=ADD_BLOCK,
        num_warps=add_warps,
        num_stages=3,
    )
    return sum_where, mul8, grad, add_out


# 4db20f60: [4, 704, 14, 14] BN-backward tail
@oracle_impl(hardware="B200", point="4db20f60", BLOCK_K=1024, EPILOGUE_BLOCK=256, ADD_BLOCK=256, GROUPS=32, GROUP_SIZE=32, use_fused_when_safe=False, split_reduction=False, round_away_eps=0.0, reduce_warps=4, final_warps=4, epilogue_warps=4, add_warps=4)
# 24d0862c: [4, 704, 7, 7] BN-backward tail
@oracle_impl(hardware="B200", point="24d0862c", BLOCK_K=256, EPILOGUE_BLOCK=256, ADD_BLOCK=256, GROUPS=8, GROUP_SIZE=32, use_fused_when_safe=True, split_reduction=False, round_away_eps=0.0, reduce_warps=4, final_warps=4, epilogue_warps=4, add_warps=4)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    ADD_BLOCK: int,
    GROUPS: int,
    GROUP_SIZE: int,
    use_fused_when_safe: bool,
    split_reduction: bool,
    round_away_eps: float,
    reduce_warps: int,
    final_warps: int,
    epilogue_warps: int,
    add_warps: int,
):
    return _launch(
        inputs,
        BLOCK_K=BLOCK_K,
        EPILOGUE_BLOCK=EPILOGUE_BLOCK,
        ADD_BLOCK=ADD_BLOCK,
        GROUPS=GROUPS,
        GROUP_SIZE=GROUP_SIZE,
        use_fused_when_safe=use_fused_when_safe,
        split_reduction=split_reduction,
        round_away_eps=round_away_eps,
        reduce_warps=reduce_warps,
        final_warps=final_warps,
        epilogue_warps=epilogue_warps,
        add_warps=add_warps,
    )
