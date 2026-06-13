"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete padded avg-pool-backward plus three-add bf16 producer, reverse-gathers the low-memory max-pool scatter into the bf16 BN/ReLU gate, accumulates both returned channel reductions, and emits the dependent bf16 BN-backward tensor without materializing the dense f32 scatter target, whereas Inductor lowers avg-pool-backward, add, `_low_memory_max_pool_offsets_to_indices`, `scatter_add`, bf16 cast, mask, reductions, and dense epilogue as separate generic regions over large intermediates; Inductor cannot do this today because scheduler/codegen treats the max-pool offset scatter and preceding pooled-add stencil as opaque producers instead of a structured scatter-reduce source feeding reductions and side-output epilogues; the fix is SCATTER_REDUCE: add a guarded pooling-scatter-reduce template that preserves bf16 rounding boundaries while fusing the structured scatter source with channel reductions and the BN-backward epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 192
SRC_H = 35
SRC_W = 35
SRC_HW = 1225
OUT_H = 71
OUT_W = 71
OUT_HW = 5041
R = N * OUT_HW
REDUCTION_SCALE = 1.5497917079944455e-06


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _avgpool_term(value):
    return _f32_mul(value.to(tl.float32), 0.1111111111111111).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    ).to(tl.float32)


@triton.jit
def _source_kernel(
    pool_grad_ptr,
    add0_ptr,
    add1_ptr,
    add2_ptr,
    source_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < TOTAL
    c = linear % 192
    w = (linear // 192) % 35
    h = (linear // (192 * 35)) % 35
    n = linear // (192 * 1225)
    base = n * (192 * 1225) + h * (35 * 192) + w * 192 + c

    up1 = h > 0
    up2 = h > 1
    left1 = w > 0
    left2 = w > 1
    row = 35 * 192

    total = tl.zeros((BLOCK,), dtype=tl.float32)
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row - row - 384, mask=mask & up2 & left2, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row - row - 192, mask=mask & up2 & left1, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row - row, mask=mask & up2, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row - 384, mask=mask & up1 & left2, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row - 192, mask=mask & up1 & left1, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - row, mask=mask & up1, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - 384, mask=mask & left2, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base - 192, mask=mask & left1, other=0.0)))
    total = _f32_add(total, _avgpool_term(tl.load(pool_grad_ptr + base, mask=mask, other=0.0)))

    value = total.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    value = _f32_add(value, tl.load(add0_ptr + linear, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    value = _f32_add(value, tl.load(add1_ptr + linear, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    value = _f32_add(value, tl.load(add2_ptr + linear, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(source_ptr + linear, value, mask=mask)


@triton.jit
def _scatter_value(
    source_ptr,
    offsets_ptr,
    n,
    c,
    h,
    w,
    active,
    R_BLOCK: tl.constexpr,
    C_BLOCK: tl.constexpr,
):
    acc = tl.zeros((R_BLOCK, C_BLOCK), dtype=tl.float32)
    for kh in tl.static_range(0, 3):
        oh_num = h - kh
        valid_h = (oh_num >= 0) & ((oh_num % 2) == 0)
        oh = oh_num // 2
        valid_h = valid_h & (oh < 35)
        for kw in tl.static_range(0, 3):
            ow_num = w - kw
            valid_w = (ow_num >= 0) & ((ow_num % 2) == 0)
            ow = ow_num // 2
            valid = active & valid_h & valid_w & (ow < 35)
            src_offset = n * (192 * 1225) + oh * (35 * 192) + ow * 192 + c
            local = kh * 3 + kw
            index = tl.load(offsets_ptr + src_offset, mask=valid, other=-1).to(tl.int32)
            source = tl.load(source_ptr + src_offset, mask=valid & (index == local), other=0.0).to(tl.float32)
            acc += source
    return acc


@triton.jit
def _partials_kernel(
    source_ptr,
    offsets_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    partial_sum1_ptr,
    partial_sum2_ptr,
    GROUP_R: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    group = tl.program_id(0)
    c_block = tl.program_id(1)
    rows_base = tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    col_mask = cols < 192

    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)
    acc1 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for start in tl.range(0, GROUP_R, BLOCK_R):
        rows = group * GROUP_R + start + rows_base
        row_mask = rows < 645248
        n = rows // 5041
        spatial = rows - n * 5041
        h = spatial // 71
        w = spatial - h * 71
        active = row_mask[:, None] & col_mask[None, :]
        out_offsets = n[:, None] * (192 * 5041) + h[:, None] * (71 * 192) + w[:, None] * 192 + cols[None, :]

        scatter = _scatter_value(source_ptr, offsets_ptr, n[:, None], cols[None, :], h[:, None], w[:, None], active, BLOCK_R, BLOCK_C)
        scatter_bf16 = scatter.to(tl.bfloat16, fp_downcast_rounding="rtne")
        x = tl.load(x_ptr + out_offsets, mask=active, other=0.0).to(tl.float32)
        centered = _f32_sub(x, mean[None, :])
        affine = _f32_add(_f32_mul(_f32_mul(centered, invstd[None, :]), weight[None, :]), bias[None, :])
        affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
        selected = tl.where(affine_bf16 <= 0.0, scalar, scatter_bf16).to(tl.float32)
        selected = tl.where(active, selected, 0.0)
        centered = tl.where(active, centered, 0.0)
        acc1 += tl.sum(selected, axis=0)
        acc2 += tl.sum(_f32_mul(selected, centered), axis=0)

    partial_offsets = group * 192 + cols
    tl.store(partial_sum1_ptr + partial_offsets, acc1, mask=col_mask)
    tl.store(partial_sum2_ptr + partial_offsets, acc2, mask=col_mask)


@triton.jit
def _final_sums_kernel(
    partial_sum1_ptr,
    partial_sum2_ptr,
    invstd_ptr,
    sum1_ptr,
    sum2_ptr,
    mul10_ptr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c_block = tl.program_id(0)
    groups = tl.arange(0, BLOCK_GROUPS)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (groups[:, None] < NUM_GROUPS) & (cols[None, :] < 192)
    offsets = groups[:, None] * 192 + cols[None, :]
    partial1 = tl.load(partial_sum1_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    partial2 = tl.load(partial_sum2_ptr + offsets, mask=mask, other=0.0, eviction_policy="evict_first").to(tl.float32)
    sum1 = tl.sum(partial1, axis=0)
    sum2 = tl.sum(partial2, axis=0)
    col_mask = cols < 192
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    tl.store(sum1_ptr + cols, sum1, mask=col_mask)
    tl.store(sum2_ptr + cols, sum2, mask=col_mask)
    tl.store(mul10_ptr + cols, _f32_mul(sum2, invstd), mask=col_mask)


@triton.jit
def _output_kernel(
    source_ptr,
    offsets_ptr,
    x_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    scalar_ptr,
    sum1_ptr,
    sum2_ptr,
    out_ptr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    r_block = tl.program_id(0)
    c_block = tl.program_id(1)
    rows = r_block * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    row_mask = rows < 645248
    col_mask = cols < 192
    active = row_mask[:, None] & col_mask[None, :]

    n = rows // 5041
    spatial = rows - n * 5041
    h = spatial // 71
    w = spatial - h * 71
    out_offsets = n[:, None] * (192 * 5041) + h[:, None] * (71 * 192) + w[:, None] * 192 + cols[None, :]

    scatter = _scatter_value(source_ptr, offsets_ptr, n[:, None], cols[None, :], h[:, None], w[:, None], active, BLOCK_R, BLOCK_C)
    scatter_bf16 = scatter.to(tl.bfloat16, fp_downcast_rounding="rtne")
    x = tl.load(x_ptr + out_offsets, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    sum1 = tl.load(sum1_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    sum2 = tl.load(sum2_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    scalar = tl.load(scalar_ptr)

    centered = _f32_sub(x, mean[None, :])
    affine = _f32_add(_f32_mul(_f32_mul(centered, invstd[None, :]), weight[None, :]), bias[None, :])
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    selected = tl.where(affine_bf16 <= 0.0, scalar, scatter_bf16).to(tl.float32)

    mean_term = _f32_mul(sum1[None, :], 1.5497917079944455e-06)
    var_term = _f32_mul(_f32_mul(_f32_mul(sum2[None, :], 1.5497917079944455e-06), invstd[None, :]), invstd[None, :])
    out = _f32_sub(selected, _f32_mul(centered, var_term))
    out = _f32_sub(out, mean_term)
    out = _f32_mul(out, _f32_mul(invstd[None, :], weight[None, :]))
    tl.store(out_ptr + out_offsets, out.to(tl.bfloat16, fp_downcast_rounding="rtne"), mask=active)


@oracle_impl(
    hardware="B200",
    point="e5f2e3e0",
    SOURCE_BLOCK=256,
    GROUP_R=1024,
    REDUCE_BLOCK_R=64,
    BLOCK_C=16,
    FINAL_BLOCK_C=16,
    OUT_BLOCK_R=128,
    num_warps_reduce=8,
    num_warps_output=8,
)
def oracle_forward(
    inputs,
    *,
    SOURCE_BLOCK: int,
    GROUP_R: int,
    REDUCE_BLOCK_R: int,
    BLOCK_C: int,
    FINAL_BLOCK_C: int,
    OUT_BLOCK_R: int,
    num_warps_reduce: int,
    num_warps_output: int,
):
    (
        arg0,
        _arg1,
        arg2,
        arg3,
        arg4,
        offsets,
        x,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        *_shape_args,
    ) = inputs
    num_groups = triton.cdiv(R, GROUP_R)
    source = torch.empty_strided((N, C, SRC_H, SRC_W), (C * SRC_HW, 1, SRC_W * C, C), device=arg0.device, dtype=torch.bfloat16)
    partial_sum1 = torch.empty_strided((num_groups, C), (C, 1), device=arg0.device, dtype=torch.float32)
    partial_sum2 = torch.empty_strided((num_groups, C), (C, 1), device=arg0.device, dtype=torch.float32)
    sum1 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    sum2 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    mul10 = torch.empty_strided((C,), (1,), device=arg0.device, dtype=torch.float32)
    out = torch.empty_strided(tuple(x.shape), tuple(x.stride()), device=arg0.device, dtype=torch.bfloat16)

    _source_kernel[(triton.cdiv(source.numel(), SOURCE_BLOCK),)](
        arg0,
        arg2,
        arg3,
        arg4,
        source,
        TOTAL=source.numel(),
        BLOCK=SOURCE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _partials_kernel[(num_groups, triton.cdiv(C, BLOCK_C))](
        source,
        offsets,
        x,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        partial_sum1,
        partial_sum2,
        GROUP_R=GROUP_R,
        BLOCK_R=REDUCE_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_reduce,
        num_stages=3,
    )
    _final_sums_kernel[(triton.cdiv(C, FINAL_BLOCK_C),)](
        partial_sum1,
        partial_sum2,
        invstd,
        sum1,
        sum2,
        mul10,
        NUM_GROUPS=num_groups,
        BLOCK_GROUPS=_next_power_of_2(num_groups),
        BLOCK_C=FINAL_BLOCK_C,
        num_warps=8,
        num_stages=3,
    )
    _output_kernel[(triton.cdiv(R, OUT_BLOCK_R), triton.cdiv(C, BLOCK_C))](
        source,
        offsets,
        x,
        mean,
        invstd,
        weight,
        bias,
        scalar,
        sum1,
        sum2,
        out,
        BLOCK_R=OUT_BLOCK_R,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps_output,
        num_stages=3,
    )
    return sum1, mul10, out
