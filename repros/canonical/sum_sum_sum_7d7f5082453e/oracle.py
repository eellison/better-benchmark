"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full max-pool-offset scatter/unpool and four bf16 batch-norm-backward reduction/epilogue outputs for shape 83737dd9 in one shared Triton schedule; whereas Inductor lowers the scatter_add, bf16 cast/add boundaries, four mask reductions, and dense epilogues as generic graph fragments; Inductor cannot do this today because it does not recognize the Inception scatter_gather producer feeding sibling BN-backward reductions with shared output scope and channels-last strides; the fix is SCATTER_REDUCE: add a guarded lowering that builds the scatter accumulator once and emits the grouped reductions and dense bf16 epilogues directly."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


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
def _zero_source_kernel(source, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(source + offsets, tl.zeros((BLOCK,), tl.float32), mask=offsets < 45158400)


@triton.jit
def _scatter_source_kernel(arg0, arg1, source, BLOCK: tl.constexpr):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = linear % 288
    tmp = linear // 288
    ow = tmp % 17
    tmp = tmp // 17
    oh = tmp % 17
    n = tmp // 17
    mask = linear < 10653696

    local = tl.load(arg1 + linear, mask=mask, other=0).to(tl.int64)
    kh = local // 3
    kw = local - kh * 3
    h = oh * 2 + kh
    w = ow * 2 + kw

    arg0_offsets = n * 221952 + oh * 13056 + ow * 768 + 480 + c
    source_offsets = n * 352800 + h * 10080 + w * 288 + c
    value = tl.load(arg0 + arg0_offsets, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_add(source + source_offsets, value, sem="relaxed", mask=mask)


@triton.jit
def _load_activation(
    c,
    nhw,
    arg4,
    arg10,
    arg15,
    arg20,
    active,
):
    n = nhw // 1225
    rem = nhw - n * 1225
    h = rem // 35
    w = rem - h * 35

    low_c = c
    mid_c = c - 64
    wide_c = c - 128
    high_c = c - 224

    low_offsets = n * 78400 + h * 2240 + w * 64 + low_c
    mid_offsets = n * 78400 + h * 2240 + w * 64 + mid_c
    wide_offsets = n * 117600 + h * 3360 + w * 96 + wide_c
    high_offsets = n * 78400 + h * 2240 + w * 64 + high_c

    low = c < 64
    mid = (c >= 64) & (c < 128)
    wide = (c >= 128) & (c < 224)
    high = c >= 224

    value_low = tl.load(arg20 + low_offsets, mask=active & low, other=0.0).to(tl.float32)
    value_mid = tl.load(arg15 + mid_offsets, mask=active & mid, other=0.0).to(tl.float32)
    value_wide = tl.load(arg10 + wide_offsets, mask=active & wide, other=0.0).to(tl.float32)
    value_high = tl.load(arg4 + high_offsets, mask=active & high, other=0.0).to(tl.float32)
    return value_low + value_mid + value_wide + value_high


@triton.jit
def _load_vector(c, ptr_high, ptr_wide, ptr_mid, ptr_low, active):
    low = c < 64
    mid = (c >= 64) & (c < 128)
    wide = (c >= 128) & (c < 224)
    high = c >= 224

    value_low = tl.load(ptr_low + c, mask=active & low, other=0.0).to(tl.float32)
    value_mid = tl.load(ptr_mid + c - 64, mask=active & mid, other=0.0).to(tl.float32)
    value_wide = tl.load(ptr_wide + c - 128, mask=active & wide, other=0.0).to(tl.float32)
    value_high = tl.load(ptr_high + c - 224, mask=active & high, other=0.0).to(tl.float32)
    return value_low + value_mid + value_wide + value_high


@triton.jit
def _source_bf16(source, arg2, arg3, c, nhw, active):
    n = nhw // 1225
    rem = nhw - n * 1225
    h = rem // 35
    w = rem - h * 35
    offsets = n * 352800 + h * 10080 + w * 288 + c

    scatter = tl.load(source + offsets, mask=active, other=0.0).to(tl.float32)
    scatter_bf16 = scatter.to(tl.bfloat16, fp_downcast_rounding="rtne")
    add2 = tl.load(arg2 + offsets, mask=active, other=0.0).to(tl.float32)
    tmp = _f32_add(scatter_bf16.to(tl.float32), add2)
    tmp_bf16 = tmp.to(tl.bfloat16, fp_downcast_rounding="rtne")
    add3 = tl.load(arg3 + offsets, mask=active, other=0.0).to(tl.float32)
    final = _f32_add(tmp_bf16.to(tl.float32), add3)
    return final.to(tl.bfloat16, fp_downcast_rounding="rtne")


@triton.jit
def _where_and_centered(
    source,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7,
    arg8,
    fill,
    arg10,
    arg11,
    arg12,
    arg13,
    arg14,
    arg15,
    arg16,
    arg17,
    arg18,
    arg19,
    arg20,
    arg21,
    arg22,
    arg23,
    arg24,
    c,
    nhw,
    active,
):
    source_value = _source_bf16(source, arg2, arg3, c, nhw, active)
    bn_input = _load_activation(c, nhw, arg4, arg10, arg15, arg20, active)
    c_active = c < 288
    mean = _load_vector(c, arg5, arg11, arg16, arg21, c_active)
    invstd = _load_vector(c, arg6, arg12, arg17, arg22, c_active)
    gamma = _load_vector(c, arg7, arg13, arg18, arg23, c_active)
    beta = _load_vector(c, arg8, arg14, arg19, arg24, c_active)

    centered = _f32_sub(bn_input, mean)
    normed = _f32_mul(centered, invstd)
    scaled = _f32_mul(normed, gamma)
    biased = _f32_add(scaled, beta)
    biased_bf16 = biased.to(tl.bfloat16, fp_downcast_rounding="rtne")
    fill_value = tl.load(fill).to(tl.bfloat16)
    selected = tl.where(biased_bf16 <= 0.0, fill_value, source_value).to(tl.float32)
    return selected, centered


@triton.jit
def _partial_reduction_kernel(
    source,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7,
    arg8,
    fill,
    arg10,
    arg11,
    arg12,
    arg13,
    arg14,
    arg15,
    arg16,
    arg17,
    arg18,
    arg19,
    arg20,
    arg21,
    arg22,
    arg23,
    arg24,
    partial0,
    partial1,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c = c_offsets[None, :]
    nhw = k_offsets[:, None]
    active = (c < 288) & (nhw < 156800)

    selected, centered = _where_and_centered(
        source,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        fill,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        c,
        nhw,
        active,
    )
    selected = tl.where(active, selected, 0.0)
    dot = tl.where(active, _f32_mul(selected, centered), 0.0)

    sum0 = tl.sum(selected, axis=0)
    sum1 = tl.sum(dot, axis=0)
    tile_count = tl.cdiv(156800, BLOCK_K)
    out_offsets = c_offsets * tile_count + tl.program_id(1)
    tl.store(partial0 + out_offsets, sum0, mask=c_offsets < 288)
    tl.store(partial1 + out_offsets, sum1, mask=c_offsets < 288)


@triton.jit
def _finalize_reduction_kernel(
    partial0,
    partial1,
    sum0_total,
    sum1_total,
    arg6,
    arg12,
    arg17,
    arg22,
    high_sum,
    high_vec,
    wide_sum,
    wide_vec,
    mid_sum,
    mid_vec,
    low_sum,
    low_vec,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    REDUCTION_TILES: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    c = c_offsets[:, None]
    tiles = tile_offsets[None, :]
    active = (c_offsets < 288)[:, None] & (tile_offsets < REDUCTION_TILES)[None, :]

    offsets = c * REDUCTION_TILES + tiles
    sum0 = tl.sum(tl.load(partial0 + offsets, mask=active, other=0.0), axis=1)
    sum1 = tl.sum(tl.load(partial1 + offsets, mask=active, other=0.0), axis=1)
    c_mask = c_offsets < 288
    tl.store(sum0_total + c_offsets, sum0, mask=c_mask)
    tl.store(sum1_total + c_offsets, sum1, mask=c_mask)

    invstd = _load_vector(c_offsets, arg6, arg12, arg17, arg22, c_mask)
    vec = _f32_mul(sum1, invstd)

    low = c_offsets < 64
    mid = (c_offsets >= 64) & (c_offsets < 128)
    wide = (c_offsets >= 128) & (c_offsets < 224)
    high = c_offsets >= 224

    tl.store(low_sum + c_offsets, sum0, mask=c_mask & low)
    tl.store(low_vec + c_offsets, vec, mask=c_mask & low)
    tl.store(mid_sum + c_offsets - 64, sum0, mask=c_mask & mid)
    tl.store(mid_vec + c_offsets - 64, vec, mask=c_mask & mid)
    tl.store(wide_sum + c_offsets - 128, sum0, mask=c_mask & wide)
    tl.store(wide_vec + c_offsets - 128, vec, mask=c_mask & wide)
    tl.store(high_sum + c_offsets - 224, sum0, mask=c_mask & high)
    tl.store(high_vec + c_offsets - 224, vec, mask=c_mask & high)


@triton.jit
def _epilogue_kernel(
    source,
    arg2,
    arg3,
    arg4,
    arg5,
    arg6,
    arg7,
    arg8,
    fill,
    arg10,
    arg11,
    arg12,
    arg13,
    arg14,
    arg15,
    arg16,
    arg17,
    arg18,
    arg19,
    arg20,
    arg21,
    arg22,
    arg23,
    arg24,
    sum0_total,
    sum1_total,
    high_dense,
    wide_dense,
    mid_dense,
    low_dense,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_offsets = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    c = c_offsets[None, :]
    nhw = k_offsets[:, None]
    active = (c < 288) & (nhw < 156800)

    selected, centered = _where_and_centered(
        source,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        fill,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        c,
        nhw,
        active,
    )
    sum0 = tl.load(sum0_total + c, mask=c < 288, other=0.0)
    sum1 = tl.load(sum1_total + c, mask=c < 288, other=0.0)
    invstd = _load_vector(c, arg6, arg12, arg17, arg22, c < 288)
    gamma = _load_vector(c, arg7, arg13, arg18, arg23, c < 288)

    invstd2 = _f32_mul(invstd, invstd)
    dot_term = _f32_mul(_f32_mul(_f32_mul(sum1, 6.3775510204081635e-06), invstd2), centered)
    mean_term = _f32_mul(sum0, 6.3775510204081635e-06)
    scale = _f32_mul(invstd, gamma)
    out = _f32_mul(_f32_sub(_f32_sub(selected, dot_term), mean_term), scale)
    out_bf16 = out.to(tl.bfloat16, fp_downcast_rounding="rtne")

    n = nhw // 1225
    rem = nhw - n * 1225
    h = rem // 35
    w = rem - h * 35

    low = c < 64
    mid = (c >= 64) & (c < 128)
    wide = (c >= 128) & (c < 224)
    high = c >= 224

    low_offsets = n * 78400 + h * 2240 + w * 64 + c
    mid_offsets = n * 78400 + h * 2240 + w * 64 + (c - 64)
    wide_offsets = n * 117600 + h * 3360 + w * 96 + (c - 128)
    high_offsets = n * 78400 + h * 2240 + w * 64 + (c - 224)

    tl.store(low_dense + low_offsets, out_bf16, mask=active & low)
    tl.store(mid_dense + mid_offsets, out_bf16, mask=active & mid)
    tl.store(wide_dense + wide_offsets, out_bf16, mask=active & wide)
    tl.store(high_dense + high_offsets, out_bf16, mask=active & high)


@oracle_impl(hardware="B200", point="83737dd9")
def oracle_forward(inputs):
    (
        arg0,
        arg1,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        _shape_param_0,
        _shape_param_1,
        _shape_param_2,
        _shape_param_3,
        _shape_param_4,
        _shape_param_5,
        _shape_param_6,
    ) = inputs
    del _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6

    device = arg0.device
    source = torch.empty_strided((128, 288, 35, 35), (352800, 1, 10080, 288), device=device, dtype=torch.float32)
    high_sum = torch.empty((64,), device=device, dtype=torch.float32)
    high_vec = torch.empty((64,), device=device, dtype=torch.float32)
    high_dense = torch.empty_strided((128, 64, 35, 35), (78400, 1, 2240, 64), device=device, dtype=torch.bfloat16)
    wide_sum = torch.empty((96,), device=device, dtype=torch.float32)
    wide_vec = torch.empty((96,), device=device, dtype=torch.float32)
    wide_dense = torch.empty_strided((128, 96, 35, 35), (117600, 1, 3360, 96), device=device, dtype=torch.bfloat16)
    mid_sum = torch.empty((64,), device=device, dtype=torch.float32)
    mid_vec = torch.empty((64,), device=device, dtype=torch.float32)
    mid_dense = torch.empty_strided((128, 64, 35, 35), (78400, 1, 2240, 64), device=device, dtype=torch.bfloat16)
    low_sum = torch.empty((64,), device=device, dtype=torch.float32)
    low_vec = torch.empty((64,), device=device, dtype=torch.float32)
    low_dense = torch.empty_strided((128, 64, 35, 35), (78400, 1, 2240, 64), device=device, dtype=torch.bfloat16)

    block_k = 256
    reduction_tiles = triton.cdiv(156800, block_k)
    partial0 = torch.empty((288, reduction_tiles), device=device, dtype=torch.float32)
    partial1 = torch.empty((288, reduction_tiles), device=device, dtype=torch.float32)
    sum0_total = torch.empty((288,), device=device, dtype=torch.float32)
    sum1_total = torch.empty((288,), device=device, dtype=torch.float32)

    _zero_source_kernel[(triton.cdiv(45158400, 1024),)](source, BLOCK=1024, num_warps=8)
    _scatter_source_kernel[(triton.cdiv(10653696, 256),)](arg0, arg1, source, BLOCK=256, num_warps=8)
    _partial_reduction_kernel[(triton.cdiv(288, 64), reduction_tiles)](
        source,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        partial0,
        partial1,
        BLOCK_C=64,
        BLOCK_K=block_k,
        num_warps=8,
    )
    _finalize_reduction_kernel[(triton.cdiv(288, 8),)](
        partial0,
        partial1,
        sum0_total,
        sum1_total,
        arg6,
        arg12,
        arg17,
        arg22,
        high_sum,
        high_vec,
        wide_sum,
        wide_vec,
        mid_sum,
        mid_vec,
        low_sum,
        low_vec,
        BLOCK_C=8,
        BLOCK_TILES=1024,
        REDUCTION_TILES=reduction_tiles,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(288, 64), reduction_tiles)](
        source,
        arg2,
        arg3,
        arg4,
        arg5,
        arg6,
        arg7,
        arg8,
        arg9,
        arg10,
        arg11,
        arg12,
        arg13,
        arg14,
        arg15,
        arg16,
        arg17,
        arg18,
        arg19,
        arg20,
        arg21,
        arg22,
        arg23,
        arg24,
        sum0_total,
        sum1_total,
        high_dense,
        wide_dense,
        mid_dense,
        low_dense,
        BLOCK_C=64,
        BLOCK_K=block_k,
        num_warps=8,
    )

    return (
        high_sum,
        high_vec,
        high_dense,
        wide_sum,
        wide_vec,
        wide_dense,
        mid_sum,
        mid_vec,
        mid_dense,
        low_sum,
        low_vec,
        low_dense,
    )
