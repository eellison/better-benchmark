"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet bf16 add/copy plus dual BatchNorm-backward return tuple, materializing the returned channels-last add tensor while sharing split-K full and sliced channel reductions with the dependent bf16 tensor epilogues, whereas Inductor schedules the layout copy, sibling reductions, and broadcast backward epilogues as separate generic regions; Inductor cannot do this today because its scheduler has no cooperative split-K multi-output lowering that preserves bf16 cast boundaries across a returned layout-changing producer and overlapping slice reductions; the fix is COOPERATIVE_SPLIT_K: add a layout-aware split-K reduction lowering that co-finalizes compatible channel sums and feeds all dependent vector and tensor outputs."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C = 112
C_SLICE = 56
SLICE_START = 56
H = 14
W = 14
HW = H * W
K_TOTAL = N * HW
NUMEL_FULL = N * C * HW
NUMEL_SLICE = N * C_SLICE * HW
INV_NHW = 9.964923469387754e-06


def _ceil_pow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _materialize_and_partial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_full_ptr,
    arg6_ptr,
    mean_slice_ptr,
    copy_out_ptr,
    partial_full_ptr,
    partial_slice_ptr,
    c_total: tl.constexpr,
    c_slice_total: tl.constexpr,
    slice_start: tl.constexpr,
    hw_total: tl.constexpr,
    k_total: tl.constexpr,
    num_tiles: tl.constexpr,
    block_c: tl.constexpr,
    block_k: tl.constexpr,
):
    c = tl.program_id(0) * block_c + tl.arange(0, block_c)[None, :]
    k = tl.program_id(1) * block_k + tl.arange(0, block_k)[:, None]
    active = (c < c_total) & (k < k_total)

    n = k // hw_total
    hw = k - n * hw_total
    cl_offsets = n * (c_total * hw_total) + hw * c_total + c
    contiguous_offsets = n * (c_total * hw_total) + c * hw_total + hw

    lhs = tl.load(arg0_ptr + contiguous_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + cl_offsets, mask=active, other=0.0).to(tl.float32)
    added_bf16 = (lhs + rhs).to(tl.bfloat16)
    added_f32 = added_bf16.to(tl.float32)
    tl.store(copy_out_ptr + cl_offsets, added_bf16, mask=active)

    centered_full = tl.load(arg2_ptr + cl_offsets, mask=active, other=0.0).to(tl.float32)
    centered_full -= tl.load(mean_full_ptr + c, mask=c < c_total, other=0.0).to(tl.float32)
    sum_full = tl.sum(tl.where(active, added_f32, 0.0), axis=0)
    dot_full = tl.sum(tl.where(active, added_f32 * centered_full, 0.0), axis=0)

    c_vec = tl.program_id(0) * block_c + tl.arange(0, block_c)
    tile = tl.program_id(1)
    full_offsets = tile * c_total + c_vec
    full_plane = num_tiles * c_total
    c_mask = c_vec < c_total
    tl.store(partial_full_ptr + full_offsets, sum_full, mask=c_mask)
    tl.store(partial_full_ptr + full_plane + full_offsets, dot_full, mask=c_mask)

    slice_c = c - slice_start
    slice_active = active & (c >= slice_start)
    slice_offsets = n * (c_slice_total * hw_total) + hw * c_slice_total + slice_c
    centered_slice = tl.load(arg6_ptr + slice_offsets, mask=slice_active, other=0.0).to(tl.float32)
    centered_slice -= tl.load(mean_slice_ptr + slice_c, mask=(c >= slice_start) & (c < c_total), other=0.0).to(tl.float32)
    sum_slice = tl.sum(tl.where(slice_active, added_f32, 0.0), axis=0)
    dot_slice = tl.sum(tl.where(slice_active, added_f32 * centered_slice, 0.0), axis=0)

    slice_c_vec = c_vec - slice_start
    slice_mask = (c_vec >= slice_start) & (c_vec < c_total)
    slice_partial_offsets = tile * c_slice_total + slice_c_vec
    slice_plane = num_tiles * c_slice_total
    tl.store(partial_slice_ptr + slice_partial_offsets, sum_slice, mask=slice_mask)
    tl.store(partial_slice_ptr + slice_plane + slice_partial_offsets, dot_slice, mask=slice_mask)


@triton.jit
def _finalize_full_kernel(
    partial_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    stats_ptr,
    c_total: tl.constexpr,
    inv_nhw: tl.constexpr,
    num_tiles: tl.constexpr,
    block_c: tl.constexpr,
    block_tiles: tl.constexpr,
):
    c = tl.program_id(0) * block_c + tl.arange(0, block_c)[:, None]
    t = tl.arange(0, block_tiles)[None, :]
    active = (c < c_total) & (t < num_tiles)
    offsets = t * c_total + c
    plane = num_tiles * c_total

    sum_x = tl.sum(tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=1)
    sum_dot = tl.sum(tl.load(partial_ptr + plane + offsets, mask=active, other=0.0).to(tl.float32), axis=1)

    c_vec = tl.program_id(0) * block_c + tl.arange(0, block_c)
    c_mask = c_vec < c_total
    invstd = tl.load(invstd_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
    coeff = (sum_dot * inv_nhw) * (invstd * invstd)
    scale = invstd * weight

    tl.store(sum_out_ptr + c_vec, sum_x, mask=c_mask)
    tl.store(vec_out_ptr + c_vec, sum_dot * invstd, mask=c_mask)
    tl.store(stats_ptr + c_vec, sum_x * inv_nhw, mask=c_mask)
    tl.store(stats_ptr + c_total + c_vec, coeff, mask=c_mask)
    tl.store(stats_ptr + 2 * c_total + c_vec, scale, mask=c_mask)


@triton.jit
def _finalize_slice_kernel(
    partial_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    vec_out_ptr,
    stats_ptr,
    c_slice_total: tl.constexpr,
    inv_nhw: tl.constexpr,
    num_tiles: tl.constexpr,
    block_c: tl.constexpr,
    block_tiles: tl.constexpr,
):
    c = tl.program_id(0) * block_c + tl.arange(0, block_c)[:, None]
    t = tl.arange(0, block_tiles)[None, :]
    active = (c < c_slice_total) & (t < num_tiles)
    offsets = t * c_slice_total + c
    plane = num_tiles * c_slice_total

    sum_x = tl.sum(tl.load(partial_ptr + offsets, mask=active, other=0.0).to(tl.float32), axis=1)
    sum_dot = tl.sum(tl.load(partial_ptr + plane + offsets, mask=active, other=0.0).to(tl.float32), axis=1)

    c_vec = tl.program_id(0) * block_c + tl.arange(0, block_c)
    c_mask = c_vec < c_slice_total
    invstd = tl.load(invstd_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_vec, mask=c_mask, other=0.0).to(tl.float32)
    coeff = (sum_dot * inv_nhw) * (invstd * invstd)
    scale = invstd * weight

    tl.store(sum_out_ptr + c_vec, sum_x, mask=c_mask)
    tl.store(vec_out_ptr + c_vec, sum_dot * invstd, mask=c_mask)
    tl.store(stats_ptr + c_vec, sum_x * inv_nhw, mask=c_mask)
    tl.store(stats_ptr + c_slice_total + c_vec, coeff, mask=c_mask)
    tl.store(stats_ptr + 2 * c_slice_total + c_vec, scale, mask=c_mask)


@triton.jit
def _full_epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_full_ptr,
    stats_ptr,
    out_ptr,
    c_total: tl.constexpr,
    hw_total: tl.constexpr,
    numel: tl.constexpr,
    block: tl.constexpr,
):
    linear = tl.program_id(0) * block + tl.arange(0, block)
    active = linear < numel
    n = linear // (c_total * hw_total)
    rem = linear - n * (c_total * hw_total)
    c = rem // hw_total
    hw = rem - c * hw_total
    cl_offsets = n * (c_total * hw_total) + hw * c_total + c

    lhs = tl.load(arg0_ptr + linear, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + cl_offsets, mask=active, other=0.0).to(tl.float32)
    x = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    centered = tl.load(arg2_ptr + cl_offsets, mask=active, other=0.0).to(tl.float32)
    centered -= tl.load(mean_full_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff = tl.load(stats_ptr + c_total + c, mask=active, other=0.0).to(tl.float32)
    scale = tl.load(stats_ptr + 2 * c_total + c, mask=active, other=0.0).to(tl.float32)

    value = x - centered * coeff
    value = value - mean_term
    value = value * scale
    tl.store(out_ptr + linear, value.to(tl.bfloat16), mask=active)


@triton.jit
def _slice_epilogue_kernel(
    arg0_ptr,
    arg1_ptr,
    arg6_ptr,
    mean_slice_ptr,
    stats_ptr,
    out_ptr,
    c_total: tl.constexpr,
    c_slice_total: tl.constexpr,
    slice_start: tl.constexpr,
    hw_total: tl.constexpr,
    numel: tl.constexpr,
    block: tl.constexpr,
):
    linear = tl.program_id(0) * block + tl.arange(0, block)
    active = linear < numel
    n = linear // (c_slice_total * hw_total)
    rem = linear - n * (c_slice_total * hw_total)
    hw = rem // c_slice_total
    c = rem - hw * c_slice_total
    full_c = c + slice_start
    contiguous_offsets = n * (c_total * hw_total) + full_c * hw_total + hw
    full_cl_offsets = n * (c_total * hw_total) + hw * c_total + full_c

    lhs = tl.load(arg0_ptr + contiguous_offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + full_cl_offsets, mask=active, other=0.0).to(tl.float32)
    x = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    centered = tl.load(arg6_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered -= tl.load(mean_slice_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff = tl.load(stats_ptr + c_slice_total + c, mask=active, other=0.0).to(tl.float32)
    scale = tl.load(stats_ptr + 2 * c_slice_total + c, mask=active, other=0.0).to(tl.float32)

    value = x - centered * coeff
    value = value - mean_term
    value = value * scale
    tl.store(out_ptr + linear, value.to(tl.bfloat16), mask=active)


@oracle_impl(
    hardware="B200",
    point="641bb11a",
    block_c=2,
    block_k=1024,
    final_block_c=32,
    full_block=256,
    slice_block=256,
    reduce_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    block_c: int,
    block_k: int,
    final_block_c: int,
    full_block: int,
    slice_block: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    arg0, arg1, arg2, mean_full, inv_full, weight_full, arg6, mean_slice, inv_slice, weight_slice, _, _ = inputs
    device = arg0.device
    num_tiles = triton.cdiv(K_TOTAL, block_k)
    block_tiles = _ceil_pow2(num_tiles)

    copy_out = torch.empty_strided((N, C, H, W), (C * HW, 1, W * C, C), device=device, dtype=torch.bfloat16)
    sum_full = torch.empty((C,), device=device, dtype=torch.float32)
    vec_full = torch.empty((C,), device=device, dtype=torch.float32)
    out_full = torch.empty_strided((N, C, H, W), (C * HW, HW, W, 1), device=device, dtype=torch.bfloat16)
    sum_slice = torch.empty((C_SLICE,), device=device, dtype=torch.float32)
    vec_slice = torch.empty((C_SLICE,), device=device, dtype=torch.float32)
    out_slice = torch.empty_strided((N, C_SLICE, H, W), (C_SLICE * HW, 1, W * C_SLICE, C_SLICE), device=device, dtype=torch.bfloat16)

    partial_full = torch.empty((2, num_tiles, C), device=device, dtype=torch.float32)
    partial_slice = torch.empty((2, num_tiles, C_SLICE), device=device, dtype=torch.float32)
    stats_full = torch.empty((3, C), device=device, dtype=torch.float32)
    stats_slice = torch.empty((3, C_SLICE), device=device, dtype=torch.float32)

    _materialize_and_partial_kernel[(triton.cdiv(C, block_c), num_tiles)](
        arg0,
        arg1,
        arg2,
        mean_full,
        arg6,
        mean_slice,
        copy_out,
        partial_full,
        partial_slice,
        c_total=C,
        c_slice_total=C_SLICE,
        slice_start=SLICE_START,
        hw_total=HW,
        k_total=K_TOTAL,
        num_tiles=num_tiles,
        block_c=block_c,
        block_k=block_k,
        num_warps=reduce_warps,
    )
    _finalize_full_kernel[(triton.cdiv(C, final_block_c),)](
        partial_full,
        inv_full,
        weight_full,
        sum_full,
        vec_full,
        stats_full,
        c_total=C,
        inv_nhw=INV_NHW,
        num_tiles=num_tiles,
        block_c=final_block_c,
        block_tiles=block_tiles,
        num_warps=4,
    )
    _finalize_slice_kernel[(triton.cdiv(C_SLICE, final_block_c),)](
        partial_slice,
        inv_slice,
        weight_slice,
        sum_slice,
        vec_slice,
        stats_slice,
        c_slice_total=C_SLICE,
        inv_nhw=INV_NHW,
        num_tiles=num_tiles,
        block_c=final_block_c,
        block_tiles=block_tiles,
        num_warps=4,
    )
    _full_epilogue_kernel[(triton.cdiv(NUMEL_FULL, full_block),)](
        arg0,
        arg1,
        arg2,
        mean_full,
        stats_full,
        out_full,
        c_total=C,
        hw_total=HW,
        numel=NUMEL_FULL,
        block=full_block,
        num_warps=epilogue_warps,
    )
    _slice_epilogue_kernel[(triton.cdiv(NUMEL_SLICE, slice_block),)](
        arg0,
        arg1,
        arg6,
        mean_slice,
        stats_slice,
        out_slice,
        c_total=C,
        c_slice_total=C_SLICE,
        slice_start=SLICE_START,
        hw_total=HW,
        numel=NUMEL_SLICE,
        block=slice_block,
        num_warps=epilogue_warps,
    )
    return copy_out, sum_full, vec_full, out_full, sum_slice, vec_slice, out_slice
