"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete bf16 phlippe DenseNet BN-backward plus pool-backward tail, returning the f32 `sum(where)`, the f32 scale-gradient vector, and the bf16 `[128, 64, 32, 32]` avg-pool-backward tensor. Inductor currently schedules the masked bf16 `where` producer, sibling channel reductions, reduction-dependent BN epilogue, sliced residual add, and 2x2 pool-backward expansion as separate generic regions; it cannot do this today because scheduler/codegen does not form one full-scope multi-output reduction plan whose finalized channel scalars feed a layout-changing bf16 consumer while preserving explicit bf16/f32 cast boundaries. The fix is SCHEDULER_FUSION: share the masked producer across both channel reductions, finalize the small per-channel summaries once, and sink them into a direct structured 2x2 pool-backward writer."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
INPUT_C = 80
C = 64
H = 16
W = 16
OUT_H = 32
OUT_W = 32
HW = H * W
OUT_HW = OUT_H * OUT_W
TOTAL_SPATIAL = N * HW
NUMEL = N * C * HW
SLICE_START = 16
SCALE = 3.0517578125e-05


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
def _bf16_add(a, b):
    return _f32_add(a.to(tl.float32), b.to(tl.float32)).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )


@triton.jit
def _dual_reduce_partial_kernel(
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < 32768

    n = k // 256
    hw = k - n * 256
    offsets = n * 16384 + c * 256 + hw

    mask_input = tl.load(mask_input_ptr + offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source = tl.load(source_ptr + offsets, mask=active, other=0.0)
    selected_bf16 = tl.where(mask_input <= 0.0, fill_value, source)
    selected = selected_bf16.to(tl.float32)

    centered_source = tl.load(centered_source_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(centered_source, mean)
    product = _f32_mul(selected, centered)

    partial_offset = c * NUM_TILES + tile
    tl.store(partial_sum_ptr + partial_offset, tl.sum(tl.where(active, selected, 0.0), axis=0))
    tl.store(partial_dot_ptr + partial_offset, tl.sum(tl.where(active, product, 0.0), axis=0))


@triton.jit
def _finalize_reduce_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scale_grad_ptr,
    mean_term_ptr,
    variance_scale_ptr,
    output_scale_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    tile_mask = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles

    sum_values = tl.load(partial_sum_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)
    dot_values = tl.load(partial_dot_ptr + offsets, mask=tile_mask, other=0.0).to(tl.float32)
    sum_value = tl.sum(sum_values, axis=0)
    dot_value = tl.sum(dot_values, axis=0)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    dot_mean = _f32_mul(dot_value, 3.0517578125e-05)
    invstd_sq = _f32_mul(invstd, invstd)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scale_grad_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, _f32_mul(sum_value, 3.0517578125e-05))
    tl.store(variance_scale_ptr + c, _f32_mul(dot_mean, invstd_sq))
    tl.store(output_scale_ptr + c, _f32_mul(invstd, weight))


@triton.jit
def _pool_epilogue_kernel(
    residual_ptr,
    mask_input_ptr,
    fill_ptr,
    source_ptr,
    centered_source_ptr,
    mean_ptr,
    mean_term_ptr,
    variance_scale_ptr,
    output_scale_ptr,
    pool_out_ptr,
    BLOCK_ELEMS: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_ELEMS + tl.arange(0, BLOCK_ELEMS)
    active = linear < 2097152

    hw = linear % 256
    w = hw % 16
    h = hw // 16
    c = (linear // 256) % 64
    n = linear // 16384

    compact_offsets = n * 16384 + c * 256 + hw
    residual_offsets = n * 20480 + (c + 16) * 256 + hw

    mask_input = tl.load(mask_input_ptr + compact_offsets, mask=active, other=0.0)
    fill_value = tl.load(fill_ptr)
    source = tl.load(source_ptr + compact_offsets, mask=active, other=0.0)
    selected_bf16 = tl.where(mask_input <= 0.0, fill_value, source)
    selected = selected_bf16.to(tl.float32)

    centered_source = tl.load(
        centered_source_ptr + compact_offsets, mask=active, other=0.0
    ).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered = _f32_sub(centered_source, mean)

    variance_scale = tl.load(variance_scale_ptr + c, mask=active, other=0.0).to(
        tl.float32
    )
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    output_scale = tl.load(output_scale_ptr + c, mask=active, other=0.0).to(
        tl.float32
    )
    after_variance = _f32_sub(selected, _f32_mul(centered, variance_scale))
    after_mean = _f32_sub(after_variance, mean_term)
    grad_bf16 = _f32_mul(after_mean, output_scale).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    residual = tl.load(residual_ptr + residual_offsets, mask=active, other=0.0)
    add_bf16 = _bf16_add(residual, grad_bf16)
    pool_value = _f32_mul(add_bf16.to(tl.float32), 0.25).to(
        tl.bfloat16, fp_downcast_rounding="rtne"
    )

    out_base = n * 65536 + c * 1024 + (h * 2) * 32 + w * 2
    tl.store(pool_out_ptr + out_base, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 1, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 32, pool_value, mask=active)
    tl.store(pool_out_ptr + out_base + 33, pool_value, mask=active)


# phlippe_densenet train, N=128 C=64 H=W=16, residual slice channels 16:80.
@oracle_impl(
    hardware="B200",
    point="cc6a993c",
    BLOCK_K=1024,
    BLOCK_ELEMS=256,
    num_warps=4,
)
def oracle_forward(inputs, *, BLOCK_K: int, BLOCK_ELEMS: int, num_warps: int):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        _arg8_1,
    ) = inputs
    device = arg0_1.device
    num_tiles = triton.cdiv(TOTAL_SPATIAL, BLOCK_K)
    block_tiles = triton.next_power_of_2(num_tiles)

    partial_sum = torch.empty_strided(
        (C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32
    )
    partial_dot = torch.empty_strided(
        (C, num_tiles), (num_tiles, 1), device=device, dtype=torch.float32
    )
    sum_out = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    scale_grad = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    mean_term = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    variance_scale = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    output_scale = torch.empty_strided((C,), (1,), device=device, dtype=torch.float32)
    pool_out = torch.empty_strided(
        (N, C, OUT_H, OUT_W),
        (C * OUT_HW, OUT_HW, OUT_W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _dual_reduce_partial_kernel[(C, num_tiles)](
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        partial_sum,
        partial_dot,
        NUM_TILES=num_tiles,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=3,
    )
    _finalize_reduce_kernel[(C,)](
        partial_sum,
        partial_dot,
        arg6_1,
        arg7_1,
        sum_out,
        scale_grad,
        mean_term,
        variance_scale,
        output_scale,
        NUM_TILES=num_tiles,
        BLOCK_TILES=block_tiles,
        num_warps=4,
        num_stages=3,
    )
    _pool_epilogue_kernel[(triton.cdiv(NUMEL, BLOCK_ELEMS),)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        mean_term,
        variance_scale,
        output_scale,
        pool_out,
        BLOCK_ELEMS=BLOCK_ELEMS,
        num_warps=4,
        num_stages=3,
    )

    return sum_out, scale_grad, pool_out
