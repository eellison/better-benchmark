"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete ShuffleNet bf16 channel-shuffle plus BatchNorm-backward-style scope by writing the returned contiguous `cat -> view -> permute -> clone -> view` tensor directly from its two sources, sharing the high-half shuffled producer across both f32 channel reductions, and feeding the finalized summaries into the returned channels-last bf16 dense epilogue, whereas Inductor schedules the cat/shuffle materialization, bf16 affine/ReLU/where producer, sibling reductions, scale-gradient vector, and dependent broadcast epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because scheduler/codegen does not keep a fixed channel-shuffle layout producer virtual across a multi-output channel reduction with a dependent dense tensor consumer while preserving bf16 cast boundaries and output strides; the fix is SCHEDULER_FUSION: teach reduction scheduling to inline static ShuffleNet channel-shuffle producers and plan the paired per-channel accumulators plus dense BN-backward epilogue as one full-scope template."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
C = 116
OUT_C = 232
H = 14
W = 14
HW = H * W
NHW = N * HW
SCALE = 3.985969387755102e-05


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
def _load_cat_value(
    first_ptr,
    second_ptr,
    n,
    cat_c,
    hw,
    active,
    C_: tl.constexpr,
    OUT_C_: tl.constexpr,
    HW_: tl.constexpr,
):
    from_first = cat_c < C_
    second_c = tl.where(from_first, 0, cat_c - C_)
    first_value = tl.load(
        first_ptr + n * OUT_C_ * HW_ + cat_c * HW_ + hw,
        mask=active & from_first,
        other=0.0,
    )
    second_value = tl.load(
        second_ptr + n * C_ * HW_ + hw * C_ + second_c,
        mask=active & ~from_first,
        other=0.0,
    )
    return tl.where(from_first, first_value, second_value)


@triton.jit
def _partial_reduce_shuffle_kernel(
    first_ptr,
    second_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    shuffle_out_ptr,
    partials_ptr,
    C_: tl.constexpr,
    OUT_C_: tl.constexpr,
    HW_: tl.constexpr,
    NHW_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < NHW_
    n = k // HW_
    hw = k - n * HW_

    low = _load_cat_value(first_ptr, second_ptr, n, c * 2, hw, active, C_, OUT_C_, HW_)
    high = _load_cat_value(
        first_ptr, second_ptr, n, c * 2 + 1, hw, active, C_, OUT_C_, HW_
    )
    out_base = n * OUT_C_ * HW_ + c * HW_ + hw
    tl.store(shuffle_out_ptr + out_base, low, mask=active)
    tl.store(shuffle_out_ptr + out_base + C_ * HW_, high, mask=active)

    bn_offsets = n * C_ * HW_ + hw * C_ + c
    bn_input = tl.load(bn_input_ptr + bn_offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c).to(tl.float32)
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    fill_value = tl.load(fill_ptr).to(tl.float32)

    centered = _f32_sub(bn_input, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    where_value = tl.where(affine_bf16 <= 0.0, fill_value, high.to(tl.float32))
    where_value = tl.where(active, where_value, 0.0)
    centered = tl.where(active, centered, 0.0)

    partial_offset = c * NUM_TILES + tile
    plane = C_ * NUM_TILES
    tl.store(partials_ptr + partial_offset, tl.sum(where_value, axis=0))
    tl.store(
        partials_ptr + plane + partial_offset,
        tl.sum(_f32_mul(where_value, centered), axis=0),
    )


@triton.jit
def _finalize_kernel(
    partials_ptr,
    invstd_ptr,
    sum_out_ptr,
    scale_grad_out_ptr,
    stats_ptr,
    C_: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_TILES
    offsets = c * NUM_TILES + tiles
    plane = C_ * NUM_TILES

    sum_where = tl.sum(
        tl.load(partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_centered = tl.sum(
        tl.load(partials_ptr + plane + offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_where)
    tl.store(scale_grad_out_ptr + c, _f32_mul(sum_centered, invstd))
    tl.store(stats_ptr + c, _f32_mul(sum_where, SCALE_))
    tl.store(
        stats_ptr + C_ + c,
        _f32_mul(_f32_mul(sum_centered, SCALE_), _f32_mul(invstd, invstd)),
    )


@triton.jit
def _epilogue_kernel(
    first_ptr,
    second_ptr,
    bn_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    stats_ptr,
    out_ptr,
    C_: tl.constexpr,
    OUT_C_: tl.constexpr,
    HW_: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < NUMEL
    c = offsets % C_
    nhw = offsets // C_
    hw = nhw % HW_
    n = nhw // HW_

    source = _load_cat_value(
        first_ptr, second_ptr, n, c * 2 + 1, hw, active, C_, OUT_C_, HW_
    ).to(tl.float32)
    bn_input = tl.load(bn_input_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=active, other=0.0).to(tl.float32)
    fill_value = tl.load(fill_ptr).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=active, other=0.0).to(tl.float32)
    centered_term = tl.load(stats_ptr + C_ + c, mask=active, other=0.0).to(
        tl.float32
    )

    centered = _f32_sub(bn_input, mean)
    normalized = _f32_mul(centered, invstd)
    affine = _f32_add(_f32_mul(normalized, weight), bias)
    affine_bf16 = affine.to(tl.bfloat16, fp_downcast_rounding="rtne")
    where_value = tl.where(affine_bf16 <= 0.0, fill_value, source)
    out = _f32_sub(where_value, _f32_mul(centered, centered_term))
    out = _f32_sub(out, mean_term)
    out = _f32_mul(out, _f32_mul(invstd, weight))
    tl.store(
        out_ptr + offsets,
        out.to(tl.bfloat16, fp_downcast_rounding="rtne"),
        mask=active,
    )


@oracle_impl(
    hardware="B200",
    point="82bc110b",
    BLOCK_K=1024,
    EPILOGUE_BLOCK=512,
    reduce_warps=8,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    BLOCK_K: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    first, second, bn_input, mean, invstd, weight, bias, fill, _shape0, _shape1 = inputs
    num_tiles = triton.cdiv(NHW, BLOCK_K)

    shuffle_out = torch.empty_strided(
        (N, OUT_C, H, W),
        (OUT_C * HW, HW, W, 1),
        device=first.device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty_strided((C,), (1,), device=first.device, dtype=torch.float32)
    scale_grad_out = torch.empty_strided(
        (C,), (1,), device=first.device, dtype=torch.float32
    )
    out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=first.device,
        dtype=torch.bfloat16,
    )
    partials = torch.empty((2, C, num_tiles), device=first.device, dtype=torch.float32)
    stats = torch.empty((2, C), device=first.device, dtype=torch.float32)

    _partial_reduce_shuffle_kernel[(C, num_tiles)](
        first,
        second,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        shuffle_out,
        partials,
        C_=C,
        OUT_C_=OUT_C,
        HW_=HW,
        NHW_=NHW,
        NUM_TILES=num_tiles,
        BLOCK_K=BLOCK_K,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partials,
        invstd,
        sum_out,
        scale_grad_out,
        stats,
        C_=C,
        NUM_TILES=num_tiles,
        BLOCK_TILES=triton.next_power_of_2(num_tiles),
        SCALE_=SCALE,
        num_warps=4,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(N * C * HW, EPILOGUE_BLOCK),)](
        first,
        second,
        bn_input,
        mean,
        invstd,
        weight,
        bias,
        fill,
        stats,
        out,
        C_=C,
        OUT_C_=OUT_C,
        HW_=HW,
        NUMEL=N * C * HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=3,
    )
    return shuffle_out, sum_out, scale_grad_out, out
