"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the full GhostNet bf16 hard-sigmoid/pooled-add producer, the live half-channel slice feeding the scalar-fill `where`, both fp32 `[0, 2, 3]` channel reductions, the returned scale-gradient vector, and the final channels-last bf16 BN-backward epilogue while writing the returned `[512, 960, 7, 7]` tensor at its captured stride; Inductor schedules the broadcast 1x1 pool expansion, returned producer, slice, ReLU-gated mask, sibling reductions, and dense epilogue as generic pointwise/reduction regions; Inductor cannot do this today because scheduler/codegen has no guarded structured scatter-reduce plan that treats the broadcast pooled source and sliced returned producer as one reusable source while preserving bf16 cast boundaries and channels-last output scope; the fix is SCATTER_REDUCE: add a GhostNet average-pool/hard-sigmoid BN-backward lowering that writes the required producer once and reduces the dependent sliced channels directly into the BN epilogue."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C_FULL = 960
C = 480
H = 7
W = 7
HW = H * W
K_TOTAL = N * HW
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
def _bf16_round_f32(x):
    return x.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _add1_value(gate_ptr, arg1_ptr, pool_ptr, n, hw, c, active):
    gate_offset = n * 960 + c
    data_offset = n * 47040 + hw * 960 + c

    gate = tl.load(gate_ptr + gate_offset, mask=active, other=0.0).to(tl.float32)
    clamped = tl.minimum(tl.maximum(_f32_add(gate, 3.0), 0.0), 6.0)
    gate_bf16 = _bf16_round_f32(clamped / 6.0)

    lhs = tl.load(arg1_ptr + data_offset, mask=active, other=0.0).to(tl.float32)
    pool = tl.load(pool_ptr + gate_offset, mask=active, other=0.0).to(tl.float32)
    mul_value = _bf16_round_f32(_f32_mul(lhs, gate_bf16))
    pool_value = _bf16_round_f32(pool / 49.0)
    return _bf16_round_f32(_f32_add(mul_value, pool_value))


@triton.jit
def _add1_full_kernel(
    gate_ptr,
    arg1_ptr,
    pool_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL
    c = linear % 960
    k = linear // 960
    n = k // 49
    hw = k - n * 49
    value = _add1_value(gate_ptr, arg1_ptr, pool_ptr, n, hw, c, active)
    out_offset = n * 47040 + hw * 960 + c
    tl.store(out_ptr + out_offset, value, mask=active)


@triton.jit
def _masked_source_from_add1(
    add1_ptr,
    bn_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    n,
    hw,
    c,
    active,
):
    add1_offset = n * 47040 + hw * 960 + c + 480
    add1 = tl.load(add1_ptr + add1_offset, mask=active, other=0.0).to(tl.float32)
    bn_offset = n * 23520 + hw * 480 + c
    bn_src = tl.load(bn_src_ptr + bn_offset, mask=active, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    centered = _f32_sub(bn_src, mean)

    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    normed = _f32_mul(_f32_mul(centered, invstd), weight)
    affine = _bf16_round_f32(_f32_add(normed, bias))
    fill = tl.load(fill_ptr).to(tl.float32)
    source = tl.where(affine <= 0.0, fill, add1).to(tl.float32)
    return source, centered, bn_offset


@triton.jit
def _partial_reduce_kernel(
    add1_ptr,
    bn_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    X_TOTAL: tl.constexpr,
    XBLOCK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    x_idx = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:, None]
    r = tl.arange(0, BLOCK_K)[None, :]
    x_active = x_idx < X_TOTAL
    c = x_idx % 480
    tile = x_idx // 480
    k = tile * BLOCK_K + r
    active = x_active & (k < 25088)
    n = k // 49
    hw = k - n * 49

    source, centered, _ = _masked_source_from_add1(
        add1_ptr,
        bn_src_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        n,
        hw,
        c,
        active,
    )
    source_for_sum = tl.where(active, source, 0.0)
    centered_for_sum = tl.where(active, centered, 0.0)
    tl.store(
        partial_sum_ptr + x_idx,
        tl.sum(source_for_sum, axis=1)[:, None],
        mask=x_active,
    )
    tl.store(
        partial_dot_ptr + x_idx,
        tl.sum(_f32_mul(source_for_sum, centered_for_sum), axis=1)[:, None],
        mask=x_active,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    sum_out_ptr,
    scaled_dot_out_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    NUM_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    active = tiles < NUM_TILES
    offsets = c + 480 * tiles
    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=active, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean_term = _f32_mul(sum_value, 3.985969387755102e-05)
    invstd_sq = _f32_mul(invstd, invstd)
    coeff = _f32_mul(_f32_mul(dot_value, 3.985969387755102e-05), invstd_sq)
    output_scale = _f32_mul(invstd, weight)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))
    tl.store(mean_term_ptr + c, mean_term)
    tl.store(coeff_ptr + c, coeff)
    tl.store(output_scale_ptr + c, output_scale)


@triton.jit
def _epilogue_kernel(
    add1_ptr,
    bn_src_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    bias_ptr,
    fill_ptr,
    mean_term_ptr,
    coeff_ptr,
    output_scale_ptr,
    grad_out_ptr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = linear < TOTAL
    c = linear % 480
    k = linear // 480
    n = k // 49
    hw = k - n * 49
    source, centered, bn_offset = _masked_source_from_add1(
        add1_ptr,
        bn_src_ptr,
        mean_ptr,
        invstd_ptr,
        weight_ptr,
        bias_ptr,
        fill_ptr,
        n,
        hw,
        c,
        active,
    )
    coeff = tl.load(coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    output_scale = tl.load(output_scale_ptr + c, mask=active, other=0.0).to(
        tl.float32
    )
    correction = _f32_mul(centered, coeff)
    corrected = _f32_sub(_f32_sub(source, correction), mean_term)
    out_value = _bf16_round_f32(_f32_mul(corrected, output_scale))
    tl.store(grad_out_ptr + bn_offset, out_value, mask=active)


@oracle_impl(
    hardware="B200",
    point="a1e026e9",
    PRODUCER_BLOCK=256,
    REDUCE_BLOCK_K=256,
    XBLOCK=8,
    EPILOGUE_BLOCK=256,
    BLOCK_TILES=128,
    reduce_warps=8,
)
def oracle_forward(
    inputs,
    *,
    PRODUCER_BLOCK: int,
    REDUCE_BLOCK_K: int,
    XBLOCK: int,
    EPILOGUE_BLOCK: int,
    BLOCK_TILES: int,
    reduce_warps: int,
):
    (
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        _shape_param_0,
    ) = inputs
    del _shape_param_0

    device = arg1_1.device
    add1_out = torch.empty_strided(
        (N, C_FULL, H, W),
        (C_FULL * HW, 1, W * C_FULL, C_FULL),
        device=device,
        dtype=torch.bfloat16,
    )
    sum_out = torch.empty((C,), device=device, dtype=torch.float32)
    scaled_dot_out = torch.empty((C,), device=device, dtype=torch.float32)
    num_tiles = triton.cdiv(K_TOTAL, REDUCE_BLOCK_K)
    partial_sum = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles * C,), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    coeff = torch.empty((C,), device=device, dtype=torch.float32)
    output_scale = torch.empty((C,), device=device, dtype=torch.float32)
    grad_out = torch.empty_strided(
        (N, C, H, W),
        (C * HW, 1, W * C, C),
        device=device,
        dtype=torch.bfloat16,
    )

    add1_total = N * C_FULL * HW
    _add1_full_kernel[(triton.cdiv(add1_total, PRODUCER_BLOCK),)](
        arg0_1,
        arg1_1,
        arg2_1,
        add1_out,
        TOTAL=add1_total,
        BLOCK=PRODUCER_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    _partial_reduce_kernel[(triton.cdiv(num_tiles * C, XBLOCK),)](
        add1_out,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        partial_sum,
        partial_dot,
        X_TOTAL=num_tiles * C,
        XBLOCK=XBLOCK,
        BLOCK_K=REDUCE_BLOCK_K,
        num_warps=reduce_warps,
        num_stages=3,
    )
    _finalize_kernel[(C,)](
        partial_sum,
        partial_dot,
        arg5_1,
        arg6_1,
        sum_out,
        scaled_dot_out,
        mean_term,
        coeff,
        output_scale,
        NUM_TILES=num_tiles,
        BLOCK_TILES=BLOCK_TILES,
        num_warps=4,
        num_stages=4,
    )
    _epilogue_kernel[(triton.cdiv(N * C * HW, EPILOGUE_BLOCK),)](
        add1_out,
        arg3_1,
        arg4_1,
        arg5_1,
        arg6_1,
        arg7_1,
        arg8_1,
        mean_term,
        coeff,
        output_scale,
        grad_out,
        TOTAL=N * C * HW,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
        num_stages=4,
    )
    return add1_out, sum_out, scaled_dot_out, grad_out
