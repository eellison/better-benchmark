"""Gap diagnosis (classification: COOPERATIVE_SPLIT_K): this oracle computes the complete GhostNet dual BatchNorm-backward scope, including the visible channels-last bf16 copy of `arg0 + arg1`, both per-channel sum/product reductions, and both reduction-dependent bf16 gradient epilogues, whereas Inductor emits separate generic reductions and pointwise kernels around the layout-changing copy/clone/slice chain; Inductor cannot do this today because its scheduler does not form one cooperative split-K plan that preserves the shared bf16 materialization boundary while feeding sibling reductions and dense epilogues; the fix is COOPERATIVE_SPLIT_K: teach Inductor to tile the N/H/W reduction domain once for compatible channel reductions, co-finalize the channel summaries, and sink those summaries into the dependent tensor epilogues."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 512
C0 = 160
C1 = 80
SLICE_START = 80
H = 7
W = 7
HW = H * W
SPATIAL = N * HW
FIRST_TILES = 196
SECOND_TILES = 196
SCALE = 3.985969387755102e-05


@triton.jit
def _first_partials_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    mean_ptr,
    copy_out_ptr,
    prod_partials_ptr,
    sum_partials_ptr,
    C_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)[:, None]
    r = tl.arange(0, R_BLOCK)[None, :]
    tile = tl.program_id(1)
    spatial = tile * 128 + r
    mask = (c < 160) & (spatial < 25088)

    hw = spatial % 49
    n = spatial // 49
    contiguous = n * 7840 + c * 49 + hw
    channels_last = c + 160 * spatial

    lhs = tl.load(arg0_ptr + contiguous, mask=mask, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + channels_last, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg2_ptr + channels_last, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < 160, other=0.0).to(tl.float32)

    summed = (lhs + rhs).to(tl.bfloat16).to(tl.float32)
    tl.store(copy_out_ptr + channels_last, summed, mask=mask)
    centered = activation - mean
    product = summed * centered

    prod_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)
    x_sum = tl.sum(tl.where(mask, summed, 0.0), axis=1)

    c_flat = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    out_mask = c_flat < 160
    partial_offset = c_flat * 196 + tile
    tl.store(prod_partials_ptr + partial_offset, prod_sum, mask=out_mask)
    tl.store(sum_partials_ptr + partial_offset, x_sum, mask=out_mask)


@triton.jit
def _first_finalize_kernel(
    prod_partials_ptr,
    sum_partials_ptr,
    weight_ptr,
    bias_ptr,
    sum_out_ptr,
    mul_out_ptr,
    stats_ptr,
    C_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)[:, None]
    t = tl.arange(0, T_BLOCK)[None, :]
    mask = (c < 160) & (t < 196)
    offsets = c * 196 + t

    prod_values = tl.load(prod_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_values = tl.load(sum_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    prod_sum = tl.sum(prod_values, axis=1)
    x_sum = tl.sum(sum_values, axis=1)

    c_flat = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    c_mask = c_flat < 160
    weight = tl.load(weight_ptr + c_flat, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_flat, mask=c_mask, other=0.0).to(tl.float32)
    mean_term = x_sum * 3.985969387755102e-05
    prod_coeff = (prod_sum * 3.985969387755102e-05) * (weight * weight)
    output_scale = weight * bias

    tl.store(sum_out_ptr + c_flat, x_sum, mask=c_mask)
    tl.store(mul_out_ptr + c_flat, prod_sum * weight, mask=c_mask)
    tl.store(stats_ptr + c_flat, mean_term, mask=c_mask)
    tl.store(stats_ptr + 160 + c_flat, prod_coeff, mask=c_mask)
    tl.store(stats_ptr + 320 + c_flat, output_scale, mask=c_mask)


@triton.jit
def _first_epilogue_kernel(
    copy_ptr,
    arg2_ptr,
    mean_ptr,
    stats_ptr,
    grad_out_ptr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < 4014080

    hw = linear % 49
    c = (linear // 49) % 160
    n = linear // 7840
    spatial = n * 49 + hw
    channels_last = c + 160 * spatial

    summed = tl.load(copy_ptr + channels_last, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(arg2_ptr + channels_last, mask=mask, other=0.0).to(tl.float32)
    centered = activation - tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    prod_coeff = tl.load(stats_ptr + 160 + c, mask=mask, other=0.0).to(tl.float32)
    output_scale = tl.load(stats_ptr + 320 + c, mask=mask, other=0.0).to(tl.float32)
    out = ((summed - centered * prod_coeff) - mean_term) * output_scale
    tl.store(grad_out_ptr + linear, out, mask=mask)


@triton.jit
def _second_partials_kernel(
    copy_ptr,
    activation_ptr,
    mean_ptr,
    prod_partials_ptr,
    sum_partials_ptr,
    C_BLOCK: tl.constexpr,
    R_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)[:, None]
    r = tl.arange(0, R_BLOCK)[None, :]
    tile = tl.program_id(1)
    spatial = tile * 128 + r
    mask = (c < 80) & (spatial < 25088)

    copy_offsets = 80 + c + 160 * spatial
    activation_offsets = c + 80 * spatial
    x = tl.load(copy_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + activation_offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=c < 80, other=0.0).to(tl.float32)
    centered = activation - mean
    product = x * centered

    prod_sum = tl.sum(tl.where(mask, product, 0.0), axis=1)
    x_sum = tl.sum(tl.where(mask, x, 0.0), axis=1)

    c_flat = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    out_mask = c_flat < 80
    partial_offset = c_flat * 196 + tile
    tl.store(prod_partials_ptr + partial_offset, prod_sum, mask=out_mask)
    tl.store(sum_partials_ptr + partial_offset, x_sum, mask=out_mask)


@triton.jit
def _second_finalize_kernel(
    prod_partials_ptr,
    sum_partials_ptr,
    weight_ptr,
    bias_ptr,
    sum_out_ptr,
    mul_out_ptr,
    stats_ptr,
    C_BLOCK: tl.constexpr,
    T_BLOCK: tl.constexpr,
):
    c = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)[:, None]
    t = tl.arange(0, T_BLOCK)[None, :]
    mask = (c < 80) & (t < 196)
    offsets = c * 196 + t

    prod_values = tl.load(prod_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    sum_values = tl.load(sum_partials_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    prod_sum = tl.sum(prod_values, axis=1)
    x_sum = tl.sum(sum_values, axis=1)

    c_flat = tl.program_id(0) * C_BLOCK + tl.arange(0, C_BLOCK)
    c_mask = c_flat < 80
    weight = tl.load(weight_ptr + c_flat, mask=c_mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_flat, mask=c_mask, other=0.0).to(tl.float32)
    mean_term = x_sum * 3.985969387755102e-05
    prod_coeff = (prod_sum * 3.985969387755102e-05) * (weight * weight)
    output_scale = weight * bias

    tl.store(sum_out_ptr + c_flat, x_sum, mask=c_mask)
    tl.store(mul_out_ptr + c_flat, prod_sum * weight, mask=c_mask)
    tl.store(stats_ptr + c_flat, mean_term, mask=c_mask)
    tl.store(stats_ptr + 80 + c_flat, prod_coeff, mask=c_mask)
    tl.store(stats_ptr + 160 + c_flat, output_scale, mask=c_mask)


@triton.jit
def _second_epilogue_kernel(
    copy_ptr,
    activation_ptr,
    mean_ptr,
    stats_ptr,
    grad_out_ptr,
    BLOCK: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = linear < 2007040

    c = linear % 80
    spatial = linear // 80
    copy_offsets = 80 + c + 160 * spatial

    x = tl.load(copy_ptr + copy_offsets, mask=mask, other=0.0).to(tl.float32)
    activation = tl.load(activation_ptr + linear, mask=mask, other=0.0).to(tl.float32)
    centered = activation - tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    mean_term = tl.load(stats_ptr + c, mask=mask, other=0.0).to(tl.float32)
    prod_coeff = tl.load(stats_ptr + 80 + c, mask=mask, other=0.0).to(tl.float32)
    output_scale = tl.load(stats_ptr + 160 + c, mask=mask, other=0.0).to(tl.float32)
    out = ((x - centered * prod_coeff) - mean_term) * output_scale
    tl.store(grad_out_ptr + linear, out, mask=mask)


@oracle_impl(
    hardware="B200",
    point="f96ef00e",
    FIRST_C_BLOCK=16,
    SECOND_C_BLOCK=16,
    EPILOGUE_BLOCK=1024,
    first_warps=4,
    second_warps=4,
)
def oracle_forward(
    inputs,
    *,
    FIRST_C_BLOCK: int,
    SECOND_C_BLOCK: int,
    EPILOGUE_BLOCK: int,
    first_warps: int,
    second_warps: int,
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
        arg9_1,
        _shape_param_0,
        _shape_param_1,
    ) = inputs
    del _shape_param_0, _shape_param_1

    device = arg0_1.device
    first_prod = torch.empty_strided((C0, FIRST_TILES), (FIRST_TILES, 1), device=device, dtype=torch.float32)
    first_sum = torch.empty_strided((C0, FIRST_TILES), (FIRST_TILES, 1), device=device, dtype=torch.float32)
    sum_1 = torch.empty_strided((C0,), (1,), device=device, dtype=torch.float32)
    mul_8 = torch.empty_strided((C0,), (1,), device=device, dtype=torch.float32)
    first_stats = torch.empty_strided((3, C0), (C0, 1), device=device, dtype=torch.float32)
    copy_1 = torch.empty_strided(
        (N, C0, H, W),
        (C0 * HW, 1, W * C0, C0),
        device=device,
        dtype=torch.bfloat16,
    )
    grad_1 = torch.empty_strided(
        (N, C0, H, W),
        (C0 * HW, HW, W, 1),
        device=device,
        dtype=torch.bfloat16,
    )

    _first_partials_kernel[(triton.cdiv(C0, FIRST_C_BLOCK), FIRST_TILES)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        copy_1,
        first_prod,
        first_sum,
        C_BLOCK=FIRST_C_BLOCK,
        R_BLOCK=128,
        num_warps=first_warps,
    )
    _first_finalize_kernel[(triton.cdiv(C0, FIRST_C_BLOCK),)](
        first_prod,
        first_sum,
        arg4_1,
        arg5_1,
        sum_1,
        mul_8,
        first_stats,
        C_BLOCK=FIRST_C_BLOCK,
        T_BLOCK=256,
        num_warps=4,
    )
    _first_epilogue_kernel[(triton.cdiv(N * C0 * HW, EPILOGUE_BLOCK),)](
        copy_1,
        arg2_1,
        arg3_1,
        first_stats,
        grad_1,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )

    second_prod = torch.empty_strided((C1, SECOND_TILES), (SECOND_TILES, 1), device=device, dtype=torch.float32)
    second_sum = torch.empty_strided((C1, SECOND_TILES), (SECOND_TILES, 1), device=device, dtype=torch.float32)
    sum_3 = torch.empty_strided((C1,), (1,), device=device, dtype=torch.float32)
    mul_17 = torch.empty_strided((C1,), (1,), device=device, dtype=torch.float32)
    second_stats = torch.empty_strided((3, C1), (C1, 1), device=device, dtype=torch.float32)
    grad_2 = torch.empty_strided(
        (N, C1, H, W),
        (C1 * HW, 1, W * C1, C1),
        device=device,
        dtype=torch.bfloat16,
    )

    _second_partials_kernel[(triton.cdiv(C1, SECOND_C_BLOCK), SECOND_TILES)](
        copy_1,
        arg6_1,
        arg7_1,
        second_prod,
        second_sum,
        C_BLOCK=SECOND_C_BLOCK,
        R_BLOCK=128,
        num_warps=second_warps,
    )
    _second_finalize_kernel[(triton.cdiv(C1, SECOND_C_BLOCK),)](
        second_prod,
        second_sum,
        arg8_1,
        arg9_1,
        sum_3,
        mul_17,
        second_stats,
        C_BLOCK=SECOND_C_BLOCK,
        T_BLOCK=256,
        num_warps=4,
    )
    _second_epilogue_kernel[(triton.cdiv(N * C1 * HW, EPILOGUE_BLOCK),)](
        copy_1,
        arg6_1,
        arg7_1,
        second_stats,
        grad_2,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return copy_1, sum_1, mul_8, grad_1, sum_3, mul_17, grad_2
