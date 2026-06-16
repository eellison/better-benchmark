"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle materializes the returned bf16 masked-add side output once, shares that producer across the duplicated channel sums, centered-product sum, and dependent bf16 batch-norm-backward epilogue for every returned output, whereas Inductor schedules the masked bf16 producer, sibling channel reductions, vector epilogues, and full-tensor epilogue as separate generic regions over materialized intermediates; Inductor cannot do this today because its scheduler/codegen has no full-scope mixed-dtype channel-reduction fusion that preserves a returned bf16 side output while sharing finalized per-channel summaries with a later broadcast epilogue; the fix is SCHEDULER_FUSION: add a scheduler lowering for masked channel reductions with explicit bf16 producer stores, sibling vector finalization, and dependent full-tensor epilogue codegen."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


N = 128
SCALE = 3.0517578125e-05


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


@triton.jit
def _materialize_and_partial_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    scalar_ptr,
    arg4_ptr,
    mean_ptr,
    where_out_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    c = tl.program_id(0)
    tile = tl.program_id(1)
    k = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    active = k < K_TOTAL
    n = k // HW
    hw = k - n * HW
    offsets = n * (C * HW) + c * HW + hw

    lhs = tl.load(arg0_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    rhs = tl.load(arg1_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    mask_src = tl.load(arg2_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    full_value = tl.load(scalar_ptr).to(tl.bfloat16)
    add_bf16 = (lhs + rhs).to(tl.bfloat16)
    where_bf16 = tl.where(mask_src <= 0.0, full_value, add_bf16)
    where_f32 = where_bf16.to(tl.float32)
    tl.store(where_out_ptr + offsets, where_bf16, mask=active)

    centered = tl.load(arg4_ptr + offsets, mask=active, other=0.0).to(tl.float32)
    centered -= tl.load(mean_ptr + c).to(tl.float32)
    where_active = tl.where(active, where_f32, 0.0)
    centered_active = tl.where(active, centered, 0.0)
    partial_offset = tile * C + c
    tl.store(partial_sum_ptr + partial_offset, tl.sum(where_active, axis=0))
    tl.store(
        partial_dot_ptr + partial_offset,
        tl.sum(where_active * centered_active, axis=0),
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    weight_ptr,
    out_bf16_sum_ptr,
    out_sum_ptr,
    out_scaled_dot_ptr,
    mean_term_ptr,
    coeff_ptr,
    out_scale_ptr,
    C: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    SCALE_: tl.constexpr,
):
    c = tl.arange(0, BLOCK_C)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = (c[None, :] < C) & (tiles[:, None] < NUM_TILES)
    offsets = tiles[:, None] * C + c[None, :]
    sum_where = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    sum_dot = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )

    c_mask = c < C
    invstd = tl.load(invstd_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    tl.store(out_bf16_sum_ptr + c, sum_where, mask=c_mask)
    tl.store(out_sum_ptr + c, sum_where, mask=c_mask)
    tl.store(out_scaled_dot_ptr + c, sum_dot * invstd, mask=c_mask)
    tl.store(mean_term_ptr + c, sum_where * SCALE_, mask=c_mask)
    tl.store(coeff_ptr + c, sum_dot * SCALE_ * invstd * invstd, mask=c_mask)
    tl.store(out_scale_ptr + c, invstd * weight, mask=c_mask)


@triton.jit
def _epilogue_kernel(
    where_out_ptr,
    arg4_ptr,
    mean_ptr,
    mean_term_ptr,
    coeff_ptr,
    out_scale_ptr,
    grad_out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    linear = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    active = linear < NUMEL
    c = (linear // HW) % C

    where_f32 = tl.load(where_out_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered = tl.load(arg4_ptr + linear, mask=active, other=0.0).to(tl.float32)
    centered -= tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    mean_term = tl.load(mean_term_ptr + c, mask=active, other=0.0).to(tl.float32)
    coeff = tl.load(coeff_ptr + c, mask=active, other=0.0).to(tl.float32)
    out_scale = tl.load(out_scale_ptr + c, mask=active, other=0.0).to(tl.float32)
    out = (where_f32 - centered * coeff - mean_term) * out_scale
    tl.store(grad_out_ptr + linear, out, mask=active)


def _run(
    arg0,
    arg1,
    arg2,
    scalar,
    arg4,
    mean,
    invstd,
    weight,
    *,
    C,
    H,
    W,
    BLOCK_K,
    BLOCK_E,
    FINAL_BLOCK_C,
    REDUCE_WARPS,
    EPILOGUE_WARPS,
):
    hw = H * W
    k_total = N * hw
    numel = N * C * hw
    num_tiles = triton.cdiv(k_total, BLOCK_K)
    device = arg0.device

    where_out = torch.empty_strided((N, C, H, W), (C * hw, hw, W, 1), device=device, dtype=torch.bfloat16)
    out_bf16_sum = torch.empty((C,), device=device, dtype=torch.float32)
    out_sum = torch.empty((C,), device=device, dtype=torch.float32)
    out_scaled_dot = torch.empty((C,), device=device, dtype=torch.float32)
    grad_out = torch.empty_strided((N, C, H, W), (C * hw, hw, W, 1), device=device, dtype=torch.bfloat16)
    partial_sum = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    partial_dot = torch.empty((num_tiles, C), device=device, dtype=torch.float32)
    mean_term = torch.empty((C,), device=device, dtype=torch.float32)
    coeff = torch.empty((C,), device=device, dtype=torch.float32)
    out_scale = torch.empty((C,), device=device, dtype=torch.float32)

    _materialize_and_partial_kernel[(C, num_tiles)](
        arg0,
        arg1,
        arg2,
        scalar,
        arg4,
        mean,
        where_out,
        partial_sum,
        partial_dot,
        C=C,
        HW=hw,
        K_TOTAL=k_total,
        BLOCK_K=BLOCK_K,
        num_warps=REDUCE_WARPS,
        num_stages=4,
    )
    _finalize_kernel[(1,)](
        partial_sum,
        partial_dot,
        invstd,
        weight,
        out_bf16_sum,
        out_sum,
        out_scaled_dot,
        mean_term,
        coeff,
        out_scale,
        C=C,
        NUM_TILES=num_tiles,
        BLOCK_C=FINAL_BLOCK_C,
        BLOCK_TILES=_next_power_of_2(num_tiles),
        SCALE_=SCALE,
        num_warps=1,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(numel, BLOCK_E),)](
        where_out,
        arg4,
        mean,
        mean_term,
        coeff,
        out_scale,
        grad_out,
        C=C,
        HW=hw,
        NUMEL=numel,
        BLOCK_E=BLOCK_E,
        num_warps=EPILOGUE_WARPS,
        num_stages=4,
    )
    return where_out, out_bf16_sum, out_sum, out_scaled_dot, grad_out


@oracle_impl(
    hardware="B200",
    point="8addce5e",
    C=32,
    H=16,
    W=16,
    BLOCK_K=1024,
    BLOCK_E=256,
    FINAL_BLOCK_C=32,
    REDUCE_WARPS=4,
    EPILOGUE_WARPS=4,
)
@oracle_impl(
    hardware="B200",
    point="d10f1ba2",
    C=64,
    H=8,
    W=8,
    BLOCK_K=1024,
    BLOCK_E=256,
    FINAL_BLOCK_C=64,
    REDUCE_WARPS=4,
    EPILOGUE_WARPS=4,
)
def oracle_forward(
    inputs,
    *,
    C,
    H,
    W,
    BLOCK_K,
    BLOCK_E,
    FINAL_BLOCK_C,
    REDUCE_WARPS,
    EPILOGUE_WARPS,
):
    return _run(
        *inputs,
        C=C,
        H=H,
        W=W,
        BLOCK_K=BLOCK_K,
        BLOCK_E=BLOCK_E,
        FINAL_BLOCK_C=FINAL_BLOCK_C,
        REDUCE_WARPS=REDUCE_WARPS,
        EPILOGUE_WARPS=EPILOGUE_WARPS,
    )
