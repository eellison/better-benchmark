"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the full MobileNetV3 gate/average-pool/ReLU-mask producer, both fp32 channel reductions, the fp32 side vector, and the returned channels-last bf16 BN-backward epilogue in one shared reduction plan while preserving Inductor's compiled f32 reduction producer and bf16 dense-output cast boundaries, whereas Inductor schedules the sibling reductions plus dependent epilogue as separate generic kernels; Inductor cannot do this today because its scheduler does not form a full-scope multi-output channel-reduction lowering that shares this live producer across compatible sums and sinks the finalized channel scalars into the dense epilogue; the fix is SCHEDULER_FUSION: teach reduction scheduling to fuse sibling channel reductions with their structured producer and dependent full-tensor epilogue while preserving the compiled cast boundaries."""

import torch
import triton
import triton.language as tl

from oracle_harness import oracle_impl


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


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
def _minimum(a, b):
    mask = a < b
    mask = mask | (a != a)
    return tl.where(mask, a, b)


@triton.jit
def _maximum(a, b):
    mask = a > b
    mask = mask | (a != a)
    return tl.where(mask, a, b)


@triton.jit
def _round_bf16_to_f32(value):
    return value.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)


@triton.jit
def _producer_f32(
    gate_ptr,
    gated_x_ptr,
    pooled_ptr,
    relu_mask_ptr,
    fill_ptr,
    offsets,
    n,
    c,
    mask,
    C: tl.constexpr,
):
    relu_src = tl.load(relu_mask_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    gate_src = tl.load(gate_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gate = _f32_add(gate_src, 3.0)
    gate = _maximum(gate, 0.0)
    gate = _minimum(gate, 6.0)
    gate = _f32_mul(gate, 0.16666666666666666)

    gated_x = tl.load(gated_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.load(pooled_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    base = _f32_add(_f32_mul(gated_x, gate), _f32_mul(pooled, 0.0012755102040816326))
    return tl.where(relu_src <= 0.0, fill, base)


@triton.jit
def _producer_bf16(
    gate_ptr,
    gated_x_ptr,
    pooled_ptr,
    relu_mask_ptr,
    fill_ptr,
    offsets,
    n,
    c,
    mask,
    C: tl.constexpr,
):
    relu_src = tl.load(relu_mask_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    gate_src = tl.load(gate_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gate = _f32_add(gate_src, 3.0)
    gate = _maximum(gate, 0.0)
    gate = _minimum(gate, 6.0)
    gate = _round_bf16_to_f32(_f32_mul(gate, 0.16666666666666666))

    gated_x = tl.load(gated_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.load(pooled_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gated = _round_bf16_to_f32(_f32_mul(gated_x, gate))
    pool_grad = _round_bf16_to_f32(_f32_mul(pooled, 0.0012755102040816326))
    base = _round_bf16_to_f32(_f32_add(gated, pool_grad))
    return tl.where(relu_src <= 0.0, fill, base)


@triton.jit
def _partial_reduce_kernel(
    gate_ptr,
    gated_x_ptr,
    pooled_ptr,
    relu_mask_ptr,
    fill_ptr,
    mean_ptr,
    centered_x_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_lanes = tl.arange(0, BLOCK_K)
    k = tl.program_id(1) * GROUP_K + k_lanes
    mask = (
        (k_lanes[:, None] < GROUP_K)
        & (k[:, None] < K_TOTAL)
        & (c[None, :] < C)
    )
    n = k // HW
    offsets = k[:, None] * C + c[None, :]

    producer = _producer_f32(
        gate_ptr,
        gated_x_ptr,
        pooled_ptr,
        relu_mask_ptr,
        fill_ptr,
        offsets,
        n[:, None],
        c[None, :],
        mask,
        C,
    )
    mean = tl.load(mean_ptr + c, mask=c < C, other=0.0).to(tl.float32)
    centered_x = tl.load(centered_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(centered_x, mean[None, :])
    dot = _f32_mul(producer, centered)

    partial_offsets = c * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(tl.where(mask, producer, 0.0), axis=0),
        mask=c < C,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(tl.where(mask, dot, 0.0), axis=0),
        mask=c < C,
    )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scaled_dot_out_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
):
    c = tl.program_id(0)
    tiles = tl.arange(0, BLOCK_TILES)
    mask = tiles < NUM_K_TILES
    offsets = c * NUM_K_TILES + tiles

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + offsets, mask=mask, other=0.0).to(tl.float32),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(sum_out_ptr + c, sum_value)
    tl.store(dot_tmp_ptr + c, dot_value)
    tl.store(scaled_dot_out_ptr + c, _f32_mul(dot_value, invstd))


@triton.jit
def _epilogue_kernel(
    gate_ptr,
    gated_x_ptr,
    pooled_ptr,
    relu_mask_ptr,
    fill_ptr,
    mean_ptr,
    centered_x_ptr,
    invstd_ptr,
    weight_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    c = offsets % C
    k = offsets // C
    n = k // HW

    producer = _producer_bf16(
        gate_ptr,
        gated_x_ptr,
        pooled_ptr,
        relu_mask_ptr,
        fill_ptr,
        offsets,
        n,
        c,
        mask,
        C,
    )
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    centered_x = tl.load(centered_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered = _f32_sub(centered_x, mean)

    invstd = tl.load(invstd_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=0.0).to(tl.float32)
    sum_value = tl.load(sum_ptr + c, mask=mask, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=mask, other=0.0).to(tl.float32)

    mean_term = _f32_mul(sum_value, 2.4912308673469386e-06)
    dot_scaled = _f32_mul(dot_value, 2.4912308673469386e-06)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(dot_scaled, invstd_sq)
    input_scale = _f32_mul(invstd, weight)
    without_var = _f32_sub(producer, _f32_mul(centered, var_term))
    without_mean = _f32_sub(without_var, mean_term)
    out = _f32_mul(without_mean, input_scale)

    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(
    hardware="B200",
    point="e664460a",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    epilogue_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="28bbb717",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=8,
    epilogue_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="8f42374e",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=4,
    epilogue_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="a6881a60",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    reduce_warps=4,
    epilogue_warps=4,
)
def oracle_forward(
    inputs,
    *,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    reduce_warps: int,
    epilogue_warps: int,
):
    (
        gate,
        gated_x,
        pooled,
        relu_mask,
        fill,
        mean,
        centered_x,
        invstd,
        weight,
        _shape,
    ) = inputs
    n, c, h, w = gated_x.shape
    hw = h * w
    k_total = n * hw
    total = k_total * c
    num_k_tiles = triton.cdiv(k_total, GROUP_K)

    sum_out = torch.empty_strided((c,), (1,), device=gated_x.device, dtype=torch.float32)
    dot_tmp = torch.empty_strided((c,), (1,), device=gated_x.device, dtype=torch.float32)
    scaled_dot = torch.empty_strided((c,), (1,), device=gated_x.device, dtype=torch.float32)
    partial_sum = torch.empty_strided(
        (c, num_k_tiles),
        (num_k_tiles, 1),
        device=gated_x.device,
        dtype=torch.float32,
    )
    partial_dot = torch.empty_like(partial_sum)
    grad_input = torch.empty_strided(
        tuple(gated_x.shape),
        tuple(gated_x.stride()),
        device=gated_x.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        gate,
        gated_x,
        pooled,
        relu_mask,
        fill,
        mean,
        centered_x,
        partial_sum,
        partial_dot,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        NUM_K_TILES=num_k_tiles,
        GROUP_K=GROUP_K,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        num_warps=reduce_warps,
        num_stages=4,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        invstd,
        sum_out,
        dot_tmp,
        scaled_dot,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=_next_power_of_2(num_k_tiles),
        num_warps=8,
        num_stages=1,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        gate,
        gated_x,
        pooled,
        relu_mask,
        fill,
        mean,
        centered_x,
        invstd,
        weight,
        sum_out,
        dot_tmp,
        grad_input,
        C=c,
        HW=hw,
        TOTAL=total,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=epilogue_warps,
        num_stages=4,
    )

    return sum_out, scaled_dot, grad_input
