"""Gap diagnosis (classification: SCATTER_REDUCE): this oracle computes the complete MobileNetV3 hard-sigmoid-gated average-pool-backward producer, sibling channel reductions, and dependent BN-backward dense output for all captured channels-last bf16 points. Inductor currently materializes the expanded pool-gradient/gate producer as a large f32 tensor, then reduces and rereads it in separate generic kernels; it cannot do this today because scheduler/codegen does not model the structured `[N,C,1,1]` pooled source and hard-sigmoid gate as a scatter-reduce producer feeding both channel summaries and the full tensor epilogue while preserving the lowered fp32 arithmetic and final bf16 store. The fix is SCATTER_REDUCE: add a structured average-pool-backward lowering that computes the producer directly from source coordinates, shares it across the two reductions, and emits the channels-last BN-backward output without materializing the producer tensor."""

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
def _producer(
    gate_ptr,
    gated_x_ptr,
    pool_ptr,
    act_ptr,
    fill_ptr,
    offsets,
    n,
    c,
    mask,
    C: tl.constexpr,
):
    act = tl.load(act_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    fill = tl.load(fill_ptr).to(tl.float32)

    gate_src = tl.load(gate_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gate = _f32_add(gate_src, 3.0)
    gate = _maximum(gate, 0.0)
    gate = _minimum(gate, 6.0)
    gate = _round_bf16_to_f32(_f32_mul(gate, 0.16666666666666666))

    gated_x = tl.load(gated_x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    pooled = tl.load(pool_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    gated = _round_bf16_to_f32(_f32_mul(gated_x, gate))
    pool_grad = _round_bf16_to_f32(_f32_mul(pooled, 0.00510204081632653))
    base = _round_bf16_to_f32(_f32_add(gated, pool_grad))

    middle_gate = _f32_add(_f32_mul(act, 0.3333333333333333), 0.5)
    middle = _f32_mul(base, middle_gate)
    out = tl.where(act < 3.0, middle, base)
    out = tl.where(act <= -3.0, fill, out)
    return _round_bf16_to_f32(out)


@triton.jit
def _partial_reduce_kernel(
    gate_ptr,
    gated_x_ptr,
    pool_ptr,
    act_ptr,
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

    producer = _producer(
        gate_ptr,
        gated_x_ptr,
        pool_ptr,
        act_ptr,
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

    sum_values = tl.where(mask, producer, 0.0)
    dot_values = tl.where(mask, _f32_mul(producer, centered), 0.0)
    partial_offsets = c * NUM_K_TILES + tl.program_id(1)

    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(sum_values, axis=0),
        mask=c < C,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(dot_values, axis=0),
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
    pool_ptr,
    act_ptr,
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

    producer = _producer(
        gate_ptr,
        gated_x_ptr,
        pool_ptr,
        act_ptr,
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

    mean_term = _f32_mul(sum_value, 9.964923469387754e-06)
    dot_scaled = _f32_mul(dot_value, 9.964923469387754e-06)
    invstd_sq = _f32_mul(invstd, invstd)
    var_term = _f32_mul(dot_scaled, invstd_sq)
    input_scale = _f32_mul(invstd, weight)
    without_var = _f32_sub(producer, _f32_mul(centered, var_term))
    without_mean = _f32_sub(without_var, mean_term)
    out = _f32_mul(without_mean, input_scale)

    tl.store(out_ptr + offsets, out, mask=mask)


@oracle_impl(
    hardware="B200",
    point="5816bf68",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="06aedd03",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="230be935",
    GROUP_K=98,
    BLOCK_K=128,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="212c1566",
    GROUP_K=98,
    BLOCK_K=128,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="d6f080d2",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="3468f7bd",
    GROUP_K=392,
    BLOCK_K=512,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=8,
)
@oracle_impl(
    hardware="B200",
    point="26a83dcc",
    GROUP_K=98,
    BLOCK_K=128,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=4,
)
@oracle_impl(
    hardware="B200",
    point="50dc27bc",
    GROUP_K=98,
    BLOCK_K=128,
    BLOCK_C=16,
    EPILOGUE_BLOCK=1024,
    num_warps=4,
)
def oracle_forward(
    inputs,
    *,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    EPILOGUE_BLOCK: int,
    num_warps: int,
):
    (
        gate,
        gated_x,
        pooled,
        act,
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
        act,
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
        num_warps=num_warps,
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
        act,
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
        num_warps=4,
        num_stages=4,
    )

    return sum_out, scaled_dot, grad_input
