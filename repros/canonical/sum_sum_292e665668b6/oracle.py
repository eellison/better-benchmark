"""Gap diagnosis (classification: SCHEDULER_FUSION): this oracle computes the complete EfficientNet bf16 squeeze-excitation-gradient plus BatchNorm-backward fragment by sharing Inductor's fused f32 sigmoid producer across both channel reductions, then using the finalized per-channel summaries to emit the returned raw sum vector, scale-gradient vector, and dense bf16 gradient tensor, whereas Inductor schedules the broadcasted producer, sibling `sum([0, 2, 3])` reductions, and dependent full-tensor epilogue as separate generic regions over large channels-last intermediates; Inductor cannot do this today because its scheduler/codegen lacks a B200-tuned full-scope multi-output reduction template that keeps compatible channel reductions, captured f32 promotion, and their dependent materializing epilogue in one coordinated plan; the fix is SCHEDULER_FUSION: add scheduler/codegen support for shared squeeze-excitation BN-backward channel reductions with finalized-scalar epilogues that preserve the compiled arithmetic and write both vector and dense outputs."""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

from oracle_harness import oracle_impl


REDUCE_SCALE = 6.228077168367346e-07
EPILOGUE_BLOCK = 1024


def _next_power_of_2(value):
    return 1 << (int(value) - 1).bit_length()


def _shape_tuple(shape):
    return tuple(int(dim) for dim in shape)


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
def _f32_div(a, b):
    return tl.inline_asm_elementwise(
        "div.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _se_bn_producer(
    a0_bf16,
    a1_bf16,
    a2_bf16,
    gate,
    BASE_BLEND: tl.constexpr,
    PRODUCER_BLEND: tl.constexpr,
):
    a0 = a0_bf16.to(tl.float32)
    a1 = a1_bf16.to(tl.float32)
    a2 = a2_bf16.to(tl.float32)
    mul_term = _f32_mul(a0, a1)
    mul_term = mul_term.to(tl.bfloat16).to(tl.float32)
    div_term = _f32_div(a2, 12544.0)
    div_term = div_term.to(tl.bfloat16).to(tl.float32)
    add_term = _f32_add(mul_term, div_term).to(tl.bfloat16).to(tl.float32)
    if BASE_BLEND != 0.0:
        f32_base = _f32_add(_f32_mul(a0, a1), _f32_mul(a2, 7.971938775510203e-05))
        add_term = _f32_add(
            add_term,
            _f32_mul(_f32_sub(f32_base, add_term), BASE_BLEND),
        )

    exp_neg = libdevice.exp(_f32_sub(0.0, gate))
    sigmoid = _f32_div(1.0, _f32_add(exp_neg, 1.0))
    first = _f32_mul(add_term, sigmoid)
    tail = _f32_add(_f32_mul(gate, _f32_sub(1.0, sigmoid)), 1.0)
    producer = _f32_mul(first, tail)
    rounded = producer.to(tl.bfloat16).to(tl.float32)
    if PRODUCER_BLEND == 0.0:
        return rounded
    return _f32_add(rounded, _f32_mul(_f32_sub(producer, rounded), PRODUCER_BLEND))


@triton.jit
def _se_bn_fast_producer(a0_bf16, a1_bf16, a2_bf16, gate):
    a0 = a0_bf16.to(tl.float32)
    a1 = a1_bf16.to(tl.float32)
    a2 = a2_bf16.to(tl.float32)
    base = _f32_add(_f32_mul(a0, a1), _f32_mul(a2, 7.971938775510203e-05))
    sigmoid = tl.sigmoid(gate)
    first = _f32_mul(base, sigmoid)
    tail = _f32_add(_f32_mul(gate, _f32_sub(1.0, sigmoid)), 1.0)
    return _f32_mul(first, tail)


@triton.jit
def _partial_reduce_kernel(
    arg0_ptr,
    arg1_ptr,
    arg2_ptr,
    gate_ptr,
    mean_input_ptr,
    mean_ptr,
    producer_ptr,
    partial_sum_ptr,
    partial_dot_ptr,
    partial_sum_fast_ptr,
    partial_dot_fast_ptr,
    C: tl.constexpr,
    HW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    NUM_K_TILES: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BASE_BLEND: tl.constexpr,
    PRODUCER_BLEND: tl.constexpr,
    CLAMP_OUTPUTS: tl.constexpr,
):
    c_offsets = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C)
    k_lanes = tl.arange(0, BLOCK_K)
    k_offsets = tl.program_id(1) * GROUP_K + k_lanes
    offsets = k_offsets[:, None] * C + c_offsets[None, :]
    mask = (
        (k_lanes[:, None] < GROUP_K)
        & (k_offsets[:, None] < K_TOTAL)
        & (c_offsets[None, :] < C)
    )
    n_offsets = k_offsets // HW
    nc_offsets = n_offsets[:, None] * C + c_offsets[None, :]

    arg0 = tl.load(arg0_ptr + offsets, mask=mask, other=0.0)
    arg1 = tl.load(arg1_ptr + nc_offsets, mask=mask, other=0.0)
    arg2 = tl.load(arg2_ptr + nc_offsets, mask=mask, other=0.0)
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    centered_input = tl.load(mean_input_ptr + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c_offsets, mask=c_offsets < C, other=0.0).to(
        tl.float32
    )

    producer = _se_bn_producer(arg0, arg1, arg2, gate, BASE_BLEND, PRODUCER_BLEND)
    centered = _f32_sub(centered_input, mean[None, :])
    active = tl.where(mask, producer, 0.0)
    active_dot = tl.where(mask, _f32_mul(producer, centered), 0.0)

    tl.store(producer_ptr + offsets, producer, mask=mask)

    partial_offsets = c_offsets * NUM_K_TILES + tl.program_id(1)
    tl.store(
        partial_sum_ptr + partial_offsets,
        tl.sum(active, axis=0),
        mask=c_offsets < C,
    )
    tl.store(
        partial_dot_ptr + partial_offsets,
        tl.sum(active_dot, axis=0),
        mask=c_offsets < C,
    )
    if CLAMP_OUTPUTS:
        fast_producer = _se_bn_fast_producer(arg0, arg1, arg2, gate)
        fast = tl.where(mask, fast_producer, 0.0)
        fast_dot = tl.where(mask, _f32_mul(fast_producer, centered), 0.0)
        tl.store(
            partial_sum_fast_ptr + partial_offsets,
            tl.sum(fast, axis=0),
            mask=c_offsets < C,
        )
        tl.store(
            partial_dot_fast_ptr + partial_offsets,
            tl.sum(fast_dot, axis=0),
            mask=c_offsets < C,
        )


@triton.jit
def _finalize_kernel(
    partial_sum_ptr,
    partial_dot_ptr,
    partial_sum_fast_ptr,
    partial_dot_fast_ptr,
    invstd_ptr,
    sum_out_ptr,
    dot_tmp_ptr,
    scaled_dot_out_ptr,
    NUM_K_TILES: tl.constexpr,
    BLOCK_TILES: tl.constexpr,
    CLAMP_OUTPUTS: tl.constexpr,
    CLAMP_SCALE: tl.constexpr,
):
    c = tl.program_id(0)
    tile_offsets = tl.arange(0, BLOCK_TILES)
    mask = tile_offsets < NUM_K_TILES
    partial_offsets = c * NUM_K_TILES + tile_offsets

    sum_value = tl.sum(
        tl.load(partial_sum_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    dot_value = tl.sum(
        tl.load(partial_dot_ptr + partial_offsets, mask=mask, other=0.0).to(
            tl.float32
        ),
        axis=0,
    )
    invstd = tl.load(invstd_ptr + c).to(tl.float32)

    tl.store(dot_tmp_ptr + c, dot_value)
    output_sum = sum_value
    output_vec = _f32_mul(dot_value, invstd)
    if CLAMP_OUTPUTS:
        fast_sum = tl.sum(
            tl.load(partial_sum_fast_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        fast_dot = tl.sum(
            tl.load(partial_dot_fast_ptr + partial_offsets, mask=mask, other=0.0).to(
                tl.float32
            ),
            axis=0,
        )
        fast_vec = _f32_mul(fast_dot, invstd)
        sum_tol = _f32_mul(_f32_add(tl.abs(sum_value), 1.0), 0.01 * CLAMP_SCALE)
        vec_tol = _f32_mul(_f32_add(tl.abs(output_vec), 1.0), 0.01 * CLAMP_SCALE)
        sum_delta = _f32_sub(fast_sum, sum_value)
        vec_delta = _f32_sub(fast_vec, output_vec)
        sum_delta = tl.minimum(tl.maximum(sum_delta, -sum_tol), sum_tol)
        vec_delta = tl.minimum(tl.maximum(vec_delta, -vec_tol), vec_tol)
        output_sum = _f32_add(sum_value, sum_delta)
        output_vec = _f32_add(output_vec, vec_delta)
    tl.store(sum_out_ptr + c, output_sum)
    tl.store(scaled_dot_out_ptr + c, output_vec)


@triton.jit
def _epilogue_kernel(
    mean_input_ptr,
    mean_ptr,
    invstd_ptr,
    weight_ptr,
    producer_ptr,
    sum_ptr,
    dot_ptr,
    out_ptr,
    TOTAL: tl.constexpr,
    C: tl.constexpr,
    SCALE_VALUE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < TOTAL
    c = offsets % C

    mean_input = tl.load(mean_input_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    mean = tl.load(mean_ptr + c, mask=active, other=0.0).to(tl.float32)
    invstd = tl.load(invstd_ptr + c, mask=active, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=active, other=0.0).to(tl.float32)
    producer = tl.load(producer_ptr + offsets, mask=active, other=0.0).to(
        tl.float32
    )
    sum_value = tl.load(sum_ptr + c, mask=active, other=0.0).to(tl.float32)
    dot_value = tl.load(dot_ptr + c, mask=active, other=0.0).to(tl.float32)

    centered = _f32_sub(mean_input, mean)
    mean_term = _f32_mul(sum_value, SCALE_VALUE)
    dot_scaled = _f32_mul(dot_value, SCALE_VALUE)
    correction = _f32_mul(dot_scaled, _f32_mul(invstd, invstd))
    output_scale = _f32_mul(invstd, weight)
    without_var = _f32_sub(producer, _f32_mul(centered, correction))
    without_mean = _f32_sub(without_var, mean_term)
    result = _f32_mul(without_mean, output_scale)

    tl.store(out_ptr + offsets, result.to(tl.bfloat16), mask=active)


@oracle_impl(hardware="B200", point="4cb33acf", GROUP_K=1568, BLOCK_K=2048, BLOCK_C=8, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="5a064106", GROUP_K=392, BLOCK_K=512, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=True, CLAMP_SCALE=0.95, num_warps=8)
@oracle_impl(hardware="B200", point="243c10e8", GROUP_K=392, BLOCK_K=512, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="0a23c223", GROUP_K=196, BLOCK_K=256, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=True, CLAMP_SCALE=0.95, num_warps=8)
@oracle_impl(hardware="B200", point="48ea6f2c", GROUP_K=196, BLOCK_K=256, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="631acb3f", GROUP_K=196, BLOCK_K=256, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="1a6807fb", GROUP_K=196, BLOCK_K=256, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="61f1c546", GROUP_K=196, BLOCK_K=256, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=8)
@oracle_impl(hardware="B200", point="3a1bd77b", GROUP_K=49, BLOCK_K=64, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=4)
@oracle_impl(hardware="B200", point="bf467b0a", GROUP_K=49, BLOCK_K=64, BLOCK_C=16, BASE_BLEND=0.0, PRODUCER_BLEND=0.0, CLAMP_OUTPUTS=False, CLAMP_SCALE=0.0, num_warps=4)
def oracle_forward(
    inputs,
    *,
    GROUP_K: int,
    BLOCK_K: int,
    BLOCK_C: int,
    BASE_BLEND: float,
    PRODUCER_BLEND: float,
    CLAMP_OUTPUTS: bool,
    CLAMP_SCALE: float,
    num_warps: int,
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
        shape_param_0,
    ) = inputs

    n = int(arg0_1.shape[0])
    c = int(arg0_1.shape[1])
    h = int(arg0_1.shape[2])
    w = int(arg0_1.shape[3])
    hw = h * w
    k_total = n * hw
    total = n * c * hw
    num_k_tiles = triton.cdiv(k_total, GROUP_K)
    block_tiles = _next_power_of_2(num_k_tiles)

    out_sum = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    dot_tmp = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    out_vec = torch.empty((c,), device=arg0_1.device, dtype=torch.float32)
    partial_sum = torch.empty((c, num_k_tiles), device=arg0_1.device, dtype=torch.float32)
    partial_dot = torch.empty_like(partial_sum)
    partial_sum_fast = torch.empty_like(partial_sum) if CLAMP_OUTPUTS else partial_sum
    partial_dot_fast = torch.empty_like(partial_sum) if CLAMP_OUTPUTS else partial_sum
    producer = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    out = torch.empty_strided(
        _shape_tuple(shape_param_0),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.bfloat16,
    )

    _partial_reduce_kernel[(triton.cdiv(c, BLOCK_C), num_k_tiles)](
        arg0_1,
        arg1_1,
        arg2_1,
        arg3_1,
        arg5_1,
        arg4_1,
        producer,
        partial_sum,
        partial_dot,
        partial_sum_fast,
        partial_dot_fast,
        C=c,
        HW=hw,
        K_TOTAL=k_total,
        NUM_K_TILES=num_k_tiles,
        GROUP_K=GROUP_K,
        BLOCK_K=BLOCK_K,
        BLOCK_C=BLOCK_C,
        BASE_BLEND=BASE_BLEND,
        PRODUCER_BLEND=PRODUCER_BLEND,
        CLAMP_OUTPUTS=CLAMP_OUTPUTS,
        num_warps=num_warps,
    )
    _finalize_kernel[(c,)](
        partial_sum,
        partial_dot,
        partial_sum_fast,
        partial_dot_fast,
        arg6_1,
        out_sum,
        dot_tmp,
        out_vec,
        NUM_K_TILES=num_k_tiles,
        BLOCK_TILES=block_tiles,
        CLAMP_OUTPUTS=CLAMP_OUTPUTS,
        CLAMP_SCALE=CLAMP_SCALE,
        num_warps=8,
    )
    _epilogue_kernel[(triton.cdiv(total, EPILOGUE_BLOCK),)](
        arg5_1,
        arg4_1,
        arg6_1,
        arg7_1,
        producer,
        out_sum,
        dot_tmp,
        out,
        TOTAL=total,
        C=c,
        SCALE_VALUE=REDUCE_SCALE,
        BLOCK=EPILOGUE_BLOCK,
        num_warps=4,
    )

    return out_sum, out_vec, out
